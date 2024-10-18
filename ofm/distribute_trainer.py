from functools import wraps
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import copy
import os
import time
import numpy as np
from .utils import EarlyStopping, step_lr, Logger
from .modeling_ofm import OFM
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .trainer import Trainer, TrainingArguments


def get_optimizer_and_scheduler(model, lr):
    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Define a custom scheduler
    def lr_lambda(current_step: int):
        # Custom learning rate decay
        return max(0.1, 0.975**current_step)

    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


class DistributedTrainer(Trainer):
    def __init__(
        self,
        supernet: OFM,
        args: TrainingArguments,
        data_collator,
        compute_metrics,
        train_dataset,
        eval_dataset=None,
        test_dataset=None,
        tokenizer=None,
        optimizers=None,
    ):

        self.setup()
        super().__init__(
            supernet,
            args,
            data_collator,
            compute_metrics,
            train_dataset,
            eval_dataset,
            test_dataset,
            tokenizer,
            optimizers,
        )

        self.device = torch.device("cuda:{}".format(self.local_rank))

    def setup(self):
        dist.init_process_group(backend="nccl")
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()
        print(f"Rank {self.local_rank} initialized, world size: {
              self.world_size}")

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    @wraps(Trainer.get_train_dataloader)
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            # num_workers=self.args.dataloader_num_workers,
            # pin_memory=True,
            sampler=torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, shuffle=True
            ),
        )

    @wraps(Trainer.get_eval_dataloader)
    def get_eval_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            sampler=torch.utils.data.distributed.DistributedSampler(
                self.eval_dataset, num_replicas=self.world_size, rank=self.local_rank
            ),
        )

    @wraps(Trainer.get_test_dataloader)
    def get_test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            sampler=torch.utils.data.distributed.DistributedSampler(
                self.test_dataset, num_replicas=self.world_size, rank=self.local_rank
            ),
        )

    @wraps(Trainer.evaluate)
    def evaluate(self, eval_dataloader):
        self.activate_model.eval()
        all_preds = []
        all_labels = []
        eval_preds = {}
        metrics = None
        for batch in eval_dataloader:
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.activate_model(**batch)
                preds = outputs.logits.detach()
                all_preds.append(preds.cpu())
                all_labels.append(batch["labels"].detach().cpu())
        batch.clear()
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # gathered_preds = [
        #     torch.ones_like(all_preds).to(self.device) for _ in range(self.world_size)
        # ]
        # gathered_labels = [
        #     torch.ones_like(all_labels).to(self.device) for _ in range(self.world_size)
        # ]
        # dist.all_gather(gathered_preds, all_preds)
        # dist.all_gather(gathered_labels, all_labels)

        eval_preds = {"predictions": all_preds, "label_ids": all_labels}
        metrics = self._compute_metrics(eval_preds)
        # for k, v in metrics.items():
        #     if not isinstance(v, torch.Tensor):
        #         metrics[k] = torch.tensor(v, device=self.device)
        # for k in metrics.keys():
        #     dist.all_reduce(metrics[k], op=dist.ReduceOp.AVG)

        metrics["params"] = self.activate_model.config.num_parameters
        return metrics

    @wraps(Trainer.training_step)
    def training_step(self, batch, soft_labels=None):

        local_grad = {k: v.cpu()
                      for k, v in self.activate_model.state_dict().items()}

        self.activate_model.to(self.device)
        self.activate_model.train()

        self.optimizer.zero_grad()
        self.scheduler.step()

        outputs = self.activate_model(**batch)

        loss = self.compute_loss(
            outputs, batch["labels"], soft_labels=soft_labels)

        loss.sum().backward()

        # if self.local_rank == 0:
        #     print(f"satrt all reduce {self.local_rank}")
        # for param in self.activate_model.parameters():
        #     dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        #     param.grad.data /= self.world_size

        self.optimizer.step()

        with torch.no_grad():
            for k, v in self.activate_model.state_dict().items():
                local_grad[k] = local_grad[k] - v.cpu()

        self.supernet.apply_grad(local_grad)

        dist.all_reduce(loss.data, op=dist.ReduceOp.SUM)

        train_metrics = {
            "train_loss": loss.item(),
            "params": self.activate_model.config.num_parameters,
        }
        return train_metrics

    def accumulate_grad(self):
        self.supernet.model.to(self.device)
        for name, param in self.supernet.model.named_parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.world_size
        self.supernet.model.to("cpu")

    def train(self, teacher_model=None):
        step = 0
        soft_label_dict = {}

        for epoch in range(self.args.num_train_epochs):
            if self.local_rank == 0:
                print(f"{'=+'*20} Epoch {epoch+1} {'=+'*20}")

            for i, batch in enumerate(self.train_dataloader):
                if self.local_rank == 0:
                    print(f"{'=*'*20} Step {step+1} {'=*'*20}")

                batch = {k: v.to(self.device) for k, v in batch.items()}
                soft_labels = self._get_soft_labels(batch, i, teacher_model, soft_label_dict)

                # Train supernet
                self._train_model(self.supernet.model, self.supernet.total_params, {}, batch, soft_labels, step, "supernet")

                # Train smallest subnet
                smallest_model, smallest_params, smallest_arch = self.supernet.smallest_model()
                self._train_model(smallest_model, smallest_params, smallest_arch, batch, soft_labels, step, "ssubnet")

                # Train random subnet
                random_model, random_params, random_arch = self.supernet.random_resource_aware_model()
                self._train_model(random_model, random_params, random_arch, batch, soft_labels, step, "subnet")

                if self.local_rank == 0:
                    self.supernet.save_ckpt(os.path.join(self.args.output_dir, "last_model"))
                step += 1

        self.cleanup()
        return self.train_metrics

    def _get_soft_labels(self, batch, batch_index, teacher_model, soft_label_dict):
        if teacher_model is not None:
            if batch_index not in soft_label_dict:
                teacher_model.to(self.device)
                teacher_model.eval()
                soft_labels = teacher_model(**batch).logits.detach().cpu()
                soft_label_dict[batch_index] = soft_labels
                teacher_model.to("cpu")
            else:
                soft_labels = soft_label_dict[batch_index]
        else:
            self.supernet.model.to(self.device)
            self.supernet.model.eval()
            soft_labels = self.supernet.model(**batch).logits.detach().cpu()
        return soft_labels

    def _train_model(self, model, num_params, arch, batch, soft_labels, step, prefix):
        self.activate_model = model
        self.activate_model.config.num_parameters = num_params
        self.activate_model.config.arch = arch
        self.create_optimizer_and_scheduler()

        train_metrics = self.training_step(batch, soft_labels=soft_labels)
        self.accumulate_grad()

        if self.local_rank == 0:
            self.logger.log_metrics(train_metrics, step, prefix=f"steps/{prefix}")
            self.logger.print_metrics(train_metrics, step, prefix=f"steps/{prefix}")

        if (step + 1) % self.args.log_interval == 0:
            metrics = self.evaluate(self.eval_dataloader)
            if self.local_rank == 0:
                self.logger.log_metrics(metrics, step, prefix=f"steps/{prefix}")
                self.logger.print_metrics(metrics, prefix=f"steps/{prefix}")

    def train_subnet(self, subnet,teacher_model=None):
        # for epoch in tqdm(range(self.args.num_train_epochs)):
        self.activate_model = subnet
        self.create_optimizer_and_scheduler()
        step = 0
        soft_label_dict = {}
        for epoch in range(self.args.num_train_epochs):
            if self.local_rank == 0:
                print(f"=+" * 20, f"Epoch {epoch+1}", "=+" * 20)

            for i, batch in enumerate(self.train_dataloader):
                if self.local_rank == 0:
                    print("=*" * 20, f"Step {step+1}", "=*" * 20)
                
                
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if teacher_model is not None:
                    if soft_label_dict.get(i) is None:
                        teacher_model.to(self.device)
                        teacher_model.eval()
                        soft_labels = teacher_model(
                            **batch).logits.detach().cpu()
                        soft_label_dict[i] = soft_labels
                        teacher_model.to("cpu")
                    else:
                        soft_labels = soft_label_dict[i]

                else:
                    self.supernet.model.to(self.device)
                    self.supernet.model.eval()
                    soft_labels = self.supernet.model(
                        **batch).logits.detach().cpu()



                train_metrics = self.training_step(batch, soft_labels=soft_labels)
                
                if self.local_rank == 0:
                    self.logger.log_metrics(train_metrics, step, prefix="steps/supernet")
                    self.logger.print_metrics(train_metrics, step, prefix="steps/supernet")
                
                if (step + 1) % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)
                    if self.local_rank == 0:
                        self.logger.log_metrics(metrics, step, prefix="steps/supernet")
                        self.logger.print_metrics(metrics, prefix="steps/supernet")

                self.accumulate_grad()

                if self.local_rank == 0:
                    self.supernet.save_ckpt(
                        os.path.join(self.args.output_dir, "last_model")
                    )
                step += 1

        self.cleanup()
        return train_metrics
