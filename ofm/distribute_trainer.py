from functools import wraps
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import copy
import os
import time
import numpy as np
from .utils import EarlyStopping, step_lr, Logger, print_rank_0, get_all_reduce_mean
from .modeling_ofm import OFM
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

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
        local_rank=-1,
    ):

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

        self.local_rank = local_rank
        if self.local_rank == -1:
            self.device = torch.device(
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cuda:{}".format(self.local_rank))
            self.setup()

    def setup(self):
        dist.init_process_group(backend="nccl")
        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()
        print(
            f"Global rank: {self.global_rank} initialized (local rank: {self.local_rank}), world size: {self.world_size}"
        )

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    @wraps(Trainer.get_train_dataloader)
    def get_train_dataloader(self):
        print("[Remove in release]Creating train dataloader")
        if self.local_rank == -1:
            train_sampler = RandomSampler(self.train_dataset)
        else:
            train_sampler = DistributedSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            sampler=train_sampler,
        )

    @wraps(Trainer.get_eval_dataloader)
    def get_eval_dataloader(self):
        print("[Remove in release] Creating eval dataloader")
        if self.local_rank == -1:
            eval_sampler = SequentialSampler(self.eval_dataset)
        else:
            eval_sampler = DistributedSampler(self.eval_dataset)
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            sampler=eval_sampler,
        )

    @wraps(Trainer.get_test_dataloader)
    def get_test_dataloader(self):
        print("[Remove in release] Creating test dataloader")
        if self.local_rank == -1:
            test_sampler = SequentialSampler(self.test_dataset)
        else:
            test_sampler = DistributedSampler(self.test_dataset)
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            sampler=test_sampler,
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

        eval_preds = {"predictions": all_preds, "label_ids": all_labels}
        metrics = self._compute_metrics(eval_preds)

        for k, v in metrics.items():
            try:
                metrics[k] = get_all_reduce_mean(v)
            except:
                pass

        metrics["params"] = self.activate_model.config.num_parameters
        return metrics

    @wraps(Trainer.training_step)
    def training_step(self, batch, soft_labels=None):

        local_grad = {k: v.cpu() for k, v in self.activate_model.state_dict().items()}

        self.activate_model.to(self.device)
        self.activate_model.train()

        self.optimizer.zero_grad()
        self.scheduler.step()

        outputs = self.activate_model(**batch)

        loss = self.compute_loss(outputs, batch["labels"], soft_labels=soft_labels)

        loss.sum().backward()

        self.optimizer.step()

        with torch.no_grad():
            for k, v in self.activate_model.state_dict().items():
                local_grad[k] = local_grad[k] - v.cpu()

        self.supernet.apply_grad(local_grad)

        try:
            loss.data = get_all_reduce_mean(loss.data)
        except:
            pass
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

    @wraps(Trainer.train)
    def train(self):
        # for epoch in tqdm(range(self.args.num_train_epochs)):
        step = 0
        for epoch in range(self.args.num_train_epochs):
            print_rank_0(print(f"=+" * 20, f"Epoch {epoch+1}", "=+" * 20))
            for i, batch in enumerate(self.train_dataloader):

                print_rank_0("=*" * 20, f"Step {step+1}", "=*" * 20)

                batch = {k: v.to(self.device) for k, v in batch.items()}

                (
                    self.activate_model,
                    self.activate_model.config.num_parameters,
                    self.activate_model.config.arch,
                ) = (
                    copy.deepcopy(self.supernet.model),
                    self.supernet.total_params,
                    {},
                )

                self.create_optimizer_and_scheduler()

                train_metrics = self.training_step(batch)
                self.logger.log_metrics_rank_0(
                    train_metrics, step, prefix="steps/supernet", rank=self.global_rank
                )
                self.logger.print_metrics_rank_0(
                    train_metrics, step, prefix="steps/supernet", rank=self.global_rank
                )
                if (step + 1) % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)
                    self.logger.log_metrics_rank_0(
                        metrics, step, prefix="steps/supernet", rank=self.global_rank
                    )
                    self.logger.print_metrics_rank_0(
                        metrics, step, prefix="steps/supernet", rank=self.global_rank
                    )

                self.accumulate_grad()

                self.supernet.model.to(self.device)
                self.supernet.model.eval()
                soft_labels = self.supernet.model(**batch).logits.detach().cpu()

                # Train smallest subnet
                (
                    self.activate_model,
                    self.activate_model.config.num_parameters,
                    self.activate_model.config.arch,
                ) = self.supernet.smallest_model()

                self.create_optimizer_and_scheduler()

                train_metrics = self.training_step(batch, soft_labels=soft_labels)

                self.accumulate_grad()
                self.logger.log_metrics_rank_0(
                    train_metrics, step, prefix="steps/ssubnet", rank=self.global_rank
                )
                self.logger.print_metrics_rank_0(
                    train_metrics, step, prefix="steps/ssubnet", rank=self.global_rank
                )
                if (step + 1) % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)
                    self.logger.log_metrics_rank_0(
                        metrics, step, prefix="steps/ssubnet", rank=self.global_rank
                    )
                    self.logger.print_metrics_rank_0(
                        metrics, step, prefix="steps/ssubnet", rank=self.global_rank
                    )
                # Train random subnets
                (
                    self.activate_model,
                    self.activate_model.config.num_parameters,
                    self.activate_model.config.arch,
                ) = self.supernet.random_resource_aware_model()

                self.create_optimizer_and_scheduler()

                train_metrics = self.training_step(batch, soft_labels=soft_labels)

                self.accumulate_grad()

                self.logger.log_metrics_rank_0(
                    train_metrics, step, prefix="steps/subnet", rank=self.global_rank
                )
                self.logger.print_metrics_rank_0(
                    train_metrics, step, prefix="steps/subnet", rank=self.global_rank
                )

                if (step + 1) % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)
                    self.logger.log_metrics_rank_0(
                        metrics, step, prefix="steps/subnet", rank=self.global_rank
                    )
                    self.logger.print_metrics_rank_0(
                        metrics, step, prefix="steps/subnet", rank=self.global_rank
                    )

                if self.global_rank == 0:
                    self.supernet.save_ckpt(
                        os.path.join(self.args.output_dir, "last_model")
                    )
                step += 1

        self.cleanup()
        return train_metrics
