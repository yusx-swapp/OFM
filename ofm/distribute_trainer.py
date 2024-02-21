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

        # self.local_rank = dist.get_rank()
        self.local_rank = int(os.environ["RANK"])

        self.world_size = dist.get_world_size()

        # dist.init_process_group(
        #     "nccl", rank=self.local_rank, world_size=self.world_size
        # )
        print(f"Rank {self.local_rank} initialized, world size: {self.world_size}")

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
        # train_dataloader = DataLoader(
        #     train_ds,
        #     batch_size=config.batch_size,
        #     shuffle=False,
        #     sampler=DistributedSampler(train_ds, shuffle=True),
        # )

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
        for batch in eval_dataloader:
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.activate_model(**batch)
                preds = outputs.logits.detach().cpu()
                all_preds.append(preds)
                all_labels.append(batch["labels"].detach().cpu())
        batch.clear()
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        eval_preds = {"predictions": all_preds, "label_ids": all_labels}
        metrics = self._compute_metrics(eval_preds)
        metrics["params"] = self.activate_model.config.num_parameters
        return metrics

    @wraps(Trainer.training_step)
    def training_step(self, batch):
        self.activate_model.train()

        self.optimizer.zero_grad()
        self.scheduler.step()
        outputs = self.activate_model(**batch)
        loss = self.compute_loss(outputs, batch["labels"])
        loss.backward()

        if self.local_rank == 0:
            print(f"satrt all reduce {self.local_rank}")

        for param in self.activate_model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= self.world_size

        self.optimizer.step()
        dist.all_reduce(loss.data, op=dist.ReduceOp.SUM)
        train_metrics = {
            "train_loss": loss.item(),
            "params": self.activate_model.config.num_parameters,
        }
        return train_metrics

    @wraps(Trainer.train)
    def train(self):

        for epoch in range(self.args.num_train_epochs):

            for step, batch in enumerate(self.train_dataloader):
                # print(batch["labels"].shape)
                # print(batch["pixel_values"].shape)
                # move batch to device
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

                self.activate_model.to(self.device)
                self.create_optimizer_and_scheduler()

                local_grad = {
                    k: v.cpu() for k, v in self.activate_model.state_dict().items()
                }
                train_metrics = self.training_step(batch)

                ds_weights = {
                    k: v.cpu() for k, v in self.activate_model.state_dict().items()
                }
                with torch.no_grad():
                    for key in ds_weights:
                        local_grad[key] = local_grad[key] - ds_weights[key]

                if self.local_rank == 0:
                    self.logger.log_metrics(
                        train_metrics, step, prefix="steps/supernet"
                    )
                    self.logger.print_metrics(
                        train_metrics, step, prefix="steps/supernet"
                    )
                if step % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)

                    if self.local_rank == 0:
                        self.logger.log_metrics(metrics, step, prefix="steps/supernet")
                        self.logger.print_metrics(metrics, prefix="steps/supernet")

                self.supernet.apply_grad(local_grad)

                # soft_labels = self.activate_model(**batch).logits

                self.supernet.save_ckpt(
                    os.path.join(self.args.output_dir, "last_model")
                )

        self.cleanup()
        return metrics
