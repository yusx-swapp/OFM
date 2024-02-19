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
        local_rank=-1,
        world_size=-1,
    ):
        self.local_rank = local_rank
        self.world_size = world_size
        self.setup(local_rank, world_size)
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

    def setup(self, rank, world_size):
        # os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = "12355"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        # dist.init_process_group(rank=rank, world_size=world_size)
        print(f"Rank {rank} initialized, world size: {world_size}")

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
                # print(outputs.predictions)
                # eval_preds = self.activate_model(**batch)
                preds = outputs.logits.detach().cpu()
                all_preds.append(preds)
                all_labels.append(batch["labels"].detach().cpu())
        batch.clear()
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        # dist.all_reduce(all_preds, op=dist.ReduceOp.SUM)
        # dist.all_reduce(all_labels, op=dist.ReduceOp.SUM)
        eval_preds = {"predictions": all_preds, "label_ids": all_labels}
        metrics = self._compute_metrics(eval_preds)
        metrics["params"] = self.activate_model.config.num_parameters
        return metrics

    @wraps(Trainer.training_step)
    def training_step(self, batch):
        self.activate_model.train()

        self.optimizer.zero_grad()
        self.scheduler.step()
        # print optimizer LR:
        print("the current learning rate")
        current_lrs = [group["lr"] for group in self.optimizer.param_groups]
        print(current_lrs)
        print(self.optimizer.param_groups[0]["lr"])
        outputs = self.activate_model(**batch)
        loss = self.compute_loss(outputs, batch["labels"])
        loss.backward()
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
                    100,
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

                self.logger.log_metrics(train_metrics, step, prefix="steps")
                self.logger.print_metrics(train_metrics, step, prefix="steps")

                if step % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)
                    self.logger.log_metrics(metrics, step, prefix="steps")
                    self.logger.print_metrics(metrics, step, prefix="steps")

                    # self.lsave_metrics("eval" + f"-step {step}", metrics)
                self.supernet.apply_grad(local_grad)
                self.supernet.save_ckpt(
                    os.path.join(self.args.output_dir, "last_model")
                )

        self.cleanup()
        return metrics
