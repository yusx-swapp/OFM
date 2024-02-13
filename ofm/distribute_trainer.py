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

from transformers import TrainingArguments, Trainer
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_optimizer_and_scheduler(model, lr):
    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Define a custom scheduler
    def lr_lambda(current_step: int):
        # Custom learning rate decay
        return max(0.1, 0.975**current_step)

    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


class TrainingArguments:
    def __init__(
        self,
        output_dir,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        evaluation_strategy,
        save_strategy,
        save_total_limit,
        num_train_epochs,
        learning_rate,
        remove_unused_columns,
        push_to_hub,
        report_to,
        label_names,
        fp16,
        weight_decay,
        dataloader_num_workers,
        local_rank,
    ):
        self.output_dir = output_dir
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.save_total_limit = save_total_limit
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.remove_unused_columns = remove_unused_columns
        self.push_to_hub = push_to_hub
        self.report_to = report_to
        self.label_names = label_names
        self.fp16 = fp16
        self.weight_decay = weight_decay
        self.dataloader_num_workers = dataloader_num_workers
        self.local_rank = local_rank


class Trainer:
    def __init__(
        self,
        model,
        args,
        data_collator,
        compute_metrics,
        train_dataset,
        eval_dataset,
        tokenizer,
        optimizers,
    ):
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.optimizer, self.scheduler = optimizers
        self.logger = Logger(log_dir=os.path.join(args.output_dir, "logs"))
        self.train_dataloader = self.get_train_dataloader()
        if self.eval_dataset:
            self.eval_dataloader = self.get_eval_dataloader()
        if self.test_dataset:
            self.test_dataloader = self.get_test_dataloader()

    def get_train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self):

        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def create_optimizer_and_scheduler(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda x: 0.975**x)

    def compute_loss(self, outputs, labels):
        return F.cross_entropy(outputs.logits, labels)

    def training_step(self, batch):
        self.model.train()
        outputs = self.model(**batch)
        loss = self.compute_loss(outputs, batch["labels"])
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def _compute_metrics(self, eval_preds):
        if self.compute_metrics is None:
            return {}
        return self.compute_metrics(eval_preds)

    def evaluate(self, eval_dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = self.model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                all_preds.append(preds)
                all_labels.append(batch["labels"])
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self._compute_metrics(all_preds, all_labels)
        return metrics

    def distribute_train(self, rank, world_size):
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        self.model = self.model.to(rank)
        # self.model = nn.parallel.DistributedDataParallel(
        #     self.model, device_ids=[rank], output_device=rank
        # )
        self.train_dataloader = self.get_train_dataloader()
        self.eval_dataloader = self.get_eval_dataloader()
        self.test_dataloader = self.get_test_dataloader()
        self.train()
        cleanup()

    def train(self):
        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                loss = self.training_step(batch)
                self.logger.log_metrics(
                    {"train_loss": loss.item()},
                    step,
                    prefix="steps",
                )
                if step % 100 == 0:
                    metrics = self.evaluate(eval_dataloader)
                    self.logger.log_metrics(
                        metrics,
                        step,
                        prefix="steps",
                    )
                    self.save_metrics("eval" + f"-step {step}", metrics)
        return metrics


def ofm_dist_train(
    args,
    model: OFM,
    train_dataset,
    val_dataset,
    test_dataset=None,
    processor=None,
    collate_fn=None,
    compute_metrics=None,
    training_args=None,
    rank=-1,
    world_size=-1,
):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # TODO: Wrap summary writer in a context manager
    # writer = SummaryWriter(os.path.join(args.save_dir, "logs"))
    writer = Logger(log_dir=os.path.join(args.save_dir, "logs"))
    best_acc = 0.0
    best_f1 = 0.0
    # TODO: add training args as an s
    training_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "training"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="no",
        save_total_limit=1,
        num_train_epochs=args.num_local_epochs,
        learning_rate=args.lr,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        label_names=["labels"],
        fp16=args.fp16,
        weight_decay=1e-4,
        dataloader_num_workers=8,
        # load_best_model_at_end=True,
        local_rank=rank,
    )

    steps = 0

    train_dataset = train_dataset.shuffle()
    # train_dataset.select(range(1000))

    if rank == 0:
        ds_model = copy.deepcopy(model.model)
        ds_model_params = model.total_params
    elif rank == 1:
        (
            ds_model,
            ds_model_params,
            arc_config,
        ) = model.smallest_model()
    else:
        (
            ds_model,
            ds_model_params,
            arc_config,
        ) = model.random_resource_aware_model()

    avg_params += ds_model_params

    local_grad = {k: v.cpu() for k, v in ds_model.state_dict().items()}

    print("Training on {} parameters".format(ds_model_params))
    # random sample 5k for evaluation
    val_indices = np.random.choice(
        list(range(len(val_dataset))), size=args.epoch_eval_size, replace=False
    )
    trainer = Trainer(
        model=ds_model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset.select(mini_shard_idx),
        eval_dataset=val_dataset.select(val_indices),
        tokenizer=processor,
        optimizers=get_optimizer_and_scheduler(ds_model, lr),
    )
    train_results = trainer.train()

    epoch_train_loss += train_results.metrics["train_loss"]

    ds_weights = {k: v.cpu() for k, v in ds_model.state_dict().items()}
    import torch

    with torch.no_grad():
        for key in ds_weights:
            local_grad[key] = local_grad[key] - ds_weights[key]

    # model.grad_accumulate(local_grad, alpha=len(mini_shard_idx))
    model.apply_grad(local_grad)

    if (steps % args.log_interval == 0) or (steps % args.num_shards == 0):
        # if False:
        # Evaluate the model

        print("*" * 20 + "Evaluating in train step {}".format(steps) + "*" * 20)

        ds_model.cuda()
        metrics = trainer.evaluate(val_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval" + f"-step {steps}", metrics)

        val_accuracy, val_f1_score = (
            metrics["eval_metric"],
            metrics["eval_f1"],
        )
        ds_model.to("cpu")
        with writer:
            writer.log_metrics(
                metrics,
                steps,
                prefix="steps",
            )

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            model.save_ckpt(os.path.join(args.save_dir, "best_model"))
        if val_f1_score > best_f1:
            best_f1 = val_f1_score
        writer.add_scalar(
            "global/best_metric",
            best_acc,
            steps,
        )
        writer.add_scalar(
            "global/best_f1",
            best_f1,
            steps,
        )

        print(
            f"Best validation accuracy: {best_acc} \nBest validation f1: {best_f1}  \nDownsized model size: {ds_model_params}"
        )
        print("*" * 20 + "Evaluation finished" + "*" * 20)

        if test_dataset:
            metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

        if steps % args.num_shards == 0:
            early_stopping(val_f1_score)

        # Apply the aggregated and normalized gradient to the full-size model
        # model.apply_accumulate_grad(args.grad_beta)

    model.save_ckpt(os.path.join(args.save_dir, "last_model"))
    # if args.push_to_hub:
    #     model.model.push_to_hub(repo_name=args.save_dir.split("/")[-1])
    return model
