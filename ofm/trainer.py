import copy
import os
import time
import numpy as np
from .utils import EarlyStopping, Logger
from .modeling_ofm import OFM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.utils.data import DataLoader


class TrainingArguments:
    def __init__(
        self,
        output_dir,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        num_train_epochs,
        learning_rate,
        push_to_hub=False,  # TODO: add APIs for push to hub
        report_to=None,  # TODO: add APIs for wandb
        label_names=None,  # TODO: add label names
        fp16=False,  # TODO: add fp16
        weight_decay=0.01,
        dataloader_num_workers=8,
        log_interval=100,
        eval_steps=1000,  # TODO: add eval steps.
        early_stopping_patience=-1,  # TODO: add early stopping
    ):
        self.output_dir = output_dir
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.push_to_hub = push_to_hub
        self.report_to = report_to
        self.label_names = label_names
        self.fp16 = fp16
        self.weight_decay = weight_decay
        self.dataloader_num_workers = dataloader_num_workers
        self.log_interval = log_interval
        self.eval_steps = eval_steps
        self.early_stopping_patience = early_stopping_patience


class Trainer:
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
        self.supernet = supernet
        self.activate_model = None
        self.args = args
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.optimizer, self.scheduler = optimizers
        self.logger = Logger(log_dir=os.path.join(args.output_dir, "logs"))
        self.train_dataloader = self.get_train_dataloader()
        if self.eval_dataset:
            self.eval_dataloader = self.get_eval_dataloader()
        if self.test_dataset:
            self.test_dataloader = self.get_test_dataloader()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # training manager
        self.best_metric = {}

    def log_metrics(self, metrics, step, log_interval, prefix):
        self.logger.log_metrics(metrics, step, prefix=prefix)
        self.logger.print_metrics(metrics, prefix=prefix)
        if step + 1 % log_interval == 0:
            metrics = self.evaluate(self.eval_dataloader)
            self.logger.log_metrics(metrics, step, prefix=prefix)
            self.logger.print_metrics(metrics, prefix=prefix)

    def update_best_metric(self, metrics):
        if self.best_metric == {}:
            self.best_metric = metrics
            # self.supernet.save_ckpt(os.path.join(self.args.output_dir, "best_model"))
        else:
            for key in metrics:
                if key == "params":
                    continue
                if metrics[key] > self.best_metric[key]:
                    self.best_metric[key] = metrics[key]
                    self.supernet.save_ckpt(
                        os.path.join(self.args.output_dir, key + "_best_model")
                    )

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
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
        )

    def get_test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def create_optimizer_and_scheduler(self):
        # TODO: if my optimizer and schedular passing by argument, skip this step
        self.optimizer = AdamW(
            self.activate_model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        if self.scheduler is None:
            self.scheduler = LambdaLR(
                self.optimizer, lr_lambda=lambda x: max(0.1, 0.975**x)
            )
        else:
            self.scheduler.optimizer = self.optimizer
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda x: 0.975**x)

    def compute_loss(self, outputs, labels, soft_labels=None):
        """returns the loss"""

        if soft_labels is not None:
            kd_loss = F.kl_div(
                F.log_softmax(outputs.logits, dim=1),
                F.softmax(soft_labels.to(self.device), dim=1),
                reduction="batchmean",
            )
            return outputs.loss + kd_loss
        return outputs.loss

    def _compute_metrics(self, eval_preds):
        if self.compute_metrics is None:
            return {}
        return self.compute_metrics(eval_preds)

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
        eval_preds = {"predictions": all_preds, "label_ids": all_labels}
        metrics = self._compute_metrics(eval_preds)
        metrics["params"] = self.activate_model.config.num_parameters
        return metrics

    def training_step(self, batch, soft_labels=None):
        local_grad = {k: v.cpu() for k, v in self.activate_model.state_dict().items()}

        self.activate_model.to(self.device)
        self.activate_model = nn.DataParallel(self.activate_model)

        self.activate_model.train()
        self.optimizer.zero_grad()
        outputs = self.activate_model(**batch)

        loss = self.compute_loss(
            outputs,
            labels=batch["labels"] if hasattr(batch, "labels") else None,
            soft_labels=soft_labels,
        )
        # loss.backward()
        loss.sum().backward()
        self.optimizer.step()
        self.scheduler.step()

        self.activate_model = self.activate_model.module

        with torch.no_grad():
            for k, v in self.activate_model.state_dict().items():
                local_grad[k] = local_grad[k] - v.cpu()

        self.supernet.apply_grad(local_grad)

        train_metrics = {
            "train_loss": loss.sum().item(),
            "params": self.activate_model.config.num_parameters,
        }
        return train_metrics

    def train(self):

        for epoch in range(self.args.num_train_epochs):
            print("==" * 20, f"Epoch {epoch}", "==" * 20)
            # TODO: add tqdm
            for step, batch in enumerate(self.train_dataloader):
                print("=*" * 20, f"Step {step}", "=*" * 20)

                # for step, batch in enumerate(self.train_dataloader):
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

                self.create_optimizer_and_scheduler()

                train_metrics = self.training_step(batch)

                self.logger.log_metrics(train_metrics, step, prefix="steps/supernet")
                self.logger.print_metrics(train_metrics, step, prefix="steps/supernet")
                if (step + 1) % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)
                    self.update_best_metric(metrics)
                    self.logger.log_metrics(metrics, step, prefix="steps/supernet")
                    self.logger.print_metrics(metrics, prefix="steps/supernet")

                self.activate_model.eval()
                self.activate_model.to(self.device)
                soft_labels = self.activate_model(**batch).logits.detach().cpu()

                # Train smallest subnet
                (
                    self.activate_model,
                    self.activate_model.config.num_parameters,
                    self.activate_model.config.arch,
                ) = self.supernet.smallest_model()

                # self.activate_model.to(self.device)
                self.create_optimizer_and_scheduler()

                train_metrics = self.training_step(batch, soft_labels=soft_labels)

                self.logger.log_metrics(train_metrics, step, prefix="steps/ssubnet")
                self.logger.print_metrics(train_metrics, prefix="steps/ssubnet")
                if (step + 1) % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)
                    self.logger.log_metrics(metrics, step, prefix="steps/ssubnet")
                    self.logger.print_metrics(metrics, prefix="steps/ssubnet")

                # Train random subnets
                (
                    self.activate_model,
                    self.activate_model.config.num_parameters,
                    self.activate_model.config.arch,
                ) = self.supernet.random_resource_aware_model()

                # self.activate_model.to(self.device)
                self.create_optimizer_and_scheduler()

                train_metrics = self.training_step(batch, soft_labels=soft_labels)

                self.logger.log_metrics(train_metrics, step, prefix="steps/subnet")
                self.logger.print_metrics(train_metrics, prefix="steps/subnet")
                if (step + 1) % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)
                    self.logger.log_metrics(metrics, step, prefix="steps/subnet")
                    self.logger.print_metrics(metrics, prefix="steps/subnet")

                self.supernet.save_ckpt(
                    os.path.join(self.args.output_dir, "last_model")
                )

        return train_metrics

    def train_subnet(self, subnet):

        self.activate_model = subnet

        if not self.optimizer or not self.scheduler:
            self.create_optimizer_and_scheduler()

        avg_train_metrics = {}
        step = 0
        for epoch in range(self.args.num_train_epochs):
            print("==" * 20, f"Epoch {epoch}", "==" * 20)
            # TODO: add tqdm
            for i, batch in enumerate(self.train_dataloader):
                print("=*" * 20, f"Step {step+1}", "=*" * 20)

                batch = {k: v.to(self.device) for k, v in batch.items()}

                # get soft labels
                self.supernet.model.eval()
                self.supernet.model.to(self.device)
                soft_labels = self.supernet.model(**batch).logits.detach().cpu()
                self.supernet.model.to("cpu")

                train_metrics = self.training_step(batch, soft_labels=soft_labels)
                for k, v in train_metrics.items():
                    avg_train_metrics[k] = avg_train_metrics.get(k, 0) + v

                self.logger.log_metrics(train_metrics, step, prefix="steps/subnet")
                self.logger.print_metrics(train_metrics, prefix="steps/subnet")
                if (step + 1) % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)
                    self.update_best_metric(metrics)
                    self.logger.log_metrics(metrics, step, prefix="steps/subnet")
                    self.logger.print_metrics(metrics, prefix="steps/subnet")

                step += 1

                self.supernet.save_ckpt(
                    os.path.join(self.args.output_dir, "last_model")
                )

        for k in avg_train_metrics:
            avg_train_metrics[k] /= step

        return avg_train_metrics


class CLIPTrainer(Trainer):
    
    def compute_loss(self, outputs, labels, soft_labels=None):
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        # Normalize the embeddings
        image_embeds = F.normalize(image_embeds, dim=1)
        text_embeds = F.normalize(text_embeds, dim=1)

        # Compute the cosine similarity between image and text embeddings
        logits_per_image = torch.matmul(image_embeds, text_embeds.t())
        logits_per_text = logits_per_image.t()

        # Create labels for contrastive loss
        labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

        # Compute the contrastive loss
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_image + loss_text) / 2

        return loss

    def train(self):

        for epoch in range(self.args.num_train_epochs):
            print("==" * 20, f"Epoch {epoch}", "==" * 20)
            # TODO: add tqdm
            for step, batch in enumerate(self.train_dataloader):
                print("=*" * 20, f"Step {step}", "=*" * 20)

                # for step, batch in enumerate(self.train_dataloader):
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

                self.create_optimizer_and_scheduler()

                train_metrics = self.training_step(batch)

                self.logger.log_metrics(train_metrics, step, prefix="steps/supernet")
                self.logger.print_metrics(train_metrics, step, prefix="steps/supernet")
                if (step + 1) % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)
                    self.update_best_metric(metrics)
                    self.logger.log_metrics(metrics, step, prefix="steps/supernet")
                    self.logger.print_metrics(metrics, prefix="steps/supernet")

                self.activate_model.eval()
                self.activate_model.to(self.device)
                # soft_labels = self.activate_model(**batch).logits.detach().cpu()
                soft_labels = None
                # Train smallest subnet
                (
                    self.activate_model,
                    self.activate_model.config.num_parameters,
                    self.activate_model.config.arch,
                ) = self.supernet.smallest_model()

                # self.activate_model.to(self.device)
                self.create_optimizer_and_scheduler()

                train_metrics = self.training_step(batch, soft_labels=soft_labels)

                self.logger.log_metrics(train_metrics, step, prefix="steps/ssubnet")
                self.logger.print_metrics(train_metrics, prefix="steps/ssubnet")
                if (step + 1) % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)
                    self.logger.log_metrics(metrics, step, prefix="steps/ssubnet")
                    self.logger.print_metrics(metrics, prefix="steps/ssubnet")

                # Train random subnets
                (
                    self.activate_model,
                    self.activate_model.config.num_parameters,
                    self.activate_model.config.arch,
                ) = self.supernet.random_resource_aware_model()

                # self.activate_model.to(self.device)
                self.create_optimizer_and_scheduler()

                train_metrics = self.training_step(batch, soft_labels=soft_labels)

                self.logger.log_metrics(train_metrics, step, prefix="steps/subnet")
                self.logger.print_metrics(train_metrics, prefix="steps/subnet")
                if (step + 1) % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)
                    self.logger.log_metrics(metrics, step, prefix="steps/subnet")
                    self.logger.print_metrics(metrics, prefix="steps/subnet")

                self.supernet.save_ckpt(
                    os.path.join(self.args.output_dir, "last_model")
                )

        return train_metrics

    def train_subnet(self, subnet):

        self.activate_model = subnet

        if not self.optimizer or not self.scheduler:
            self.create_optimizer_and_scheduler()

        avg_train_metrics = {}
        step = 0
        for epoch in range(self.args.num_train_epochs):
            print("==" * 20, f"Epoch {epoch}", "==" * 20)
            # TODO: add tqdm
            for i, batch in enumerate(self.train_dataloader):
                print("=*" * 20, f"Step {step+1}", "=*" * 20)

                batch = {k: v.to(self.device) for k, v in batch.items()}
                input_batch = {
                    "pixel_values": batch["pixel_values"],
                    "input_ids": batch["input_ids"],
                }
                train_metrics = self.training_step(input_batch)
                for k, v in train_metrics.items():
                    avg_train_metrics[k] = avg_train_metrics.get(k, 0) + v

                self.logger.log_metrics(train_metrics, step, prefix="steps/subnet")
                self.logger.print_metrics(train_metrics, prefix="steps/subnet")
                if (step + 1) % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)
                    self.update_best_metric(metrics)
                    self.logger.log_metrics(metrics, step, prefix="steps/subnet")
                    self.logger.print_metrics(metrics, prefix="steps/subnet")

                step += 1

                self.supernet.save_ckpt(
                    os.path.join(self.args.output_dir, "last_model")
                )

        for k in avg_train_metrics:
            avg_train_metrics[k] /= step

        return avg_train_metrics

    def evaluate(self, eval_dataloader):
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        self.activate_model.eval()

        true_labels = []
        pred_labels = []

        progress_bar = tqdm(self.eval_dataloader, desc="Evaluation")

        for batch in progress_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            images = batch["pixel_values"]
            input_ids = batch["input_ids"]

            labels = batch["labels"]

            with torch.no_grad():
                outputs = self.activate_model(pixel_values=images, input_ids=input_ids)
                logits = outputs.logits_per_image
                predicted_labels = torch.argmax(logits, dim=1).to("cpu").tolist()

            true_labels.extend(labels.to("cpu"))
            pred_labels.extend(predicted_labels)

            # Calculate intermediate metrics
            accuracy = accuracy_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels, average="weighted")
            precision = precision_score(true_labels, pred_labels, average="weighted")
            recall = recall_score(true_labels, pred_labels, average="weighted")

            progress_bar.set_postfix(
                {
                    "Accuracy": f"{accuracy:.4f}",
                    "F1 Score": f"{f1:.4f}",
                    "Precision": f"{precision:.4f}",
                    "Recall": f"{recall:.4f}",
                }
            )
        eval_metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
        return eval_metrics
