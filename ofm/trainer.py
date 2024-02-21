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
            num_workers=self.args.dataloader_num_workers,
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
                F.softmax(soft_labels, dim=1),
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
        self.activate_model.train()
        self.optimizer.zero_grad()
        outputs = self.activate_model(**batch)
        loss = self.compute_loss(outputs, batch["labels"], soft_labels=soft_labels)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        train_metrics = {
            "train_loss": loss.item(),
            "params": self.activate_model.config.num_parameters,
        }
        return train_metrics

    def train(self):

        for epoch in range(self.args.num_train_epochs):

            # TODO: add tqdm
            for step, batch in enumerate(self.train_dataloader):
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

                self.logger.log_metrics(train_metrics, step, prefix="steps/supernet")
                self.logger.print_metrics(train_metrics, step, prefix="steps/supernet")
                if step % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)
                    self.logger.log_metrics(metrics, step, prefix="steps/supernet")
                    self.logger.print_metrics(metrics, prefix="steps/supernet")

                    # self.lsave_metrics("eval" + f"-step {step}", metrics)

                self.supernet.apply_grad(local_grad)

                soft_labels = self.activate_model(**batch).logits

                # Train random subnets
                (
                    self.activate_model,
                    self.activate_model.config.num_parameters,
                    self.activate_model.config.arch,
                ) = self.supernet.random_resource_aware_model()

                self.activate_model.to(self.device)
                self.create_optimizer_and_scheduler()

                local_grad = {
                    k: v.cpu() for k, v in self.activate_model.state_dict().items()
                }

                train_metrics = self.training_step(batch, soft_labels=soft_labels)

                ds_weights = {
                    k: v.cpu() for k, v in self.activate_model.state_dict().items()
                }
                with torch.no_grad():
                    for key in ds_weights:
                        local_grad[key] = local_grad[key] - ds_weights[key]

                self.logger.log_metrics(train_metrics, step, prefix="steps/subnet")
                self.logger.print_metrics(train_metrics, prefix="steps/subnet")
                if step % self.args.log_interval == 0:
                    metrics = self.evaluate(self.eval_dataloader)
                    self.logger.log_metrics(metrics, step, prefix="steps/subnet")
                    self.logger.print_metrics(metrics, prefix="steps/subnet")

                    # self.lsave_metrics("eval" + f"-step {step}", metrics)

                self.supernet.apply_grad(local_grad)

                self.supernet.save_ckpt(
                    os.path.join(self.args.output_dir, "last_model")
                )

        return metrics


"""
def ofm_train(
    args,
    model: OFM,
    train_dataset,
    val_dataset,
    test_dataset=None,
    processor=None,
    collate_fn=None,
    compute_metrics=None,
    training_args=None,
):
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # TODO: Wrap summary writer in a context manager
    writer = SummaryWriter(os.path.join(args.save_dir, "logs"))
    best_acc = 0.0
    best_f1 = 0.0
    # TODO: add training args as an argument
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
    )

    steps = 0

    train_dataset = train_dataset.shuffle()
    # train_dataset.select(range(1000))
    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        print("=" * 20 + "Training for epoch {}".format(epoch) + "=" * 20)

        # sharding the dataset to args.num_shards shards
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        size = len(indices) // args.num_shards
        mini_shards = [
            indices[i * size : (i + 1) * size] for i in range(args.num_shards)
        ]
        mini_shards[-1].extend(indices[args.num_shards * size :])

        epoch_train_loss = 0.0

        lr = step_lr(args.lr, epoch, args.step_size, 0.98)
        training_args.learning_rate = lr
        np.random.seed(int(time.time()))  # Set the seed to the current time

        if args.spp:
            model.salient_parameter_prioritization()
        avg_params = 0

        # Train each downsized model independently in a sequential manner
        for idx, mini_shard_idx in enumerate(
            tqdm(mini_shards, desc="Mini-shard training")
        ):
            steps += 1

            if idx == 0:
                ds_model = copy.deepcopy(model.model)
                ds_model_params = model.total_params
            elif idx == 1:
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

                writer.add_scalar(
                    "steps/eval_metric",
                    val_accuracy,
                    steps,
                )
                writer.add_scalar(
                    "steps/eval_f1",
                    val_f1_score,
                    steps,
                )

                writer.add_scalar(
                    "steps/params",
                    ds_model_params,
                    steps,
                )

                # model.model.to("cpu")
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

        print("=" * 20 + "Training finished for epoch {}".format(epoch) + "=" * 20)

        avg_params = avg_params / len(mini_shards)
        writer.add_scalar(
            "global/params",
            avg_params,
            epoch,
        )

        writer.add_scalar(
            "global/train_loss",
            epoch_train_loss,
            epoch,
        )

        if early_stopping.has_converged():
            print("Model has converged. Stopping training.")
            break
    return model


def ofm_train_squad(
    args,
    model: OFM,
    train_dataset,
    val_dataset,
    test_dataset=None,
    processor=None,
    collate_fn=None,
    compute_metrics=None,
):
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    writer = SummaryWriter(os.path.join(args.save_dir, "logs"))
    best_acc = 0.0
    best_f1 = 0.0
    training_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "training"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="no",
        save_strategy="no",
        save_total_limit=1,
        num_train_epochs=args.num_local_epochs,
        learning_rate=args.lr,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to="none",
        label_names=["labels"],
        fp16=args.fp16,
        weight_decay=1e-4,
        dataloader_num_workers=8,
    )
    steps = 0

    train_dataset = train_dataset.shuffle()
    # train_dataset.select(range(1000))
    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        print("=" * 20 + "Training for epoch {}".format(epoch) + "=" * 20)

        epoch_train_loss = 0.0

        lr = step_lr(args.lr, epoch, args.step_size, 0.98)
        # lr = trainer.learning_rate
        # lr = trainer.state.lear
        training_args.learning_rate = lr

        np.random.seed(int(time.time()))  # Set the seed to the current time

        if args.spp:
            model.salient_parameter_prioritization()
        avg_params = 0

        # Train each downsized model independently in a sequential manner
        for idx in tqdm(range(args.num_shards), desc="Mini-shard training"):
            # for idx, mini_shard_idx in enumerate(
            #     tqdm(mini_shards, desc="Mini-shard training")
            # ):
            steps += 1

            if idx == 0:
                ds_model = copy.deepcopy(model.model)
                ds_model_params = model.total_params
            elif idx == 1:
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

            trainer = Trainer(
                model=ds_model,
                args=training_args,
                data_collator=collate_fn,
                compute_metrics=compute_metrics,
                train_dataset=train_dataset.shard(
                    num_shards=args.num_shards, index=idx
                ),
                # eval_dataset=val_dataset.select(val_indices),
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

            # model.grad_accumulate(
            #     local_grad,
            #     alpha=len(train_dataset.shard(num_shards=args.num_shards, index=idx)),
            # )
            model.apply_grad(local_grad)

            if (steps % args.log_interval == 0) or (steps % args.num_shards == 0):
                # Evaluate the model

                print("*" * 20 + "Evaluating in train step {}".format(steps) + "*" * 20)
                predictions = trainer.predict(val_dataset)
                ds_model.to("cpu")
                metrics = compute_metrics(predictions)

                # metrics = trainer.evaluate(val_dataset)
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

                val_accuracy, val_f1_score = (
                    metrics["exact_match"],
                    metrics["f1"],
                )

                writer.add_scalar(
                    "steps/eval_metric",
                    val_accuracy,
                    steps,
                )
                writer.add_scalar(
                    "steps/eval_f1",
                    val_f1_score,
                    steps,
                )

                writer.add_scalar(
                    "steps/params",
                    ds_model_params,
                    steps,
                )

                # model.model.to("cpu")
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

        print("=" * 20 + "Training finished for epoch {}".format(epoch) + "=" * 20)

        avg_params = avg_params / args.num_shards
        writer.add_scalar(
            "global/params",
            avg_params,
            epoch,
        )

        writer.add_scalar(
            "global/train_loss",
            epoch_train_loss,
            epoch,
        )

        if early_stopping.has_converged():
            print("Model has converged. Stopping training.")
            break
    return model


def ofm_train_squad_seq2seq(
    args,
    model: OFM,
    train_dataset,
    val_dataset,
    test_dataset=None,
    processor=None,
    collate_fn=None,
    compute_metrics=None,
):
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    writer = SummaryWriter(os.path.join(args.save_dir, "logs"))
    best_acc = 0.0
    best_f1 = 0.0
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.save_dir, "downsize"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="no",
        save_strategy="no",
        num_train_epochs=args.num_local_epochs,
        learning_rate=args.lr,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        label_names=["labels"],
        fp16=args.fp16,
        weight_decay=0.01,
        predict_with_generate=True,
        # load_best_model_at_end=True,
    )
    eval_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.save_dir, "global"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="no",
        save_strategy="no",
        num_train_epochs=args.num_local_epochs,
        learning_rate=args.lr,
        # remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        label_names=["labels"],
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=384,
    )

    steps = 0

    train_dataset = train_dataset.shuffle()
    # train_dataset.select(range(1000))
    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        print("=" * 20 + "Training for epoch {}".format(epoch) + "=" * 20)

        # sharding the dataset to args.num_shards shards
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        size = len(indices) // args.num_shards
        mini_shards = [
            indices[i * size : (i + 1) * size] for i in range(args.num_shards)
        ]
        mini_shards[-1].extend(indices[args.num_shards * size :])

        epoch_train_loss = 0.0

        lr = step_lr(args.lr, epoch, args.step_size, 0.98)
        training_args.learning_rate = lr
        np.random.seed(int(time.time()))  # Set the seed to the current time

        if args.spp:
            model.salient_parameter_prioritization()
        avg_params = 0

        # Train each downsized model independently in a sequential manner
        for idx, mini_shard_idx in enumerate(
            tqdm(mini_shards, desc="Mini-shard training")
        ):
            steps += 1

            if idx == 0:
                ds_model = copy.deepcopy(model.model)
                ds_model_params = model.total_params
            elif idx == 1:
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

            local_grad = copy.deepcopy(ds_model.to("cpu").state_dict())

            trainer = Seq2SeqTrainer(
                model=ds_model,
                args=training_args,
                data_collator=collate_fn,
                # compute_metrics=compute_metrics,
                train_dataset=train_dataset.select(mini_shard_idx),
                # eval_dataset=val_dataset,
                tokenizer=processor,
            )

            train_results = trainer.train()
            epoch_train_loss += train_results.metrics["train_loss"]

            ds_model.to("cpu")
            import torch

            with torch.no_grad():
                for key in ds_model.state_dict():
                    local_grad[key] = local_grad[key] - ds_model.state_dict()[key]

            model.grad_accumulate(local_grad, alpha=len(mini_shard_idx))
            model.apply_grad(local_grad)

            if (steps % args.log_interval == 0) or (steps % args.num_shards == 0):
                # Evaluate the model

                print("*" * 20 + "Evaluating in train step {}".format(steps) + "*" * 20)

                trainer = Seq2SeqTrainer(
                    model=ds_model,
                    args=eval_args,
                    train_dataset=None,
                    # data_collator=data_collator,
                )
                ds_model.eval()
                predictions = trainer.predict(val_dataset)
                ds_model.to("cpu")
                metrics = compute_metrics(predictions)

                # metrics = trainer.evaluate(val_dataset)
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

                val_accuracy, val_f1_score = (
                    metrics["HasAns_exact"],
                    metrics["HasAns_f1"],
                )

                writer.add_scalar(
                    "steps/eval_metric",
                    val_accuracy,
                    steps,
                )
                writer.add_scalar(
                    "steps/eval_f1",
                    val_f1_score,
                    steps,
                )

                writer.add_scalar(
                    "steps/params",
                    ds_model_params,
                    steps,
                )

                # model.model.to("cpu")
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

        print("=" * 20 + "Training finished for epoch {}".format(epoch) + "=" * 20)

        avg_params = avg_params / len(mini_shards)
        writer.add_scalar(
            "global/params",
            avg_params,
            epoch,
        )

        writer.add_scalar(
            "global/train_loss",
            epoch_train_loss,
            epoch,
        )

        if early_stopping.has_converged():
            print("Model has converged. Stopping training.")
            break
    return model
"""
