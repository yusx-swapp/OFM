import copy
import os
import time
import numpy as np
from .utils import EarlyStopping, step_lr
from .modeling_rafm import RAFM
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer
from tqdm import tqdm


def rafm_train(
    args,
    model: RAFM,
    data_shards,
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
        # load_best_model_at_end=True,
    )
    eval_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "global"),
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
        # load_best_model_at_end=True,
    )
    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        epoch_train_loss = 0.0

        lr = step_lr(args.lr, epoch, args.step_size, 0.98)
        training_args.learning_rate = lr
        np.random.seed(int(time.time()))  # Set the seed to the current time

        if args.spp:
            model.salient_parameter_prioritization()
        avg_params = 0

        # shuffle the data shards
        np.random.shuffle(data_shards)
        # Train each downsized model independently in a sequential manner
        for idx, data_shard in enumerate(tqdm(data_shards, desc="Mini-shard training")):
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

            trainer = Trainer(
                model=ds_model,
                args=training_args,
                data_collator=collate_fn,
                compute_metrics=compute_metrics,
                train_dataset=data_shard,
                eval_dataset=val_dataset,
                tokenizer=processor,
            )
            train_results = trainer.train()

            epoch_train_loss += train_results.metrics["train_loss"]

            ds_model.to("cpu")
            import torch

            with torch.no_grad():
                for key in ds_model.state_dict():
                    local_grad[key] = local_grad[key] - ds_model.state_dict()[key]

            model.grad_accumulate(local_grad, alpha=data_shard.num_rows)
            model.apply_grad(local_grad)

        avg_params = avg_params / len(data_shards)
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

        if epoch % args.log_interval == 0:
            # Evaluate the model

            print(f"Training finished for epoch {epoch}")

            print("*" * 20 + "Evaluating in epoch {}".format(epoch) + "*" * 20)
            # Evaluate the model at the end of each epoch
            trainer = Trainer(
                model=model.model,
                args=eval_args,
                data_collator=collate_fn,
                compute_metrics=compute_metrics,
                train_dataset=None,
                eval_dataset=val_dataset,
                tokenizer=processor,
            )

            metrics = trainer.evaluate(val_dataset)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            val_accuracy, val_f1_score = metrics["eval_accuracy"], metrics["eval_f1"]

            writer.add_scalar(
                "global/eval_accuracy",
                val_accuracy,
                epoch,
            )
            writer.add_scalar(
                "global/eval_f1",
                val_f1_score,
                epoch,
            )

            model.model.to("cpu")
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                model.save_ckpt(os.path.join(args.save_dir, "best_model"))
            if val_f1_score > best_f1:
                best_f1 = val_f1_score
            writer.add_scalar(
                "global/best_accuracy",
                best_acc,
                epoch,
            )
            writer.add_scalar(
                "global/best_f1",
                best_f1,
                epoch,
            )

            print(
                f"Best validation accuracy: {best_acc} \nBest validation f1: {best_f1} \nEpoch train loss: {epoch_train_loss}"
            )
            print("*" * 20 + "Evaluation finished" + "*" * 20)

            if test_dataset:
                metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
                trainer.log_metrics("test", metrics)
                trainer.save_metrics("test", metrics)

            early_stopping(val_f1_score)

        model.save_ckpt(os.path.join(args.save_dir, "last_model"))

        # Apply the aggregated and normalized gradient to the full-size model
        model.apply_accumulate_grad(args.grad_beta)

        if early_stopping.has_converged():
            print("Model has converged. Stopping training.")
            break
    return model


def rafm_train_imagenet(
    args,
    model: RAFM,
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
        # load_best_model_at_end=True,
    )
    eval_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "global"),
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

            local_grad = copy.deepcopy(ds_model.to("cpu").state_dict())

            trainer = Trainer(
                model=ds_model,
                args=training_args,
                data_collator=collate_fn,
                compute_metrics=compute_metrics,
                train_dataset=train_dataset.select(mini_shard_idx),
                eval_dataset=val_dataset,
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

            if steps % args.log_interval == 0:
                # Evaluate the model

                print("*" * 20 + "Evaluating in train step {}".format(steps) + "*" * 20)

                trainer = Trainer(
                    model=ds_model,
                    args=eval_args,
                    data_collator=collate_fn,
                    compute_metrics=compute_metrics,
                    train_dataset=None,
                    eval_dataset=val_dataset,
                    tokenizer=processor,
                )

                metrics = trainer.evaluate(val_dataset)
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)
                val_accuracy, val_f1_score = (
                    metrics["eval_accuracy"],
                    metrics["eval_f1"],
                )
                ds_model.to("cpu")

                writer.add_scalar(
                    "steps/eval_accuracy",
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
                    "global/best_accuracy",
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
                model.apply_accumulate_grad(args.grad_beta)

            model.save_ckpt(os.path.join(args.save_dir, "last_model"))

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


def ofm_train(
    args,
    model: RAFM,
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
        # load_best_model_at_end=True,
    )
    eval_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "global"),
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

            local_grad = copy.deepcopy(ds_model.to("cpu").state_dict())

            trainer = Trainer(
                model=ds_model,
                args=training_args,
                data_collator=collate_fn,
                compute_metrics=compute_metrics,
                train_dataset=train_dataset.select(mini_shard_idx),
                eval_dataset=val_dataset,
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

                trainer = Trainer(
                    model=ds_model,
                    args=eval_args,
                    data_collator=collate_fn,
                    compute_metrics=compute_metrics,
                    train_dataset=None,
                    eval_dataset=val_dataset,
                    tokenizer=processor,
                )

                metrics = trainer.evaluate(val_dataset)
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

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
                model.apply_accumulate_grad(args.grad_beta)

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
