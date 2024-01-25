import os
import torch
import numpy as np
from datasets import load_dataset
import functools
import evaluate
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
)
from arguments import arguments
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageDistilTrainer(Trainer):
    def __init__(
        self,
        teacher_model=None,
        student_model=None,
        temperature=None,
        lambda_param=None,
        *args,
        **kwargs
    ):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher.to(device)
        self.teacher.eval()
        self.temperature = temperature
        self.lambda_param = lambda_param

    def compute_loss(self, student, inputs, return_outputs=False):
        student_output = self.student(**inputs)

        with torch.no_grad():
            teacher_output = self.teacher(**inputs)

        # Compute soft targets for teacher and student
        soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)

        # Compute the loss
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (
            self.temperature**2
        )

        # Compute the true label loss
        student_target_loss = student_output.loss

        # Calculate final loss
        loss = (
            1.0 - self.lambda_param
        ) * student_target_loss + self.lambda_param * distillation_loss
        return (loss, student_output) if return_outputs else loss


# from rafm import RAFM, ofm_train


def compute_metrics(eval_pred):
    """This function is used to compute the metrics for the evaluation.

    Args:
        eval_pred: The output of the Trainer.evaluate function
    returns:
        A dictionary of metrics
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    accuracy = accuracy_metric.compute(
        predictions=np.argmax(eval_pred.predictions, axis=1),
        references=eval_pred.label_ids,
    )
    f1 = f1_metric.compute(
        predictions=np.argmax(eval_pred.predictions, axis=1),
        references=eval_pred.label_ids,
        average="weighted",
    )

    return {"metric": accuracy["accuracy"], "f1": f1["f1"]}


def collate_fn(batch):
    """This function is used to collate the data samples into batches.
    It is used to supply the DataLoader with the collate_fn argument.

    Args:
        batch: A list of samples from the dataset
    returns:
        A dictionary of tensors containing the batched samples
    """
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def transform(example_batch, processor):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor(
        [x.convert("RGB") for x in example_batch["img"]], return_tensors="pt"
    )

    # Include the labels
    inputs["labels"] = example_batch["label"]
    return inputs


def main(args):
    if args.model == "vit":
        model_name = "google/vit-base-patch16-224-in21k"
        processor_name = "google/vit-base-patch16-224-in21k"
    elif args.model == "vit-large":
        model_name = "google/vit-base-patch16-224-in21k"
        processor_name = "google/vit-base-patch16-224-in21k"

    # load data and preprocess

    if args.huggingface_token:
        from huggingface_hub import login

        login(args.huggingface_token)

    dataset = load_dataset(
        args.dataset, cache_dir=args.cache_dir, trust_remote_code=True
    )

    if args.dataset == "imagenet-1k":
        assert (
            args.huggingface_token is not None
        ), "Please provide a HuggingFace token to download the ImageNet dataset"
        dataset = dataset.rename_column("image", "img")

    if args.dataset in ["cifar100", "cifar10"]:
        if args.dataset == "cifar100":
            dataset = dataset.rename_column("fine_label", "label")

        train_val = dataset["train"].train_test_split(
            test_size=0.2, stratify_by_column="label", seed=123
        )
        dataset["train"] = train_val["train"]
        dataset["validation"] = train_val["test"]

    labels = dataset["train"].features["label"].names

    processor = ViTImageProcessor.from_pretrained(processor_name)
    prepared_ds = dataset.with_transform(
        functools.partial(transform, processor=processor)
    )

    # load/initialize global model and convert to raffm model
    if args.resume_ckpt:
        ckpt_path = args.resume_ckpt
        elastic_config = (
            os.path.join(ckpt_path, "elastic_space.json")
            if os.path.exists(os.path.join(ckpt_path, "elastic_space.json"))
            else args.elastic_config
        )

    else:
        ckpt_path = model_name
        elastic_config = args.elastic_config

    model = ViTForImageClassification.from_pretrained(
        ckpt_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float16,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "downsize"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_total_limit=1,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        label_names=["labels"],
        # load_best_model_at_end=True,
        # fp16=True,
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=collate_fn,
    #     compute_metrics=compute_metrics,
    #     train_dataset=prepared_ds["train"],
    #     eval_dataset=prepared_ds["validation"],
    #     tokenizer=processor,
    # )
    student_model = model
    teacher_model_name = "google/vit-large-patch16-224"
    # teacher_processor_name = "google/vit-large-patch16-224"

    teacher_model = ViTForImageClassification.from_pretrained(
        teacher_model_name,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float16,
    )

    trainer = ImageDistilTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        training_args=training_args,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["validation"],
        data_collator=collate_fn,
        tokenizer=processor,
        compute_metrics=compute_metrics,
        temperature=5,
        lambda_param=0.5,
    )
    train_results = trainer.train()
    print(train_results)


if __name__ == "__main__":
    args = arguments()
    main(args)

# python train_vit.py --model vit --save_dir ckpts/vit-base  --dataset cifar100 --num_shards 20 --elastic_config scripts/elastic_space.json
