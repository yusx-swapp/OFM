import os
import torch
import numpy as np
from datasets import load_dataset
import functools
import evaluate
from arguments import arguments
from ofm import OFM
from ofm.trainer import TrainingArguments
from ofm.trainer import CLIPTrainer as Trainer

import functools
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import torch


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
        predictions=np.argmax(eval_pred["predictions"], axis=1),
        references=eval_pred["label_ids"],
    )
    f1 = f1_metric.compute(
        predictions=np.argmax(eval_pred["predictions"], axis=1),
        references=eval_pred["label_ids"],
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
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def transform_train(example_batch, processor, label_to_text):
    # Take a list of PIL images and turn them to pixel values

    inputs = processor(
        text=[label_to_text[label] for label in example_batch["label"]],
        images=[x.convert("RGB") for x in example_batch["img"]],
        return_tensors="pt",
        padding=True,
    )
    inputs["labels"] = example_batch["label"]

    return inputs


def transform_eval(example_batch, processor, label_to_text):
    # Take a list of PIL images and turn them to pixel values

    images = [x.convert("RGB") for x in example_batch["img"]]

    # Generate text prompts for all possible labels
    text_prompts = [label_to_text[label] for label in range(len(label_to_text))]

    inputs = processor(
        text=text_prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    inputs["labels"] = example_batch["label"]

    return inputs


def main(args):
    if args.model == "vit":
        raise NotImplementedError("This script is for CLIP only")

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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

    label_to_text = {i: label for i, label in enumerate(labels)}

    prepared_train = dataset["train"].with_transform(
        functools.partial(
            transform_train, processor=processor, label_to_text=label_to_text
        )
    )
    prepared_test = dataset["test"].with_transform(
        functools.partial(
            transform_eval, processor=processor, label_to_text=label_to_text
        )
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

    model = OFM(model.to("cpu"), elastic_config)

    trainer = Trainer(
        model,
        TrainingArguments(
            output_dir=args.save_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            report_to=[],
            fp16=args.fp16,
            dataloader_num_workers=8,
            log_interval=args.log_interval,
        ),
        train_dataset=prepared_train,
        eval_dataset=prepared_test,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        optimizers=(None, None),
    )

    subnet, subnet.config.num_parameters, subnet.config.arch = model.smallest_model()

    metrics = trainer.train_subnet(subnet)

    model.save_ckpt(os.path.join(args.save_dir, "final"))


if __name__ == "__main__":
    args = arguments()
    main(args)
