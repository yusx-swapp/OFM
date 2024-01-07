import os
import torch
import numpy as np
from datasets import load_dataset
import functools
import evaluate
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
)
from arguments import arguments
from rafm.utils import DatasetSplitter
from rafm import RAFM, rafm_train


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

    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}


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
        model_name = "google/vit-base-patch16-224"
        # model_name = "edumunozsala/vit_base-224-in21k-ft-cifar100"
        # model_name = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
        processor_name = "google/vit-base-patch16-224"
    elif args.model == "vit-large":
        model_name = "google/vit-base-patch16-224"
        processor_name = "google/vit-base-patch16-224"

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

    splitter = DatasetSplitter(dataset["train"], seed=123)

    mini_shards = splitter.split(args.num_shards, k_shot=args.k_shot, replacement=False)

    for i, mini_shard in enumerate(mini_shards):
        mini_shards[i] = mini_shard.with_transform(
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
    )

    model = RAFM(model.to("cpu"), elastic_config)
    model = rafm_train(
        args,
        model,
        mini_shards,
        prepared_ds["validation"],
        processor=processor,
        collate_fn=collate_fn,
        compute_metrics=compute_metrics,
    )
    model.save_ckpt(os.path.join(args.save_dir, "final"))


if __name__ == "__main__":
    args = arguments()
    main(args)

# python train_vit.py --model vit --save_dir ckpts/vit-base  --dataset cifar100 --num_shards 20 --elastic_config scripts/elastic_space.json
