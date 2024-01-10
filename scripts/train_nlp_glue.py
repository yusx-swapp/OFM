import os
import time
import numpy as np
from datasets import load_dataset, concatenate_datasets
import functools
from rafm.trainer import ofm_train
from torch.utils.tensorboard import SummaryWriter
import copy
import argparse
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    RobertaTokenizerFast,
    T5Tokenizer,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizerFast,
    DistilBertForSequenceClassification,
    RobertaForSequenceClassification,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from rafm.utils import DatasetSplitter, step_lr, EarlyStopping
from rafm import RAFM
from arguments import arguments
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, matthews_corrcoef
import evaluate


# set no_deprecation_warning to True to avoid warning messages
def compute_metrics(eval_pred, task):
    predictions, labels = eval_pred
    f1_metric = evaluate.load("f1")

    if task == "stsb":
        pearsonr = evaluate.load("pearsonr")
        results = pearsonr.compute(
            predictions=predictions.squeeze(), references=labels.squeeze()
        )
        f1 = f1_metric.compute(
            predictions=predictions.squeeze(),
            references=labels.squeeze(),
            average="weighted",
        )
        return {"metric": results["pearsonr"], "f1": f1["f1"]}

    elif task == "cola":
        probabilities_class_1 = predictions[:, 1]
        # Convert continuous predictions to binary (0 or 1) for the CoLA task
        binary_predictions = (probabilities_class_1 > 0.5).astype(int)

        matthews_metric = evaluate.load("matthews_correlation")
        results = matthews_metric.compute(labels, binary_predictions)

        f1 = f1_metric.compute(
            predictions=binary_predictions,
            references=labels,
            average="weighted",
        )

        return {"metric": results["matthews_correlation"], "f1": f1["f1"]}
    else:
        accuracy_metric = evaluate.load("accuracy")
        predictions = predictions.argmax(-1)

        results = accuracy_metric.compute(
            predictions=predictions,
            references=labels,
        )
        f1 = f1_metric.compute(
            predictions=predictions,
            references=labels,
            average="weighted",
        )
        return {"metric": results["accuracy"], "f1": f1["f1"]}


def tokenize_function(examples, tokenizer):
    if "sentence" in examples.keys() and "question" not in examples.keys():
        inputs = tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    elif "premise" in examples.keys() and "hypothesis" in examples.keys():
        inputs = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    elif "question1" in examples.keys() and "question2" in examples.keys():
        inputs = tokenizer(
            examples["question1"],
            examples["question2"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    elif "question" in examples.keys() and "sentence" in examples.keys():
        inputs = tokenizer(
            examples["question"],
            examples["sentence"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    elif "sentence1" in examples.keys() and "sentence2" in examples.keys():
        inputs = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    inputs["labels"] = examples["label"]
    return inputs


def main(args):
    if args.huggingface_token:
        from huggingface_hub import login

        login(args.huggingface_token)

    num_classes = {
        "mnli": 3,
        "qqp": 2,
        "qnli": 2,
        "sst2": 2,
        "stsb": 1,
        "mrpc": 2,
        "rte": 2,
        "cola": 2,
    }

    if args.model == "distilbert":
        model_name = "distilbert-base-uncased"
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            model_name, cache_dir=args.cache_dir
        )
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes[args.dataset], cache_dir=args.cache_dir
        )

    elif args.model == "roberta":
        model_name = "roberta-base"
        tokenizer = RobertaTokenizerFast.from_pretrained(
            model_name, cache_dir=args.cache_dir
        )
        model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes[args.dataset], cache_dir=args.cache_dir
        )

    elif args.model == "t5":
        model_name = "t5-small"  # You can also use "t5-base" or other T5 variants
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, cache_dir=args.cache_dir
        )
    elif args.model == "bert-base":
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(
            model_name, cache_dir=args.cache_dir
        )
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes[args.dataset],
            ignore_mismatched_sizes=True,
            cache_dir=args.cache_dir,
        )

    elif args.model == "bert-large":
        model_name = "bert-large-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(
            model_name, cache_dir=args.cache_dir
        )
        model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes[args.dataset], cache_dir=args.cache_dir
        )

    # load data and preprocess
    dataset = load_dataset("glue", args.dataset, cache_dir=args.cache_dir)

    if args.dataset == "mnli":
        dataset["validation"] = concatenate_datasets(
            [dataset["validation_matched"], dataset["validation_mismatched"]]
        )

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    tokenize_val_dataset = val_dataset.with_transform(
        lambda examples: tokenize_function(examples, tokenizer),
        # batched=True,
    )

    tokenized_train_dataset = train_dataset.with_transform(
        lambda examples: tokenize_function(examples, tokenizer),
        # batched=True,
    )

    # splitter = DatasetSplitter(tokenized_train_dataset, seed=123)

    # tokenized_local_datasets = splitter.split(
    #     args.num_clients, k_shot=args.k_shot, replacement=False
    # )

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

    model = RAFM(model.to("cpu"), elastic_config)

    model = ofm_train(
        args,
        model,
        tokenized_train_dataset,
        tokenize_val_dataset,
        processor=tokenizer,
        collate_fn=None,
        compute_metrics=functools.partial(compute_metrics, task=args.dataset),
    )

    model.save_ckpt(os.path.join(args.save_dir, args.dataset, "final"))


if __name__ == "__main__":
    args = arguments()
    main(args)
