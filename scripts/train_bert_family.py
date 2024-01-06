import os
import time
import numpy as np
from datasets import load_dataset, concatenate_datasets
import functools
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



# set no_deprecation_warning to True to avoid warning messages
def compute_metrics(eval_pred, task):
    predictions, labels = eval_pred
    if task == "stsb":
        pearson_corr, _ = pearsonr(predictions.squeeze(), labels)
        return {"pearson_corr": pearson_corr}
    elif task == "cola":
        probabilities_class_1 = predictions[:, 1]
        # Convert continuous predictions to binary (0 or 1) for the CoLA task
        binary_predictions = (probabilities_class_1 > 0.5).astype(int)
        return {"matthews_corr": matthews_corrcoef(labels, binary_predictions)}
    else:
        predictions = predictions.argmax(-1)
        return {"accuracy": accuracy_score(labels, predictions)}


def evaluate(args, global_model, tokenized_test_dataset):
    # tokenized_test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer, args.dataset, args.model), batched=True)

    training_args = TrainingArguments(
        args.log_dir,
        logging_dir=args.log_dir,
        logging_steps=1000,
        save_strategy="no",
        evaluation_strategy="no",
    )

    global_model.to("cuda")  # Move the global model to GPU memory for evaluation
    # global_model = torch.compile(global_model)
    trainer = Trainer(
        model=global_model,
        args=training_args,
    )

    predictions = trainer.predict(tokenized_test_dataset)
    true_labels = tokenized_test_dataset["label"]
    true_labels = np.array(tokenized_test_dataset["label"])

    global_model.to("cpu")  # Move the global model back to CPU memory after evaluation

    if args.dataset == "stsb":
        pearson_corr = compute_metrics(
            (predictions.predictions, true_labels), args.dataset
        )["pearson_corr"]
        print(f"Pearson correlation: {pearson_corr}")
        return pearson_corr
    elif args.dataset == "cola":
        probabilities_class_1 = predictions.predictions[:, 1]
        binary_predictions = (probabilities_class_1 > 0.5).astype(int)
        matthews_corr = matthews_corrcoef(true_labels, binary_predictions)

        print(f"matthews correlation: {matthews_corr}")
        return matthews_corr

    else:
        predicted_labels = predictions.predictions.argmax(-1)
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Accuracy: {accuracy}")
        return accuracy

def tokenize_function(examples, tokenizer):
    if "sentence" in examples.keys() and "question" not in examples.keys():
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    elif "premise" in examples.keys() and "hypothesis" in examples.keys():
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    elif "question1" in examples.keys() and "question2" in examples.keys():
        return tokenizer(
            examples["question1"],
            examples["question2"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    elif "question" in examples.keys() and "sentence" in examples.keys():
        return tokenizer(
            examples["question"],
            examples["sentence"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    elif "sentence1" in examples.keys() and "sentence2" in examples.keys():
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )


def main(args):
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
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        global_model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes[args.dataset]
        )

    elif args.model == "roberta":
        model_name = "roberta-base"
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        global_model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes[args.dataset]
        )

    elif args.model == "t5":
        model_name = "t5-small"  # You can also use "t5-base" or other T5 variants
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        global_model = T5ForConditionalGeneration.from_pretrained(model_name)
    elif args.model == "bert-base":
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        global_model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes[args.dataset]
        )

    elif args.model == "bert-large":
        model_name = "bert-large-uncased-whole-word-masking"
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        global_model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes[args.dataset]
        )

    # load data and preprocess
    dataset = load_dataset("glue", args.dataset)

    if args.dataset == "mnli":
        dataset["validation"] = concatenate_datasets(
            [dataset["validation_matched"], dataset["validation_mismatched"]]
        )

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    tokenize_val_dataset = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
    )

    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
    )

    splitter = DatasetSplitter(tokenized_train_dataset, seed=123)

    tokenized_local_datasets = splitter.split(
        args.num_clients, k_shot=args.k_shot, replacement=False
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

    global_model = RaFFM(global_model.to("cpu"), elastic_config)
    global_model = federated_learning(
        args, global_model, tokenized_local_datasets, tokenize_val_dataset
    )
    global_model.save_ckpt(os.path.join(args.save_dir, args.dataset, "final"))


if __name__ == "__main__":
    args = arguments()
    main(args)
