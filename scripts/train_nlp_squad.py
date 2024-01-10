import copy
import functools
from datasets import load_dataset
import os
import collections
from tqdm.auto import tqdm
import numpy as np
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import evaluate
import time

from arguments import arguments
from rafm import trainer
from rafm.trainer import ofm_train_squad
from rafm import RAFM

random_seed = 123


def step_lr(initial_lr, epoch, decay_step, decay_rate):
    return initial_lr * (decay_rate ** (epoch // decay_step))


def compute_metrics(eval_pred, features, examples):
    start_logits, end_logits = eval_pred.predictions
    n_best = 20
    max_answer_length = 30

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)
            # If no valid answer, predict 'no_answer'
        if len(answers) == 0:
            predicted_answers.append(
                {
                    "id": example_id,
                    "prediction_text": "",
                    "no_answer_probability": 1.0,  # Assuming you're certain there's no answer
                }
            )
        else:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {
                    "id": example_id,
                    "prediction_text": best_answer["text"],
                    "no_answer_probability": 0.0,  # Assuming you're certain there is an answer
                }
            )
    metric = evaluate.load("squad_v2")
    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


def tokenize_function(examples, tokenizer):
    examples["question"] = [q.lstrip() for q in examples["question"]]
    return tokenizer(examples["question"], examples["context"], truncation=True)


def prepare_train_features(examples, tokenizer):
    examples["question"] = [q.lstrip() for q in examples["question"]]
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def preprocess_validation_examples(examples, tokenizer):
    examples["question"] = [q.lstrip() for q in examples["question"]]
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def main(args):
    if args.model == "distilbert":
        model_name = "distilbert-base-uncased"

    elif args.model == "roberta":
        # raise NotImplementedError
        model_name = "roberta-base"

    elif args.model == "t5":
        raise NotImplementedError

    elif args.model == "bert-base":
        model_name = "bert-base-uncased"

    elif args.model == "bert-large":
        model_name = "bert-large-uncased"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, cache_dir=args.cache_dir
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name, cache_dir=args.cache_dir
    )

    datasets = load_dataset("squad_v2")

    train_dataset = datasets["train"]
    val_dataset = datasets["validation"]

    tokenize_val_dataset = datasets["validation"].map(
        lambda examples: preprocess_validation_examples(examples, tokenizer),
        batched=True,
        remove_columns=datasets["validation"].column_names,
    )

    tokenized_train_dataset = train_dataset.with_transform(
        lambda examples: prepare_train_features(examples, tokenizer),
        # batched=True,
    )

    # predictions, _, _ = eval_trainer.predict(tokenize_val_dataset)
    # start_logits, end_logits = predictions
    # res = compute_metrics(
    #     start_logits, end_logits, tokenize_val_dataset, datasets["validation"]
    # )
    # print("Global validation results: ", res)

    # predictions, _, _ = eval_trainer.predict(tokenize_val_dataset)
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

    model = ofm_train_squad(
        args,
        model,
        tokenized_train_dataset,
        val_dataset=tokenize_val_dataset,
        processor=None,
        collate_fn=None,
        compute_metrics=functools.partial(
            compute_metrics,
            features=tokenize_val_dataset,
            examples=datasets["validation"],
        ),
    )

    model.save_ckpt(os.path.join(args.save_dir, args.dataset, "final"))


if __name__ == "__main__":
    args = arguments()
    main(args)

# python fl_qa_squadv2.py --algo vanilla --save_model --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --dataset squad_v2 --log_dir log_squadv2_roberta --model roberta > raffm_squadv2_roberta_100.txt
# python fl_qa_squadv2.py --save_model --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --dataset squad_v2 --log_dir log_squadv2_distilbert --model distilbert > raffm_squadv2_distilbert_100.txt


# python fl_qa_squadv2.py --save_model --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --dataset squad_v2 --log_dir log_squadv2_bert_large --model bert-large > raffm_squadv2_bertlarge_100.txt
# sbatch --gres=gpu:1 --wrap="python3 fl_qa_squadv2.py --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --dataset squad_v2 --log_dir suqadv2/100 --model bert-base --per_device_train_batch_size 16 --per_device_eval_batch_size 16"
