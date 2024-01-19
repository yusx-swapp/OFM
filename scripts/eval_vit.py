import numpy as np
from datasets import load_metric
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from rafm import RAFM
import torch


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    inputs = processor([x for x in example_batch["img"]], return_tensors="pt")

    # Don't forget to include the labels!
    inputs["labels"] = example_batch["label"]
    return inputs


def compute_metrics(p):
    metric = load_metric("accuracy")
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )


dataset = load_dataset("cifar100")
dataset = dataset.rename_column("fine_label", "label")

train_val = dataset["train"].train_test_split(test_size=0.2, seed=123)

dataset["train"] = train_val["train"]
dataset["validation"] = train_val["test"]

prepared_ds = dataset.with_transform(transform)


# use our pretrained ckpts: https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-cifar100
ckpt_path = "yusx-swapp/ofm-vit-base-patch16-224-cifar100"
labels = dataset["train"].features["label"].names

model = ViTForImageClassification.from_pretrained(
    ckpt_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
)

training_args = TrainingArguments(
    output_dir="./eval/vit-cifar100",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=128,
    evaluation_strategy="steps",
    num_train_epochs=4,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="tensorboard",
    load_best_model_at_end=True,
    dataloader_num_workers=8,
)


raffm_model = RAFM(model)
print("Original FM number of parameters:", raffm_model.total_params)

scaled_model, params, _ = raffm_model.smallest_model()


trainer = Trainer(
    model=scaled_model.to("cuda"),
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    # tokenizer=processor,
)

metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval-size-{}".format(params), metrics)


for i in range(0, 120):
    scaled_model, params, _ = raffm_model.random_resource_aware_model()
    trainer = Trainer(
        model=scaled_model.to("cuda"),
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["validation"],
        # tokenizer=processor,
    )
    print("downsize model params:", params)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval-size-{}".format(params), metrics)
