import pandas as pd
import datasets
import evaluate
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

train = pd.read_parquet("s3://fti-model-training/test_training/train")
test = pd.read_parquet("s3://fti-model-training/test_training/test")

# covert to datasets and encode labels
ds = datasets.DatasetDict(
    {
        "train": datasets.Dataset.from_pandas(train),
        "test": datasets.Dataset.from_pandas(test),
    }
)

ds = ds.class_encode_column("label")

# store label information
id2label = dict(enumerate(ds["train"].features["label"].names))
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# set model checkpoint
model_ckpt = "microsoft/deberta-v3-small"  # first parameter choice

# tokenize data
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=512)


ds_encode = ds.map(tokenize, batched=True)

# load base model
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=num_labels
)

# load in evaluation metrics
accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="outputs",  # this is directory where checkpoitns are written
    overwrite_output_dir=True,
    eval_strategy="epoch",  # remaining training parameters
    learning_rate=0.00003,
    logging_steps=100,
    num_train_epochs=10,
    save_steps=500,
    weight_decay=0.0001,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=ds_encode["train"],
    eval_dataset=ds_encode["test"],
    compute_metrics=compute_metrics,
)

trainer.train()  # trains
trainer.evaluate()  # evaluate (won't write anywhere for now)
