import pandas as pd
import numpy as np
import os
import datasets
import wandb
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from skmultilearn.model_selection import iterative_train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    DataCollatorWithPadding,
)
from scipy.special import softmax
import torch
from torch import nn
from typing import Optional
import boto3
import io
from dotenv import load_dotenv
from dotenv import dotenv_values

# create function to create balenced classes
def balenced_split(df, test_size):
    ind = np.expand_dims(np.arange(len(df)), axis=1)
    labels = np.array(df.iloc[:, 1:])  # assumes text is first column
    ind_train, _, ind_test, _ = iterative_train_test_split(ind, labels, test_size)

    return df.iloc[ind_train[:, 0]], df.iloc[ind_test[:, 0]]


# function to convert labels to list of floats
def preprocess_data(batch):
    text = batch["text"]
    labels = [
        "Agriculture",
        "Consumer Preferences",
        "Labor",
        "Harm Reduction",
        "Historic Brand",
        "Lawsuits",
        "Lobbying",
        "Market Reports",
        "Scientific Studies",
        "Personnel",
        "Public Health",
        "Regulatory",
        "Responsible Marketing",
        "Sustainability",
        "Technology",
        "Youth Access",
    ]
    # add labels
    labels_batch = {k: batch[k] for k in batch.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    batch["labels"] = labels_matrix.tolist()

    return batch


def build_data(key):
    from dotenv import load_dotenv
    from dotenv import dotenv_values

    load_dotenv(".env", override=True)
    aws_config = dotenv_values(".env")

    bucket = "fti-adhoc-prod"
    key = "MLGuide/Example_Classification_Training_Data"
    aws_access_key_id = aws_config["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = aws_config["AWS_SECRET_ACCESS_KEY"]

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    labelled_data = pd.read_parquet(io.BytesIO(obj["Body"].read()))

    labelled_data = labelled_data[
        [
            "text",
            "Agriculture",
            "Consumer Preferences",
            "Labor",
            "Harm Reduction",
            "Historic Brand",
            "Lawsuits",
            "Lobbying",
            "Market Reports",
            "Scientific Studies",
            "Personnel",
            "Public Health",
            "Regulatory",
            "Responsible Marketing",
            "Sustainability",
            "Technology",
            "Youth Access",
        ]
    ]
    # only keep rows with at least one topic label
    labelled_data = labelled_data.loc[
        labelled_data.sum(axis=1, numeric_only=True) > 0
    ].reset_index(drop=True)
    labelled_data.sum(axis=0, numeric_only=True)

    train, test = balenced_split(labelled_data, 0.25)
    test, val = balenced_split(test, 0.6)

    # store label information
    labels = [l for l in train.columns if l != "text"]
    id2label = dict(enumerate(labels))
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    # pandas -> dataset
    ds = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_pandas(train.reset_index(drop=True)),
            "test": datasets.Dataset.from_pandas(test.reset_index(drop=True)),
            "val": datasets.Dataset.from_pandas(val.reset_index(drop=True)),
        }
    )

    ds = ds.map(preprocess_data, batched=True, remove_columns=labels)

    # make sure labels are in torch format
    ds.set_format("torch")

    return ds, labels, id2label, label2id, num_labels


# define evaluation metrics
def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


# def tokenize(batch):
#     return tokenizer(batch["text"], truncation=True, padding = True, max_length=500)

# Define Sweep Config
sweep_config = {
    "method": "bayes",
    "metric": {"name": "loss", "goal": "minimize"},
    "parameters": {
        "epoch": {"values": [5, 7, 10, 12, 15, 17, 20]},
        "batch_size": {
            "distribution": "q_log_uniform_values",
            "q": 8,
            "min": 32,
            "max": 128,
        },
        "learning_rate": {"min": 0.000001, "max": 0.1},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 5,
    },
}


def train(config=None):
    # initalize new run
    with wandb.init(config=config):

        config = wandb.config
        model_ckpt = "microsoft/deberta-base"

        # load and set up data
        ds, labels, id2label, label2id, num_labels = build_data(
            "MLGuide/Example_Classification_Training_Data"
        )

        # make sure labels are in torch format
        ds.set_format("torch")

        # laod tokenizer and encode data
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

        def tokenize(batch):
            return tokenizer(
                batch["text"], truncation=True, padding=True, max_length=500
            )

        ds_encode = ds.map(tokenize, batched=True, remove_columns=["text"])

        model = AutoModelForSequenceClassification.from_pretrained(
            model_ckpt,
            problem_type="multi_label_classification",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        training_args = TrainingArguments(
            report_to="wandb",
            output_dir="outputs",
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            learning_rate=config.learning_rate,
            logging_steps=100,
            num_train_epochs=config.epoch,
            save_steps=500,
            weight_decay=0,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=ds_encode["train"],
            eval_dataset=ds_encode["val"],
            compute_metrics=compute_metrics,
        )

        trainer.train()


# set up wandb
import wandb

wandb.login()

project_name = "llm-multiclass-finetuning-example"  # @param
entity = "demo"  # @param
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# define sweep
sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)

# run sweep
wandb.agent(sweep_id, train, count=20)
