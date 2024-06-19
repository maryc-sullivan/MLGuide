import pandas as pd
import numpy as np
import os
import datasets
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from skmultilearn.model_selection import iterative_train_test_split
from scipy.special import softmax
import torch
from torch import nn
from typing import Optional
from setfit import TrainingArguments, Trainer, SetFitModel
from transformers.integrations import WandbCallback
import wandb
import evaluate

### Have to use !pip install transformers==4.39.0

# set up wandb
wandb.login()

project_name = "llm-multiclass-finetuning-example"  # @param
entity = "demo"  # @param
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# load data and keep relevant columns
labelled_data = pd.read_parquet(
    "s3://fti-adhoc-prod/MLGuide/Example_Classification_Training_Data"
)
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

# NOTE: we only have few labels for Labor (24), Agriculture (50), Personel (65), and Technology (72)
# create function to create balenced classes
def balenced_split(df, test_size):
    ind = np.expand_dims(np.arange(len(df)), axis=1)
    labels = np.array(df.iloc[:, 1:])  # assumes text is first column
    ind_train, _, ind_test, _ = iterative_train_test_split(ind, labels, test_size)

    return df.iloc[ind_train[:, 0]], df.iloc[ind_test[:, 0]]


train, test = balenced_split(labelled_data, 0.25)
test, val = balenced_split(test, 0.6)


# take a look at class distribution
len(train)
train.sum(axis=0, numeric_only=True)

len(val)
val.sum(axis=0, numeric_only=True)

len(test)
test.sum(axis=0, numeric_only=True)

# store label information
labels = [l for l in train.columns if l != "text"]
id2label = dict(enumerate(labels))
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# function to convert labels to list of floats
def preprocess_data(batch):
    # take a batch of texts
    text = batch["text"]
    # add labels
    labels_batch = {k: batch[k] for k in batch.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    batch["labels"] = labels_matrix.tolist()

    return batch


# create a smaller train set based on original train set from fine-tune-multilabel-classifier
train.sum(axis=0, numeric_only=True)
samples_per_class = 20
setfit_train = pd.DataFrame()

for label in labels:
    subset = train[train[label] == True].reset_index(drop=True)
    if len(subset) > samples_per_class:
        subset = subset.sample(samples_per_class, random_state=42, replace=False)
        setfit_train = pd.concat([setfit_train, subset], axis=0)
    else:
        setfit_train = pd.concat([setfit_train, subset], axis=0)

setfit_train.drop_duplicates(inplace=True, ignore_index=True)

ds = datasets.DatasetDict(
    {
        "train": datasets.Dataset.from_pandas(setfit_train.reset_index(drop=True)),
        "test": datasets.Dataset.from_pandas(test.reset_index(drop=True)),
        "val": datasets.Dataset.from_pandas(val.reset_index(drop=True)),
    }
)

ds = ds.map(preprocess_data, batched=True, remove_columns=labels)

# setfit evaluation metrics
multilabel_f1_metric = evaluate.load("f1", "multilabel")
multilabel_accuracy_metric = evaluate.load("accuracy", "multilabel")


def compute_setfit_metrics(y_pred, y_test):
    return {
        "f1": multilabel_f1_metric.compute(
            predictions=y_pred, references=y_test, average="micro"
        )["f1"],
        "accuracy": multilabel_accuracy_metric.compute(
            predictions=y_pred, references=y_test
        )["accuracy"],
    }


# load model
model_ckpt = "sentence-transformers/paraphrase-mpnet-base-v2"

model = SetFitModel.from_pretrained(
    model_ckpt,
    multi_target_strategy="one-vs-rest",
    use_differentiable_head=True,
    head_params={"out_features": num_labels},
    device="mps",
)

wandb.init(project=project_name, entity=entity, job_type="training")

training_args = TrainingArguments(
    output_dir="outputs",
    batch_size=(16, 8),
    num_epochs=(1, 5),
    body_learning_rate=(2e-5, 1e-5),
    head_learning_rate=1e-2,
    logging_strategy="steps",
    logging_steps=100,
    max_length=512,
    save_steps=500,
    num_iterations=20,
    evaluation_strategy="epoch",
    sampling_strategy="undersampling",
    report_to="wandb",
    end_to_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    column_mapping={"text": "text", "labels": "label"},
    metric=compute_setfit_metrics,
)

trainer.train()
wandb.finish()
