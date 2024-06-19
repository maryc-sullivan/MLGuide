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

# set up wandb
wandb.login()

project_name = "llm-multiclass-finetuning-example"  # @param
entity = "demo"  # @param
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# let's log file in s3 we are using for training data
# run = wandb.init(project=project_name, entity=entity, job_type="data")
# artifact = wandb.Artifact("multi-label", type="dataset")
# artifact.add_reference("s3://fti-adhoc-prod/MLGuide/Example_Classification_Training_Data")
# run.log_artifact(artifact)
# wandb.finish()

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

len(labelled_data)

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

# # store label information
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

# check labels
[id2label[idx] for idx, label in enumerate(ds["train"]["labels"][1]) if label == 1.0]

# define evaluation metrics
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


# set model checkpoint
model_ckpt = "microsoft/deberta-base"  # roberta vocab

# load tokenizer and encode data
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=500)


ds_encode = ds.map(tokenize, batched=True, remove_columns=["text"])

ds_encode["train"]["labels"][0]
ds_encode["train"]["input_ids"][0]

# load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    problem_type="multi_label_classification",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

wandb.init(project=project_name, entity=entity, job_type="training")

wandb.run

training_args = TrainingArguments(
    report_to="wandb",
    output_dir="outputs",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    # group_by_length=True,
    learning_rate=0.00003,
    logging_steps=100,
    num_train_epochs=5,
    save_steps=500,
    weight_decay=0.0001,
    per_device_train_batch_size=8,
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

# trainer.evaluate()  # verify compute metrics works
trainer.train()
wandb.finish()

# log run in model registry
last_run_id = "vnpmjzxf"  # @param
wandb.init(project=project_name, entity=entity, job_type="registering_best_model")
best_model = wandb.use_artifact(f"{entity}/{project_name}/model-{last_run_id}:latest")
registered_model_name = "reynolds_multi:v0"  # @param {type: "string"}
wandb.run.link_artifact(
    best_model, f"{entity}/model-registry/{registered_model_name}", aliases=["staging"]
)
wandb.finish()


# download model
with wandb.init(project=project_name, entity=entity) as run:
    # Pass the name and version of Artifact
    my_model_artifact = run.use_artifact(best_model)

    # Download model weights to a folder and return the path
    model_dir = my_model_artifact.download()

    # Load your Hugging Face model from that folder
    #  using the same model class
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, problem_type="multi_label_classification", num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)


# create prediction table
import transformers

text_classification_pipeline = transformers.pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=False,
)

table = pd.DataFrame()

for i in range(len(ds["test"])):
    sequence = ds["test"]["text"][i]
    results = text_classification_pipeline(sequence, top_k=num_labels)
    results = pd.DataFrame(results)
    results_label = results["label"].tolist()
    results_label = [c + "_score" for c in results_label]
    results_score = results["score"].tolist()
    results = dict(zip(results_label, results_score))

    truth_scores = ds["test"]["labels"][i].detach().tolist()
    truth = dict(zip(labels, truth_scores))

    results.update(truth)
    results = pd.DataFrame(results, index=[0])
    results = results.reindex(sorted(results.columns), axis=1)

    input = pd.DataFrame({"id": [i], "text": [sequence]})
    input = pd.concat([input, results], axis=1)

    table = pd.concat([table, input], axis=0)
    print(i)


run = wandb.init(project=project_name, entity=entity, job_type="inference")
wandb_table = wandb.Table(dataframe=table)
wandb.log({"inference_table": wandb_table})
wandb.finish()

# create a long table
cols = ["id"] + ["text"] + labels
table2 = pd.melt(table, id_vars=cols)

run = wandb.init(project=project_name, entity=entity, job_type="inference")
wandb_table = wandb.Table(dataframe=table2)
wandb.log({"inference_table_2": wandb_table})
wandb.finish()

# scores by ground truth
variables = [l + "_score" for l in labels]
table3 = pd.DataFrame()

for i in range(len(labels)):
    temp = table2[
        (table2[labels[i]] == 1) & (table2["variable"] == variables[i])
    ].reset_index(drop=True)
    temp["label"] = labels[i]
    temp = temp[["id", "text", "label", "value"]]
    table3 = pd.concat([table3, temp], axis=0)

table3.reset_index(drop=True, inplace=True)
table3["correct"] = np.where(table3["value"] >= 0.5, 1, 0)

run = wandb.init(project=project_name, entity=entity, job_type="inference")
wandb_table = wandb.Table(dataframe=table3)
wandb.log({"prediction_score_by_label": wandb_table})
wandb.finish()


# per class accuracy
# if prediction > .5 then
table4 = pd.DataFrame()

for i in range(len(labels)):
    temp = table[["id", "text", labels[i], variables[i]]]
    temp["label"] = labels[i]
    temp["truth"] = temp[labels[i]]
    temp["prediction"] = temp[variables[i]]
    temp["prediction"] = np.where(temp["prediction"] >= 0.5, 1, 0)
    temp = temp[["id", "text", "label", "truth", "prediction"]]
    table4 = pd.concat([table4, temp], axis=0)

table4["TP"] = np.where((table4["truth"] == 1) & (table4["prediction"] == 1), 1, 0)
table4["FP"] = np.where((table4["truth"] == 0) & (table4["prediction"] == 1), 1, 0)
table4["TN"] = np.where((table4["truth"] == 0) & (table4["prediction"] == 0), 1, 0)
table4["FN"] = np.where((table4["truth"] == 1) & (table4["prediction"] == 0), 1, 0)

table4 = table4[["label", "TP", "FP", "TN", "FN"]]
table4 = table4.groupby(["label"]).sum()
table4.reset_index(inplace=True)

table4["accuracy"] = (table4["TP"] + table4["TN"]) / (
    table4["TP"] + table4["TN"] + table4["FP"] + table4["FN"]
)
table4["percision"] = (table4["TP"]) / (table4["TP"] + table4["FP"])
table4["recall"] = (table4["TP"]) / (table4["TP"] + table4["FN"])
table4["f1"] = 2 * (
    (table4["percision"] * table4["recall"]) / (table4["percision"] + table4["recall"])
)

run = wandb.init(project=project_name, entity=entity, job_type="inference")
wandb_table = wandb.Table(dataframe=table4)
wandb.log({"confusion_matrix": wandb_table})
wandb.finish()
