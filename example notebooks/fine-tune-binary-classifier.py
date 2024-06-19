import pandas as pd
import numpy as np
import os
import datasets
import evaluate
import wandb
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from scipy.special import softmax

labelled_data = pd.read_parquet(
    "s3://fti-adhoc-prod/MLGuide/Example_Classification_Training_Data"
)
labelled_data = labelled_data[["text", "relevance_label"]]
labelled_data = labelled_data.rename(columns={"relevance_label": "label"})

# let's random sample our discard so we have equal number of samples
num_relevant = len(labelled_data[labelled_data["label"] == "Relevant"])
keep = (
    labelled_data[labelled_data["label"] == "Discard"]
    .sample(num_relevant)
    .index.tolist()
)
keep.extend(labelled_data[labelled_data["label"] == "Relevant"].index.tolist())
labelled_data = labelled_data.iloc[keep]

# create train (75%), test(10%), and validation(15%) sets
train, test = train_test_split(
    labelled_data,
    test_size=0.25,
    random_state=42,
    shuffle=True,
    stratify=labelled_data["label"],
)

test, val = train_test_split(
    test, test_size=0.6, random_state=42, shuffle=True, stratify=test["label"]
)

# covert to datasets and encode labels
ds = datasets.DatasetDict(
    {
        "train": datasets.Dataset.from_pandas(train.reset_index(drop=True)),
        "test": datasets.Dataset.from_pandas(test.reset_index(drop=True)),
        "val": datasets.Dataset.from_pandas(val.reset_index(drop=True)),
    }
)

ds = ds.class_encode_column("label")

# store label information
id2label = dict(enumerate(ds["train"].features["label"].names))
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# set model checkpoint
model_ckpt = "microsoft/deberta-v3-small"

# tokenize data
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=512)


ds_encode = ds.map(tokenize, batched=True)

# load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=num_labels
)

# load in evaluation metrics
accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # metrics from the datasets library have a `compute` method
    return accuracy_metric.compute(predictions=predictions, references=labels)


# set up wandb
wandb.login()

project_name = "llm-finetuning-example"  # @param
entity = "marycatherine-sullivan"  # @param
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

wandb.init(project=project_name, entity=entity, job_type="training")

training_args = TrainingArguments(
    report_to="wandb",
    output_dir="outputs",
    overwrite_output_dir=True,
    eval_strategy="epoch",
    learning_rate=5e-5,
    logging_steps=100,
    save_steps=1000,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=ds_encode["train"],
    eval_dataset=ds_encode["val"],
    compute_metrics=compute_metrics,
)

trainer.evaluate()  # verify compute metrics works
trainer.train()
wandb.finish()

# log run in model registry
last_run_id = "1bxxrc2q"  # @param
wandb.init(project=project_name, entity=entity, job_type="registering_best_model")
best_model = wandb.use_artifact(f"{entity}/{project_name}/model-{last_run_id}:latest")
registered_model_name = "deberta-relevance"  # @param {type: "string"}
wandb.run.link_artifact(
    best_model, f"{entity}/model-registry/{registered_model_name}", aliases=["staging"]
)
wandb.finish()

# download "final" model
with wandb.init(project=project_name, entity=entity) as run:
    # Pass the name and version of Artifact
    my_model_artifact = run.use_artifact(best_model)

    # Download model weights to a folder and return the path
    model_dir = my_model_artifact.download()

    # Load your Hugging Face model from that folder
    #  using the same model class
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

# wandb.finish()

# write prediction results to table
from scipy.special import softmax

run = wandb.init(project=project_name, entity=entity, job_type="inference")
wandb_table = wandb.Table(
    columns=["text", "label", "prediction", "discard", "relevant"]
)

for i in range(len(ds["test"])):
    sequence = ds["test"]["text"][i]

    encoded_input = tokenizer(sequence, return_tensors="pt")
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores).tolist()

    label = id2label.get(ds["test"]["label"][i])
    prediction = id2label.get(output.logits.argmax().item())
    discard = scores[0]
    relevant = scores[1]
    wandb_table.add_data(sequence, label, prediction, discard, relevant)

print(label)
wandb.log({"inference_table": wandb_table})
wandb.finish()
