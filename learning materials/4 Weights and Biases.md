anytime you run wandb.init() it starts an experiment 

wandb.log() or using report_to="wandb" in transformers logs training metrics to wandb
wandb.log() has to be a dictionary with key and value pairs

experiements does not have to have the same training set for each experiments 
use can use artifacts to store what data is being for what experiments

wandb.login(host="https://ftisc.wandb.io/", key = "local-f89c2a822ee1ac33511ca756aaae3763cb7e45c0")
should store parameters in .env

entity = personal or team to log to
project_name = high level location to store experiments

wandb.finish() finishes the experiment 

after wandb.finish() you can then log a seperate run for evaluation
evaluations can be done within training or as a seperate 

In artifications you can use automations to trigger an airflow dag 
