# Finetuning-Paraphrase-Detection

This repository contains code for finetuning a model mainly for paraphrase detection tasks. It uses Docker for environment setup and training.

## Setup Instructions

### Prerequisites
* Make sure you have Docker installed on your machine.
* if you use Docker-compose make sure to create a `.env` file in the root directory.

###  1. Clone the repository

```bash
git clone git@github.com:lucagaf/Finetuning-Paraphrase-Detection.git
cd Finetuning-Paraphrase-Detection
```


### Variant 1: Build the Docker image and run the container
```bash
docker build -t training .
```

```bash
docker run -e WANDB_API_KEY="<YOUR-WANDB_API_KEY>" -e WANDB_RUN_ENVIRONEMENT="<LOCATION where you start training>" training
```

If you want to specify different hyperparameters you can adjust them with specifying the environment variables in the `docker run` command.

```bash
docker run -e WANDB_API_KEY="<YOUR-WANDB_API_KEY>" -e WANDB_RUN_ENVIRONEMENT="<LOCATION where you start training>" -e VARIABLE="VALUE" training
```
Those are the availabe hyperparameters to adjust: 
* WANDB_PROJECTNAME="MLops-Project2"
*  MODEL_NAME="distilbert-base-uncased"
*  TASK_NAME="mrpc"
*  SEED=42
*  LR=2e-5
*  WARMUP_STEPS=300
*  BATCH_SIZE=16
*  BETA1=0.85






### Variant 2: Use Docker-compose
First of all you have to create a `.env` file in the root directory. The `.env` file should look like this:

```
WANDB_API_KEY=<YOUR-WANDB_API_KEY>
WANDB_RUN_ENVIRONEMENT=<LOCATION where you start training>
```

To build the Docker image, run the following command:


```bash 
docker-compose build
```


```bash
 docker-compose run training
 ```


