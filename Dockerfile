# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and source code
COPY requirements.txt .
COPY src/ src/
RUN pip install --no-cache-dir -r requirements.txt

ENV WANDB_KEY=""
ENV WANDB_PROJECTNAME="MLops-Project2"
ENV MODEL_NAME="distilbert-base-uncased"
ENV TASK_NAME="mrpc"
ENV SEED=42
ENV LR=2e-5
ENV WARMUP_STEPS=300
ENV BATCH_SIZE=16
ENV BETA1=0.85

CMD python src/main.py \
    -wandb_key $WANDB_KEY \
    -wandb_projectname $WANDB_PROJECTNAME \
    -model_name $MODEL_NAME \
    -task_name $TASK_NAME \
    -seed $SEED \
    -lr $LR \
    -warmup_steps $WARMUP_STEPS \
    -batch_size $BATCH_SIZE \
    -beta1 $BETA1
