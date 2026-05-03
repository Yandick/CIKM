#!/usr/bin/env bash
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-../data/recbench/processed/sft/movie_train.parquet}
VAL_FILE=${VAL_FILE:-../data/recbench/processed/sft/movie_test.parquet}
PROJECT_NAME=${PROJECT_NAME:-faithrec}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-recbench-sft}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_LENGTH=${MAX_LENGTH:-2048}

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" -m verl.trainer.sft_trainer \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.messages_key=messages \
  data.max_length="$MAX_LENGTH" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
  model.path="$MODEL_PATH" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.total_epochs=1 \
  trainer.n_gpus_per_node="$NPROC_PER_NODE" \
  trainer.logger='["console"]' \
  "$@"
