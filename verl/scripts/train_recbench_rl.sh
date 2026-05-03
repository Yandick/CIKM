#!/usr/bin/env bash
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-../data/recbench/processed/rl/movie_train.parquet}
VAL_FILE=${VAL_FILE:-../data/recbench/processed/rl/movie_test.parquet}
PROJECT_NAME=${PROJECT_NAME:-faithrec}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-recbench-rl}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
ROLLOUT_N=${ROLLOUT_N:-5}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}
ACTOR_LR=${ACTOR_LR:-5e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}
REWARD_PATH=${REWARD_PATH:-verl/utils/reward_score/recbench_json.py}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  reward.reward_manager.name=naive \
  reward.custom_reward_function.path="$REWARD_PATH" \
  reward.custom_reward_function.name=reward_func \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
  trainer.nnodes="$NNODES" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  "$@"
