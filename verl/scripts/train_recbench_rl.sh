#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CONFIG_FILE=${CONFIG_FILE:-}

if [[ "${1:-}" == "--env-file" || "${1:-}" == "--config" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "Usage: $0 [--env-file path] [hydra overrides...]" >&2
    exit 2
  fi
  CONFIG_FILE="$2"
  shift 2
fi

if [[ -n "$CONFIG_FILE" ]]; then
  if [[ -f "$CONFIG_FILE" ]]; then
    RESOLVED_CONFIG_FILE="$CONFIG_FILE"
  elif [[ -f "$SCRIPT_DIR/$CONFIG_FILE" ]]; then
    RESOLVED_CONFIG_FILE="$SCRIPT_DIR/$CONFIG_FILE"
  elif [[ -f "$SCRIPT_DIR/../$CONFIG_FILE" ]]; then
    RESOLVED_CONFIG_FILE="$SCRIPT_DIR/../$CONFIG_FILE"
  else
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 2
  fi
  set -a
  # shellcheck source=/dev/null
  source "$RESOLVED_CONFIG_FILE"
  set +a
fi

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-../data/recbench/processed/rl/movie_train.parquet}
VAL_FILE=${VAL_FILE:-../data/recbench/processed/rl/movie_dev.parquet}
PROJECT_NAME=${PROJECT_NAME:-faithrec}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-recbench-rl}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-8}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1536}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:--1}
REWARD_PATH=${REWARD_PATH:-verl/utils/reward_score/recbench_json.py}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-gdpo}
GDPO_REWARD_KEYS=${GDPO_REWARD_KEYS:-"['recommendation','faithfulness']"}
GDPO_REWARD_WEIGHTS=${GDPO_REWARD_WEIGHTS:-"[1.0,0.5]"}
REWARD_MODE=${REWARD_MODE:-faithrl_hybrid}
REWARD_K=${REWARD_K:-10}
BASELINE_CORRECT_RATE=${BASELINE_CORRECT_RATE:-0.5}
BASELINE_UNFAITHFUL_RATE=${BASELINE_UNFAITHFUL_RATE:-0.5}
HYBRID_WEIGHT=${HYBRID_WEIGHT:-0.1}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_DIR=${WANDB_DIR:-./wandb}

export WANDB_MODE
export WANDB_DIR
if [ -n "${WANDB_ENTITY:-}" ]; then
  export WANDB_ENTITY
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator="$ADV_ESTIMATOR" \
  algorithm.gdpo_reward_keys="$GDPO_REWARD_KEYS" \
  algorithm.gdpo_reward_weights="$GDPO_REWARD_WEIGHTS" \
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
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
  actor_rollout_ref.actor.kl_loss_type="$KL_LOSS_TYPE" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  reward.reward_manager.name=naive \
  reward.custom_reward_function.path="$REWARD_PATH" \
  reward.custom_reward_function.name=reward_func \
  reward.custom_reward_function.reward_kwargs.reward_mode="$REWARD_MODE" \
  reward.custom_reward_function.reward_kwargs.k="$REWARD_K" \
  reward.custom_reward_function.reward_kwargs.baseline_correct_rate="$BASELINE_CORRECT_RATE" \
  reward.custom_reward_function.reward_kwargs.baseline_unfaithful_rate="$BASELINE_UNFAITHFUL_RATE" \
  reward.custom_reward_function.reward_kwargs.hybrid_weight="$HYBRID_WEIGHT" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger="$TRAINER_LOGGER" \
  trainer.log_val_generations="$LOG_VAL_GENERATIONS" \
  trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
  trainer.nnodes="$NNODES" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  "$@"
