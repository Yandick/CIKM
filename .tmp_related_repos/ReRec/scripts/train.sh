set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH="models/Qwen2.5-3B-Instruct"
PROJECT_NAME="ReRecec"
EXPERIMENT_NAME="basemodel-setting-domain"

# Set your W&B credentials before running:
# export WANDB_API_KEY='<your_wandb_api_key>'
# export WANDB_ENTITY='<your_wandb_entity>'


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=raee \
    data.train_files=data/book_train.parquet \
    data.val_files=data/book_test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=768 \
    data.max_response_length=768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.loss_agg_mode='token-mean' \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=raee \
    reward_model.raee.min_token_reward_ratio=0.7 \
    reward_model.raee.max_token_reward_ratio=1.0 \
    reward_model.raee.mode=rule \
    custom_reward_function.path=verl/utils/reward_score/rec.py \
    custom_reward_function.name=reward_func \
    custom_reward_function.dual_graph_enhanced=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.val_metric_key="val/ndcg@k" \
    trainer.val_metric_mode="max" \
    trainer.rollout_data_dir=logs/$EXPERIMENT_NAME \
    trainer.validation_data_dir=logs/$EXPERIMENT_NAME \
    trainer.resume_mode=auto \
    trainer.resume_from_path="" \
    trainer.curriculum_learning.enable=True \
    trainer.curriculum_learning.difficulty_metric=ndcg@k \
    trainer.curriculum_learning.difficulty_threshold=0.1 \
    trainer.total_epochs=50 $@