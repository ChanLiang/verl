set -x

nproc_per_node=2
# RUN_NAME="new_gsm8k-sft-rl-gemma-2-2b-it"
# PROJECT_NAME="gsm8k-sft-rl-gemma-2b-it"
RUN_NAME="gsm8k-rl-gemma-2-2b-it"
PROJECT_NAME="gsm8k-rl-gemma-2-2b-it"

save_path=/mnt/teamdrive/projects/backup_chenlian/rl/$PROJECT_NAME/$RUN_NAME
LOG_FILE="${RUN_NAME}.log"

# model_path=/mnt/teamdrive/projects/backup_chenlian/sft/gsm8k-sft-gemma-2b-it/gsm8k-sft-gemma-2b-it/global_step_58
# model_path=/mnt/teamdrive/projects/backup_chenlian/sft/408_new_gsm8k-sft-gemma-2-2b-it/global_step_58
model_path=/mnt/teamdrive/projects/backup_chenlian/cache/gemma-2b-it

# sleep 40m

MODEL_SAVE_DIR=/mnt/teamdrive/projects/backup_chenlian/rl/$PROJECT_NAME/$RUN_NAME
mkdir -p $MODEL_SAVE_DIR

export WANDB_API_KEY=346ac63bfef1ebae6bb5d71512417bcb81102dfd

# export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=2,3

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.model.path=$model_path \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=True \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=$MODEL_SAVE_DIR \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@ \
    2>&1 | tee "log/${LOG_FILE}"
