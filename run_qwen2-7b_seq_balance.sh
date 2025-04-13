set -x

MODEL_NAME="Qwen2-7B-Instruct"
# MODEL_NAME="Qwen2.5-7B"

BS=1024                        # batch size
# MINI_BS=256                    # mini batch size
MINI_BS=512                    # mini batch size翻倍
MINI_BS=256                    # mini batch size
ACTOR_LR=1e-6                 # actor learning rate
KL_COEF=0.001                 # KL loss coefficient
N_ROLLOUT=5                   # number of rollouts
EPOCHS=10                      # total epochs

# GPU_PER_NODE=8               # GPUs per node
GPU_PER_NODE=4               # GPUs per node
GPU_UTIL=0.6                 # GPU memory utilization
TP_SIZE=2                    # tensor parallel size

save_freq=7 # 1 epoch, 7k training samples
test_freq=7

# DATE=409
DATE=410
RUN_NAME="${DATE}_${MODEL_NAME}_bs${BS}_mbs${MINI_BS}_alr${ACTOR_LR}_kl${KL_COEF}_roll${N_ROLLOUT}_ep${EPOCHS}_tp${TP_SIZE}"
PROJECT_NAME="gsm8k_rl_${MODEL_NAME}"
LOG_FILE="${RUN_NAME}.log"

model_path=/mnt/teamdrive/projects/backup_chenlian/cache/$MODEL_NAME
MODEL_SAVE_DIR=/mnt/teamdrive/projects/backup_chenlian/rl/$PROJECT_NAME/$RUN_NAME
mkdir -p $MODEL_SAVE_DIR

export WANDB_API_KEY=346ac63bfef1ebae6bb5d71512417bcb81102dfd
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=$BS \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BS \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL \
    actor_rollout_ref.rollout.n=$N_ROLLOUT \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$GPU_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$EPOCHS $@ \
    2>&1 | tee "log/${LOG_FILE}"