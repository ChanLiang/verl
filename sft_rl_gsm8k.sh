#!/bin/bash
# date=407
date=408

# MODEL_NAME=sfted2epoch_Qwen2.5-0.5B-Instruct
# MODEL_NAME=sfted1epoch_Qwen2.5-0.5B-Instruct

# MODEL_NAME=sfted2epoch1e-6_Qwen2.5-0.5B-Instruct
# MODEL_NAME=sfted2epoch3e-7_Qwen2.5-0.5B-Instruct
MODEL_NAME=sfted2epoch1e-5_Qwen2.5-0.5B-Instruct

EPOCHS=5                          # total epochs
STEPS=120                         # total training steps
BS=256                            # batch size
PROMPT_LEN=512                    # max prompt length
RESP_LEN=256                      # max response length
ACTOR_LR=1e-6                     # actor learning rate
CRITIC_LR=1e-5                    # critic learning rate
KL_COEF=0.001                     # KL coefficient
PPO_MINI_BS=64                    # PPO mini batch size
PPO_MICRO_BS=4                    # PPO micro batch size per GPU
SEED=42                           # random seed

save_freq=30 # 1 epoch, 7k training samples
test_freq=10

RUN_NAME="${date}_gsm8k_${MODEL_NAME}_bs${BS}_alr${ACTOR_LR}_clr${CRITIC_LR}_kl${KL_COEF}_ep${EPOCHS}_step${STEPS}"
PROJECT_NAME="gsm8k_${MODEL_NAME}"
LOG_FILE="${RUN_NAME}.log"

# model_path=/mnt/teamdrive/projects/backup_chenlian/cache/$MODEL_NAME
# model_path=/mnt/teamdrive/projects/backup_chenlian/rl/sft_gsm8k_Qwen2.5-0.5B-Instruct/407_sft_gsm8k_Qwen2.5-0.5B-Instruct_bs256_lr1e-4_ep2/global_step_58
# model_path=/mnt/teamdrive/projects/backup_chenlian/rl/sft_gsm8k_Qwen2.5-0.5B-Instruct/407_sft_gsm8k_Qwen2.5-0.5B-Instruct_bs256_lr1e-4_ep2/global_step_29

# model_path=/mnt/teamdrive/projects/backup_chenlian/rl/sft_gsm8k_Qwen2.5-0.5B-Instruct/407_sft_gsm8k_Qwen2.5-0.5B-Instruct_bs256_lr1e-6_ep2/global_step_58
# model_path=/mnt/teamdrive/projects/backup_chenlian/rl/sft_gsm8k_Qwen2.5-0.5B-Instruct/407_sft_gsm8k_Qwen2.5-0.5B-Instruct_bs256_lr3e-7_ep2/global_step_58
model_path=/mnt/teamdrive/projects/backup_chenlian/rl/sft_gsm8k_Qwen2.5-0.5B-Instruct/407_sft_gsm8k_Qwen2.5-0.5B-Instruct_bs256_lr1e-5_ep2/global_step_58

MODEL_SAVE_DIR=/mnt/teamdrive/projects/backup_chenlian/rl/$PROJECT_NAME/$RUN_NAME
mkdir -p $MODEL_SAVE_DIR

export WANDB_API_KEY=346ac63bfef1ebae6bb5d71512417bcb81102dfd
export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=1

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=data/gsm8k/train.parquet \
 data.val_files=data/gsm8k/test.parquet \
 data.train_batch_size=$BS \
 data.max_prompt_length=$PROMPT_LEN \
 data.max_response_length=$RESP_LEN \
 actor_rollout_ref.model.path=$model_path \
 actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
 actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BS \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BS \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 +actor_rollout_ref.rollout.seed=$SEED \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=$CRITIC_LR \
 critic.model.path=$model_path \
 critic.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BS \
 algorithm.kl_ctrl.kl_coef=$KL_COEF \
 trainer.logger=['console','wandb'] \
 trainer.project_name=$PROJECT_NAME \
 trainer.experiment_name=$RUN_NAME \
 trainer.val_before_train=True \
 trainer.default_local_dir=$MODEL_SAVE_DIR \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=$save_freq \
 trainer.test_freq=$test_freq \
 trainer.total_epochs=$EPOCHS \
 trainer.total_training_steps=$STEPS \
 2>&1 | tee "log/${LOG_FILE}"