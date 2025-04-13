#!/bin/bash
set -x # 脚本调试模式

date=407
MODEL_NAME=Qwen2.5-0.5B-Instruct
EPOCHS=2                          

BS=256                            
# LR=1e-4                           
LR=1e-5                           
# LR=1e-6                           
# LR=3e-7                           

RUN_NAME="${date}_sft_gsm8k_${MODEL_NAME}_bs${BS}_lr${LR}_ep${EPOCHS}"
PROJECT_NAME="sft_gsm8k_${MODEL_NAME}"
LOG_FILE="${RUN_NAME}.log"

model_path=/mnt/teamdrive/projects/backup_chenlian/cache/$MODEL_NAME
MODEL_SAVE_DIR=/mnt/teamdrive/projects/backup_chenlian/rl/$PROJECT_NAME/$RUN_NAME
mkdir -p $MODEL_SAVE_DIR

export WANDB_API_KEY=346ac63bfef1ebae6bb5d71512417bcb81102dfd
export CUDA_VISIBLE_DEVICES=2
# export CUDA_VISIBLE_DEVICES=3

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=$LR \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    model.partial_pretrain=$model_path \
    trainer.default_local_dir=$MODEL_SAVE_DIR \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.total_epochs=$EPOCHS \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null \
    2>&1 | tee "log/${LOG_FILE}"