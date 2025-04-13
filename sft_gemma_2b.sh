# Tested with 2 & 4 GPUs

set -x

nproc_per_node=2
RUN_NAME="408_gsm8k-sft-gemma-2-2b-it"
PROJECT_NAME="408_gsm8k-sft-gemma-2-2b-it"
# save_path=/mnt/teamdrive/projects/backup_chenlian/sft/$PROJECT_NAME/$RUN_NAME
save_path=/mnt/teamdrive/projects/backup_chenlian/sft/$PROJECT_NAME
mkdir -p $save_path
LOG_FILE="${RUN_NAME}.log"

model_path=/mnt/teamdrive/projects/backup_chenlian/cache/gemma-2-2b-it

# Shift the arguments so $@ refers to the rest
shift 2

export WANDB_API_KEY=346ac63bfef1ebae6bb5d71512417bcb81102dfd

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=$model_path \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.total_epochs=2 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@ \
    2>&1 | tee "log/${LOG_FILE}"