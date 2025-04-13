# model.torch_dtype=bfloat16
# model.attn_implementation=eager
PROJECT_NAME="408_new_gsm8k-sft-gemma-2-2b-it"
save_path=/mnt/teamdrive/projects/backup_chenlian/sft/$PROJECT_NAME

torchrun --standalone --nnodes=1 --nproc_per_node=2 -m verl.trainer.fsdp_sft_trainer data.train_files=train.parquet data.val_files=test.parquet data.prompt_key=extra_info data.response_key=extra_info '+data.prompt_dict_keys=[question]' '+data.response_dict_keys=[answer]' data.micro_batch_size=8 model.partial_pretrain=google/gemma-2b-it trainer.default_local_dir=sft_gemma2_ckpt trainer.project_name=gsm8k-sft trainer.experiment_name=gsm8k-sft-gemma-2b-it trainer.total_epochs=2 'trainer.logger=[console,wandb]' trainer.default_hdfs_dir=null data.train_files=verl-data/gsm8k/train.parquet data.val_files=verl-data/gsm8k/test.parquet model.partial_pretrain=google/gemma-2-2b-it trainer.experiment_name=gsm8k-sft-gemma-2-2b-it trainer.default_local_dir=$save_path >log/408_gsm8k-sft-gemma-2-2b-it.log 2>&1