# python trainer/trainer_full_sft.py \
#   --use_wandb
CUDA_VISIBLE_DEVICES=0,2,3,4,5 torchrun --nproc_per_node=5 trainer/trainer_full_sft.py --use_wandb \
  --batch_size 40 \
  
