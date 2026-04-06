#!/usr/bin/env bash

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp
export HF_HUB_CACHE=/root/autodl-tmp/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hub

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 trainer/train_ppo.py \
  --use_wandb \
  --data_path /root/autodl-tmp/rlaif.jsonl \
  --reward_model_path /root/autodl-tmp/hub/models--internlm--internlm2-1_8b-reward/snapshots/25f3593492ab4625ce00fce8c5e67802d6e702ca \
  --debug_mode \
  --rollout_engine torch && /usr/bin/shutdown  
