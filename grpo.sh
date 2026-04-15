#!/usr/bin/env bash
set -euo pipefail

# 统一从仓库根目录启动，避免相对路径错乱
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp
export HF_HUB_CACHE=/root/autodl-tmp/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hub

# 参照 ppo.sh 的风格：把关键路径显式写出来
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 trainer/train_grpo.py \
  --use_wandb \
  --data_path /root/autodl-tmp/rlaif.jsonl \
  --reward_model_path /root/autodl-tmp/hub/models--internlm--internlm2-1_8b-reward/snapshots/25f3593492ab4625ce00fce8c5e67802d6e702ca \
  --debug_mode \
  --rollout_engine torch

# 如果你需要像 ppo.sh 一样训练完自动关机，取消下一行注释即可
# /usr/bin/shutdown

