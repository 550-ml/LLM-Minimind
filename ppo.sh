#!/usr/bin/env bash
# 自定义 PPO 起点：把 ckpt 放到 --save_dir 下，命名为 {前缀}_{hidden_size}.pth，例如 out/my_sft_512.pth，则加：
#   --ppo_init_weight my_sft
#
# Reward 模型默认不联网（local_files_only）：请使用本机已下载目录，例如：
#   --reward_model_path /path/to/internlm2-1_8b-reward
# 或在有网的机器上先 huggingface-cli download ...，再在本机用缓存路径。
# 若必须在线拉取：加上 --reward_allow_hf_download
# export HF_HUB_OFFLINE=1   # 可选，与默认行为一致，彻底禁止 hub 访问

HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0,2,3,5 torchrun --nproc_per_node=4 trainer/trainer_ppo.py \
  --use_wandb \
  --save_dir out \
  --save_weight full_sft_minibig_512_1024 \
  --epochs 2 \
  --batch_size 2 \
  --learning_rate 8e-8 \
  --critic_learning_rate 8e-8 \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --reasoning 0 \
  --data_path dataset/rlaif-mini.jsonl \
  --reward_model_path internlm/internlm2-1_8b-reward
