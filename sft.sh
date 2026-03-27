#!/usr/bin/env bash
# SFT 默认已基于预训练：脚本里 --from_weight 默认是 pretrain，
# 会加载 <项目根>/out/pretrain_{hidden_size}.pth（MoE 时为 pretrain_{hidden_size}_moe.pth）。
# 请先跑完预训练并保证上述文件存在，或与 --hidden_size / --use_moe 一致。
# 若预训练保存的前缀不是 pretrain，显式传: --from_weight <前缀>
# 权重目录为项目根下 out/；若在别处请先复制或软链到 out/

CUDA_VISIBLE_DEVICES=0,2,3,4,5 torchrun --nproc_per_node=5 trainer/trainer_full_sft.py \
  --use_wandb \
  --save_weight full_sft_minibig_512_1024 \
  --from_weight full_sft_minibig_512 \
  --data_path ./dataset/sft_1024.jsonl \
  --batch_size 14 \
  --epochs 1 \
  --learning_rate 5e-6 \
  --max_seq_len 1024