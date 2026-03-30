#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# 与 eval.py 一致：load_from / save_dir 为相对项目根路径；预训练权重用 --weight pretrain 时会走 bos+原文，不用 chat_template
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --load_from model \
  --save_dir out \
  --weight pretrain \
  --lora_weight None \
  --hidden_size 512 \
  --num_hidden_layers 12 \
  --use_moe 0 \
  --max_new_tokens 512 \
  --temperature 0.85 \
  --top_p 0.95 \
  --open_thinking 0 \
  --historys 0 \
  --show_speed 1 \
  --device cuda

# 需要更长外推可取消下行注释：
# CUDA_VISIBLE_DEVICES=0 python eval.py ... --inference_rope_scaling
