# pre_train ablation: NanoMind full
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 trainer/train_pretrain.py --use_wandb \
  --wandb_project NanoMind-Pretrain-Ablation \
  --model_variant full \
  --save_weight pretrain_full \
  --batch_size 130 \
  --epochs 6 \
  --num_hidden_layers 8 \
  --hidden_size 768 \
  --data_path /root/autodl-tmp/pretrain_t2t.jsonl \
  --eval_interval 1000 \
  --val_ratio 0.02 \
  --from_weight none \
  --use_compile 1

# pre_train ablation: Baseline without block attention residual
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 trainer/train_pretrain.py --use_wandb \
  --wandb_project NanoMind-Pretrain-Ablation \
  --model_variant baseline \
  --save_weight pretrain_baseline \
  --batch_size 130 \
  --epochs 6 \
  --num_hidden_layers 8 \
  --hidden_size 768 \
  --data_path /root/autodl-tmp/pretrain_t2t.jsonl \
  --eval_interval 1000 \
  --val_ratio 0.02 \
  --from_weight none \
  --use_compile 1
