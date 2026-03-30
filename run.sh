# pre_train
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 trainer/trainer_pretrain.py --use_wandb \
  --batch_size 16 \
  --epochs 2 \
  --num_hidden_layers 10 \
  ----total_batch_size_tokens 262144 \
  --data_path /root/autodl-tmp/pretrain_t2t_mini.jsonl
# python trainer/trainer_pretrain.py --use_wandb \
#   --batch_size  \
#   --epochs 6 \
#   --use_moe 1 \
#   --from_weight none \
#   --from_resume 0
# eval pre_train
# python eval.py --load_from ./out/pretrain_512.pth