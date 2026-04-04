# pre_train
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 trainer/train_pretrain.py --use_wandb \
  --batch_size 130 \
  --epochs 3 \
  --num_hidden_layers 8 \
  --hidden_size 768 \
  --data_path /root/autodl-tmp/pretrain_t2t.jsonl \
  --log_interval 50 \
  --from_weight pretrain \
  --use_compile 1 && /usr/bin/shutdown  
# CUDA_VISIBLE_DEVICES=0 python trainer/trainer_pretrain2.py --use_wandb \
#   --batch_size 16 \
#   --epochs 2 \
#   --num_hidden_layers 8 \
#   --hidden_size 768 \
#   --save_dir ./test \
#   --data_path /root/autodl-tmp/pretrain_t2t_mini.jsonl \
# python trainer/trainer_pretrain.py --use_wandb \
#   --batch_size  \
#   --epochs 6 \
#   --use_moe 1 \
#   --from_weight none \
#   --from_resume 0
# eval pre_train
# python eval.py --load_from ./out/pretrain_512.pth