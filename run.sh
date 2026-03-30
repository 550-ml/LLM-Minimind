# pre_train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 trainer/trainer_pretrain2.py --use_wandb \
  --batch_size 70 \
  --epochs 2 \
  --from_resume 1 \
  --num_hidden_layers 12 \
  --hidden_size 512 \
  --data_path /root/autodl-tmp/pretrain_t2t.jsonl \
  --use_compile 1 && /usr/bin/shutdown  
# python trainer/trainer_pretrain.py --use_wandb \
#   --batch_size  \
#   --epochs 6 \
#   --use_moe 1 \
#   --from_weight none \
#   --from_resume 0
# eval pre_train
# python eval.py --load_from ./out/pretrain_512.pth