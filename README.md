# 数据下载
```
https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
```
modelscope download --dataset gongjy/minimind_dataset pretrain_t2t_mini.jsonl --local_dir  /root/autodl-tmp
modelscope download --dataset gongjy/minimind_dataset sft_2048.jsonl --local_dir ./dataset
```
```

# 模型搭建
我想综合两者，学透LLM的架构
同时追求训练效率、研究迭代速度和工程可跑性