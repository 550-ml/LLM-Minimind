import os
import sys

__package__ = "trainer"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import argparse
import time
import warnings
import math
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, random_split
from model.NanoMind import NanoMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def unwrap_model(model):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw_model, '_orig_mod', raw_model)


def register_hidden_norm_hooks(model):
    hidden_norms = {}
    raw_model = unwrap_model(model)

    def make_hook(layer_idx):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple) and len(output) > 1 and torch.is_tensor(output[1]):
                hidden_norms[f"hidden_norm/layer_{layer_idx}"] = (
                    output[1].detach().float().norm(dim=-1).mean()
                )
        return hook

    handles = [
        layer.register_forward_hook(make_hook(layer_idx))
        for layer_idx, layer in enumerate(raw_model.model.layers)
    ]
    return hidden_norms, handles


def get_layer_grad_norms(model):
    raw_model = unwrap_model(model)
    grad_norms = {}
    for layer_idx, layer in enumerate(raw_model.model.layers):
        squared_norm = None
        for param in layer.parameters():
            if param.grad is None:
                continue
            param_norm = param.grad.detach().float().norm(2)
            squared = param_norm * param_norm
            squared_norm = squared if squared_norm is None else squared_norm + squared
        if squared_norm is not None:
            grad_norms[f"grad_norm/layer_{layer_idx}"] = torch.sqrt(squared_norm).item()
    return grad_norms


@torch.no_grad()
def evaluate(loader, max_batches=0):
    if loader is None:
        return None

    model.eval()
    total_loss = torch.tensor(0.0, device=args.device)
    total_tokens = torch.tensor(0.0, device=args.device)

    for batch_idx, (input_ids, labels) in enumerate(loader, start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break

        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        with autocast_ctx:
            res = model(input_ids, labels=labels)

        valid_tokens = (labels[..., 1:] != -100).sum()
        if res.loss is not None and valid_tokens.item() > 0:
            total_loss += res.loss.detach().float() * valid_tokens
            total_tokens += valid_tokens

        del input_ids, labels, res

    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    avg_loss = total_loss / total_tokens.clamp(min=1)
    loss_value = avg_loss.item()
    ppl_value = math.exp(loss_value) if loss_value < 50 else float("inf")
    model.train()
    return {"valid_loss": loss_value, "valid_ppl": ppl_value}


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    global last_grad_metrics
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        total_grad_norm = None
        layer_grad_norms = {}
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            layer_grad_norms = get_layer_grad_norms(model)
            last_grad_metrics = {"grad_norm/total": float(total_grad_norm), **layer_grad_norms}

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss =  0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            log_data = {
                "loss": current_loss,
                "logits_loss": current_logits_loss,
                "aux_loss": current_aux_loss,
                "learning_rate": current_lr,
                "epoch_time": eta_min,
            }
            log_data.update(last_grad_metrics)
            log_data.update({key: value.item() for key, value in hidden_norms.items()})
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log(log_data)

        if args.eval_interval > 0 and val_loader is not None and (step % args.eval_interval == 0 or step == iters):
            metrics = evaluate(val_loader, args.eval_batches)
            if metrics is not None and is_main_process():
                Logger(f'Validation: loss: {metrics["valid_loss"]:.4f}, ppl: {metrics["valid_ppl"]:.2f}')
                if wandb: wandb.log(metrics)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir=os.path.join(PROJECT_ROOT, "checkpoints"))
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NanoMind Pretraining")
    parser.add_argument("--save_dir", type=str, default=os.path.join(PROJECT_ROOT, "out"), help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=5, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default=os.path.join(PROJECT_ROOT, "dataset", "pretrain_hq.jsonl"), help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="NanoMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--model_variant", type=str, default="full", choices=["full", "baseline"], help="模型变体：full=NanoMind BAR，baseline=无BAR")
    parser.add_argument("--val_ratio", type=float, default=0.02, help="从训练集固定切出的验证集比例，0表示不验证")
    parser.add_argument("--eval_interval", type=int, default=1000, help="验证间隔step，0表示不验证")
    parser.add_argument("--eval_batches", type=int, default=0, help="每次验证最多跑多少个batch，0表示跑完整验证集")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = NanoMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=os.path.join(PROJECT_ROOT, "checkpoints")) if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"NanoMind-{args.model_variant}-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device, model_variant=args.model_variant)
    full_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    val_loader = None
    if args.val_ratio > 0:
        val_size = max(1, int(len(full_ds) * args.val_ratio))
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        val_sampler = DistributedSampler(val_ds, shuffle=False) if dist.is_initialized() else None
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        Logger(f'Dataset Split: train={train_size}, valid={val_size}, val_ratio={args.val_ratio}')
    else:
        train_ds = full_ds
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    last_grad_metrics = {}
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    hidden_norms, hidden_hook_handles = register_hidden_norm_hooks(model)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
