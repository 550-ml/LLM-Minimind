import math
from typing import Optional, Tuple
from sympy.ntheory import factor_
from transformers import PretrainedConfig
import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
# 继承PretrainedConfig从而传到huggingface
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
 

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.parameter(torch.ones(dim))


    def _norm(self, x):
        # [batch_size, n_token, dim] 只对x本身做大小的调整
        return  torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)  # [batch_size, n_token, 1]
        
    def forward(self, x):
        return x * self._norm(x.float()).type_as(x) * self.weight

def precompute_freqs_cis(dim: int, end:int(32*1024), rope_base, rope_scaling: Optional[dict]=None):
    # 初始化
    freqs, attn_factor = (1.0 / rope_base ** (torch.arange(0, dim, 2)[: dim//2].float()/ dim)), 1.0

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
    rope_scaling.get("original_max_position_embeddings", 2048),
    rope_scaling.get("factor", 16),
    rope_scaling.get("beta_fast", 32.0),
    rope_scaling.get("beta_slow", 1.0),
    rope_scaling.get("attention_factor", 1.0),
)
    if end/ orig_max >1.0:
        inv_dim = lambda b: (dim * torch.log(orig_max/ (2 * math.pi * b))/ (2 * math.log(rope_base)))

        # 计算高频，中频，低频
        low, high = (
            max(math.floor(inv_dim(beta_fast)), 0),
            max(math.ceil(inv_dim(beta_slow)), dim // 2-1)
        )
        
        # 计算y
        ramp = torch.clamp(
            (torch.arange(dim//2, device=freqs.device).float()-low)/max(high - low, 0.001),
            0,
            1
        )
        freqs = freqs * (1 - ramp + ramp / factor)
    
    # 计算位置索引
    t = torch.arange(end, device=freqs.device)

    # 外积
    freqs = torch.outer(t, freqs).float()  # [end] * [dim//2] -> [end, freqs]

    # j计算
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


# RopE代码
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        # x -> [B, H, N, D]
        dim = x.shape[-1]
        return torch.cat([-x[..., dim//2:], x[..., :dim//2]], dim=-1)
    # q, k: [B, H, N, D]
    # cos, sin: [N, D] 或 [1, 1, N, D] 其中 N 是序列长度, D 是特征维度
    q_embed = (q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复 key/value head 使其匹配 attention head 数量

    Args:
        x: 输入张量, shape 为 (B, N_token, num_key_value_heads, head_dim)
        n_rep: 每个 key/value head 需要重复的次数

    Returns:
        输出张量, shape 为 (B, N_token, num_key_value_heads * n_rep, head_dim)
    """
    B, N_token, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    
    x = x.unsqueeze(3)
    x = x.expand(B, N_token, num_key_value_heads, n_rep, head_dim)
    x = x.reshape(B, N_token, num_key_value_heads * n_rep, head_dim)
    return x

class Attention(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads

        assert args.num_attention_heads % args.num_key_value_heads == 0
        
        self.n_local_heads = args.num_attention_heads
        self.n_rep = self.n_local_heads // self.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        
        # [B, N_token, hidden_size]
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim,  args.hidden_size, bias=False)
        
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # 检查是否支持 Flash Attention (torch >= 2.0)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attention
    
    def forward(
        self, 
        x:torch.Tensor,
        position_embedding: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor]=None,
    ):
        # * 投影qk，v, [B, N, D]
        B, N_token, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        #* view拆分多个头
        xq = xq.view(B, N_token, self.n_local_heads, self.head_dim)  #
        xk = xk.view(B, N_token, self.num_key_value_heads, self.head_dim)  #
        xv = xv.view(B, N_token, self.num_key_value_heads, self.head_dim)  #
        
        #* q，k rope
        cos, sin = position_embedding
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        #* kvrepeat，注意kvchache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim = 1)
            xv = torch.cat([past_key_value[1], xv], dim = 1)
        past_kv = (xk, xv) if use_cache else None
        
        xq = xq.transpose(1,2)
        xk = repeat_kv(xk, self.n_rep).transpose(1,2)
        xv = repeat_kv(xv, self.n_rep).transpose(1,2)
        #* attention计算
        if (
            self.flash
            and (N_token > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask==1))
        ):
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            # 手写
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.full((N_token, N_token),-math.inf), diagonal=1)
            scores[:,:,:, -N_token:] += mask

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            
            scores = F.softmax(scores.float(), dim=-1)
            scores = self.attn_dropout(scores)
            output = scores @ xv
        
        output = output.transpose(1,2).reshape(B, N_token, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
        #* 拼接头，投影


    
class FeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias = False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias = False)
        self.act_fn = ACT2FN(config.hidden_act)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias = False)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) *self.up_proj(x)
        return self.dropout(self.down_proj(gated))

class MokioMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attention = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = (
            FeedForward(config)
            if not config.use_moe
            else MoEFeedForward(config)  # ！修正：原MoEFeedForaward拼写错误
        )

    def forward(
        self,
        hidden_states,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        res = hidden_states

        hidden_states, present_key_value = self.self_attention(
            self.input_layernorm(hidden_states),  # pre-norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )

        hidden_states = res + hidden_states

        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value
    

class MikioMindBlock(nn.Moudule):
    def __init__(
        self, 
        layer_id: int, 
        config: MokioMindConfig
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention = Attention(config)
        self.mlp = (
            FeedForward(config) if not config.use_moe else None
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
    
    def forward(
        self,
        hidden_states,
        position_embedding: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]= None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None   
    ):
        #* hidden_states [B, N, D]
        res = hidden_states
        #* RMSNorm
        hidden_states ,present_key_value = self.input_layernorm(hidden_states)
        self.self_attention(
            hidden_states,
            position_embedding,
            past_key_value,
            use_cache,
            attention_mask
        )

        hidden_states = res+hidden_states

        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value
