import math
from typing import List, Optional, Tuple, Union

from numpy import repeat
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast

class NanoMindConfig(PretrainedConfig):
    model_type = "nanomind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### block attention residual
        self.attn_res_block_size = 2 # 2 is the default block size for the attention residual
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # x:[B,S,D]
        norm = torch.rsqrt((x**2).mean(-1, keepdim=True) + self.eps)
        return x * norm.type_as(x) * self.weight

class FeedForward(nn.Module):
    def __init__(self, config: NanoMindConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        return self.dropout(self.down_proj(gate * self.up_proj(x)))

def norm(x):
    return F.rms_norm(x, (x.size(-1) ,))

def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    # x [B, S, N_kv_head, dim]
    B, S, n_kv_heads, head_dim = x.shape
    x = x[:,:,:,None,:].expand(B, S, n_kv_heads, n_rep, head_dim).reshape(B, S, n_kv_heads*n_rep, head_dim)
    return x

class GroupQueryAttention(nn.Module):
    def __init__(self, config: NanoMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.n_rep = self.num_attention_heads // self.num_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        assert self.num_attention_heads % self.num_kv_heads ==0
        assert config.hidden_size % config.num_attention_heads == 0

        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias = False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and config.flash_attn
        )
    def forward(
        self,
        x: torch.Tensor, # [B, Seq_len, D]
        position_embedding: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
        use_cache=False,
        attention_mask:Optional[torch.Tensor] = None
    ):
        B, S, D = x.shape
        # * 转换qkv
        q = self.q_proj(x).view(B, S, self.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # * RoPE
        cos, sin = position_embedding
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # * kv_cache
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)  # 
            v = torch.cat([kv_cache[1], v], dim=1)  # 
        past_kv = (k, v) if use_cache else None

        q, k, v = (
            q.transpose(1, 2),
            repeat_kv(k, self.n_rep).transpose(1,2),
            repeat_kv(v, self.n_rep).transpose(1,2)
        )

        # *做SDPA
        Lq = q.size(2)
        Lk = k.size(2)
        if (
            self.flash
            and Lq > 1
            and kv_cache is None
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        elif (
            self.flash
            and Lq == 1
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if Lq > 1:
                causal_mask = torch.triu(
                    torch.full((Lq, Lq), float("-inf"), device=scores.device),
                    diagonal=1,
                )
                scores[:, :, :, -Lq:] += causal_mask
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(q)) @ v
        output = output.transpose(1, 2).reshape(B, S, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
    
class BlcokAttentionResidual(nn.Module):
    def __init__(self, config: NanoMindConfig):
        super().__init__()
        self.norm = RMSNorm(dim=config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, 1, bias=False)
    
    def forward(
        self,
        blocks: List[torch.Tensor],  #
        partial_block: torch.Tensor  # [B, S, D]
    ):
        values = torch.stack(blocks + [partial_block], dim=0)
        key = self.norm(values)  # [L, B, S, D]
        logits = self.proj(key).squeeze(-1)  #[L, B, S]
        alpha = torch.softmax(logits, dim=0)  #[L,B, S]每个token
        hidden_states = (alpha.unsqueeze(-1) * values).sum(dim=0)
        return hidden_states

def precompute_freqs(
    dim: int, 
    end:int = int(32*1024),
    rope_base: float=1e6,
    rope_scaling: Optional[dict] = None,
):
    assert dim % 2 ==0
    freqs = 1.0 / rope_base**(torch.arange(0, dim, 2).float() / dim)  #[dim/2]
    attn_factor = 1.0
    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)
    position = torch.arange(end, device=freqs.device)
    freqs = torch.outer(position, freqs).float()
    embs = torch.cat([freqs, freqs], dim=-1)
    cos  = torch.cos(embs) * attn_factor
    sin = torch.sin(embs) * attn_factor
    return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        # [B, S, H, D]
        B, S, H, D = x.shape
        x1 = x[:,:,:,:D//2]
        x2 = x[:,:,:,D//2:]
        return torch.cat([-x2, x1], dim=-1)
    q_emb = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
    k_emb = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
    return q_emb, k_emb


class NanoMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: NanoMindConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention = GroupQueryAttention(config)
        self.mlp = FeedForward(config)
        
        self.layer_id =layer_id
        self.block_size = config.attn_res_block_size
        self.attn_res = BlcokAttentionResidual(config)
        self.mlp_res = BlcokAttentionResidual(config)
    
    def forward(
        self,
        hidden_states,
        blocks,
        partial_block,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if partial_block is None:
            partial_block = hidden_states

        # attn
        h = self.attn_res(blocks, partial_block)
        if (self.layer_id +1)% (self.block_size // 2) == 0:
            blocks = blocks + [partial_block]
            partial_block = None
        attn_out, present_kv = self.self_attention(
            self.input_layernorm(h),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        # TODO 实现注意力残差
        partial_block = partial_block + attn_out if partial_block is not None else attn_out
        
        # mlp
        h = self.mlp_res(blocks, partial_block)
        mlp_out = self.mlp(
            self.post_attention_layernorm(h)
        )
        partial_block = partial_block + mlp_out
        return blocks, partial_block , present_kv

class NanoMindModel(nn.Module):
    def __init__(self, config:NanoMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [NanoMindBlock(i, config) for i in range(config.num_hidden_layers)]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs(
            dim = config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        hidden_size = self.config.hidden_size
        std = hidden_size**-0.5
        bound = math.sqrt(3.0) * std

        torch.nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)

        for layer in self.layers:
            torch.nn.init.ones_(layer.input_layernorm.weight)
            torch.nn.init.ones_(layer.post_attention_layernorm.weight)

            torch.nn.init.uniform_(layer.self_attention.q_proj.weight, -bound, bound)
            torch.nn.init.uniform_(layer.self_attention.k_proj.weight, -bound, bound)
            torch.nn.init.uniform_(layer.self_attention.v_proj.weight, -bound, bound)
            torch.nn.init.zeros_(layer.self_attention.o_proj.weight)

            torch.nn.init.uniform_(layer.mlp.gate_proj.weight, -0.4 * bound, 0.4 * bound)
            torch.nn.init.uniform_(layer.mlp.up_proj.weight, -0.4 * bound, 0.4 * bound)
            torch.nn.init.zeros_(layer.mlp.down_proj.weight)

            torch.nn.init.ones_(layer.attn_res.norm.weight)
            torch.nn.init.ones_(layer.mlp_res.norm.weight)
            torch.nn.init.zeros_(layer.attn_res.proj.weight)
            torch.nn.init.zeros_(layer.mlp_res.proj.weight)

        torch.nn.init.ones_(self.norm.weight)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        # input_ids [B, S]
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        
        # 计算start_pos
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        
        # Embedding + dropout
        hidden_states = self.dropout(
            self.embed_tokens(input_ids)
        )  # [bsz, seq_len, hidden]
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )

        blocks = []
        partial_block = None
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            blocks, partial_block, present = layer(
                hidden_states,
                blocks,
                partial_block,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            hidden_states = partial_block
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = 0.0
        return hidden_states, presents, aux_loss


class NanoMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NanoMindConfig

    def __init__(self, config: NanoMindConfig):
        super().__init__(config)
        self.model = NanoMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self):
        self.model.init_weights()
        torch.nn.init.normal_(
            self.lm_head.weight, mean=0.0, std=self.config.hidden_size**-0.5
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output
        
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(
                input_ids=input_ids[:, past_len:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]): logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
