"""A small decoder-only transformer.

Llama-style choices: RMSNorm, rotary position embeddings, SwiGLU MLP,
optional grouped-query attention, no biases, tied input/output embeddings.
Attention uses PyTorch's scaled_dot_product_attention, which picks a fused
kernel on CUDA and MPS where one is available.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def rope_cache(seq_len: int, head_dim: int, theta: float, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D). Rotate pairs (x1, x2) from the two halves of the head dim.
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None].to(x.dtype)
    sin = sin[None, None].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.n_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.n_kv_heads, self.head_dim).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if kv_cache is not None:
            if kv_cache.get("k") is not None:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # With a cache and a single new token, every cached position is visible.
        causal = kv_cache is None or T > 1
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, dropout_p=self.dropout if self.training else 0.0
        )
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = rope_cache(cfg.seq_len, cfg.d_model // cfg.n_heads, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale residual output projections by depth (GPT-2 trick).
        for name, p in self.named_parameters():
            if name.endswith("attn.o.weight") or name.endswith("mlp.down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    def forward(self, idx, targets=None, kv_caches=None, start_pos: int = 0):
        B, T = idx.shape
        assert start_pos + T <= self.cfg.seq_len, "sequence longer than seq_len"
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]

        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if kv_caches is None else kv_caches[i])
        x = self.norm(x)

        if targets is None:
            return self.lm_head(x[:, -1:, :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
        )
        return logits, loss

    def configure_optimizer(self, weight_decay, lr, betas, device_type):
        # Decay matrices only; leave norms (1-D params) alone.
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95,
                 stop_ids: set[int] | None = None):
        """Sample with a KV cache. idx: (1, T) prompt tokens. Returns new token ids."""
        return list(self.generate_stream(idx, max_new_tokens, temperature, top_k, top_p, stop_ids))

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95,
                        stop_ids: set[int] | None = None):
        """Like generate, but yields each new token id as soon as it is sampled."""
        self.eval()
        caches = [{} for _ in self.blocks]
        prompt = idx[:, -(self.cfg.seq_len - 1):]
        logits, _ = self(prompt, kv_caches=caches, start_pos=0)
        pos = prompt.size(1)
        for _ in range(max_new_tokens):
            if pos >= self.cfg.seq_len:
                break
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                if top_p < 1.0:
                    sp, si = probs.sort(descending=True)
                    keep = sp.cumsum(-1) - sp < top_p
                    sp = sp * keep
                    probs = torch.zeros_like(probs).scatter(-1, si, sp)
                    probs = probs / probs.sum(-1, keepdim=True)
                nxt = torch.multinomial(probs, 1)
            tok = nxt.item()
            yield tok
            if stop_ids and tok in stop_ids:
                break
            logits, _ = self(nxt, kv_caches=caches, start_pos=pos)
            pos += 1
