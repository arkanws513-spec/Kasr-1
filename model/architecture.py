"""Small decoder-only Transformer with RoPE, causal attention and SwiGLU."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.config import KaserConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return x * scale * self.weight

def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def _apply_rope(q, k, positions):
    dim = q.size(-1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim))
    angles = torch.einsum("t,d->td", positions.float(), inv_freq)
    emb = torch.cat((angles, angles), dim=-1)[None, None, :, :]
    cos, sin = emb.cos().to(q.dtype), emb.sin().to(q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

class CausalSelfAttention(nn.Module):
    def __init__(self, config: KaserConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        shape = (b, t, self.n_head, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        q, k = _apply_rope(q, k, torch.arange(t, device=x.device))
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask[None, None, :, :], torch.finfo(scores.dtype).min)
        probs = self.attn_dropout(F.softmax(scores.float(), dim=-1).to(scores.dtype))
        y = (probs @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(y))

class SwiGLU(nn.Module):
    def __init__(self, config: KaserConfig):
        super().__init__()
        hidden = int((8 * config.n_embd / 3 + 63) // 64 * 64)
        self.gate = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))

class KaserCoreEngine(nn.Module):
    def __init__(self, config: KaserConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        if idx.ndim != 2: raise ValueError("input ids must have shape [batch, sequence]")
        if idx.size(1) > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            if targets is not None: targets = targets[:, -self.config.block_size:]
        x = self.dropout(self.token_embedding(idx))
        for block in self.blocks: x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=160, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.08):
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :].float()
            if repetition_penalty > 1:
                for token in set(idx[0, -128:].tolist()):
                    logits[:, token] = logits[:, token] / repetition_penalty if logits[:, token] > 0 else logits[:, token] * repetition_penalty
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k and top_k < logits.size(-1):
                    cutoff = torch.topk(logits, top_k).values[:, -1, None]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                if top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    remove = probs.cumsum(-1) - probs > top_p
                    sorted_logits[remove] = float("-inf")
                    logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
