"""Kaser-1 decoder-only Transformer with safe KV caching and online fine-tuning."""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import KaserConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        ).to(dtype=x.dtype)
        return x * scale * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: KaserConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout_p = config.dropout

        if self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")

        self.qkv = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias
        )
        self.proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias
        )
        self.resid_dropout = nn.Dropout(config.dropout)

        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        angles = torch.outer(
            positions.to(device=q.device, dtype=torch.float32),
            self.inv_freq.to(device=q.device),
        )
        emb = torch.cat((angles, angles), dim=-1)
        cos = emb.cos().to(dtype=q.dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=q.dtype)[None, None, :, :]
        return (
            q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        batch, seq_len, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=-1)
        shape = (batch, seq_len, self.n_head, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)

        past_len = kv_cache[0].size(2) if kv_cache is not None else 0
        if positions is None:
            positions = torch.arange(
                past_len, past_len + seq_len, device=x.device
            )
        else:
            positions = positions.to(device=x.device)
            if positions.ndim != 1 or positions.numel() != seq_len:
                raise ValueError("positions must be a 1-D tensor matching sequence length")

        q, k = self._apply_rope(q, k, positions)

        if kv_cache is not None:
            k = torch.cat((kv_cache[0], k), dim=2)
            v = torch.cat((kv_cache[1], v), dim=2)

        # SDPA's is_causal=True uses an upper-left causal mask. With a KV cache,
        # query and key lengths differ, so use an explicit offset-aware mask.
        if kv_cache is None:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            key_positions = torch.arange(k.size(2), device=x.device)
            query_positions = past_len + torch.arange(seq_len, device=x.device)
            allowed = key_positions[None, :] <= query_positions[:, None]
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=allowed,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        out = self.resid_dropout(self.proj(y))
        new_cache = (k, v) if use_cache else None
        return out, new_cache


class SwiGLU(nn.Module):
    def __init__(self, config: KaserConfig):
        super().__init__()
        hidden = int((8 * config.n_embd / 3 + 63) // 64 * 64)
        self.gate = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: KaserConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        attn_out, new_cache = self.attn(
            self.attn_norm(x),
            positions=positions,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class KaserCoreEngine(nn.Module):
    def __init__(self, config: KaserConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize modules before tying weights, so shared weights are initialized once.
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * max(config.n_layer, 1)),
                )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[KVCache]] = None,
        use_cache: bool = False,
    ):
        if idx.ndim != 2:
            raise ValueError("input ids must have shape [batch, sequence]")
        if idx.size(1) == 0:
            raise ValueError("input sequence must not be empty")
        if kv_caches is not None and len(kv_caches) != len(self.blocks):
            raise ValueError("kv_caches must contain one cache per Transformer block")

        if idx.size(1) > self.config.block_size:
            removed = idx.size(1) - self.config.block_size
            idx = idx[:, -self.config.block_size:]
            if targets is not None:
                targets = targets[:, -self.config.block_size:]
            if positions is not None:
                positions = positions[removed:]

        if targets is not None and targets.shape != idx.shape:
            raise ValueError("targets must have the same shape as idx")

        x = self.dropout(self.token_embedding(idx))
        new_caches: Optional[List[KVCache]] = [] if use_cache else None

        for layer_index, block in enumerate(self.blocks):
            cache = kv_caches[layer_index] if kv_caches is not None else None
            x, new_cache = block(
                x,
                positions=positions,
                kv_cache=cache,
                use_cache=use_cache,
            )
            if use_cache:
                new_caches.append(new_cache)

        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        if use_cache:
            return logits, loss, new_caches
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 160,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.08,
        eos_token_id: Optional[int] = 2,
    ) -> torch.Tensor:
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError("generate() expects non-empty input ids shaped [batch, sequence]")
        if max_new_tokens <= 0:
            return idx.clone()

        was_training = self.training
        self.eval()
        try:
            idx = idx.clone()
            block_size = self.config.block_size
            initial_len = idx.size(1)
            context = idx[:, -block_size:]
            start_pos = initial_len - context.size(1)
            positions = torch.arange(
                start_pos, initial_len, device=idx.device
            )
            logits, _, caches = self(
                context, positions=positions, use_cache=True
            )
            logits = logits[:, -1, :].float()

            finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
            for step in range(max_new_tokens):
                next_token = self._sample_next(
                    logits, idx, temperature, top_k, top_p, repetition_penalty
                )
                if eos_token_id is not None:
                    next_token = torch.where(
                        finished.unsqueeze(-1),
                        torch.full_like(next_token, eos_token_id),
                        next_token,
                    )
                    finished |= next_token.squeeze(-1).eq(eos_token_id)
                idx = torch.cat((idx, next_token), dim=1)
                if (eos_token_id is not None and finished.all()) or step + 1 == max_new_tokens:
                    break

                # Keep a bounded rolling cache, then append the newest token.
                keep = max(block_size - 1, 0)
                if keep == 0:
                    caches = [(k[:, :, :0], v[:, :, :0]) for k, v in caches]
                else:
                    caches = [
                        (k[:, :, -keep:], v[:, :, -keep:])
                        for k, v in caches
                    ]

                position = torch.tensor([initial_len + step], device=idx.device)
                logits, _, caches = self(
                    next_token,
                    positions=position,
                    kv_caches=caches,
                    use_cache=True,
                )
                logits = logits[:, -1, :].float()
            return idx
        finally:
            self.train(was_training)

    @torch.no_grad()
    def _sample_next(
        self,
        logits: torch.Tensor,
        idx: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> torch.Tensor:
        logits = logits.clone()

        if repetition_penalty and repetition_penalty > 1.0:
            recent = idx[:, -128:]
            for batch_index in range(idx.size(0)):
                tokens = torch.unique(recent[batch_index])
                values = logits[batch_index, tokens]
                adjusted = torch.where(
                    values > 0,
                    values / repetition_penalty,
                    values * repetition_penalty,
                )
                logits[batch_index, tokens] = adjusted

        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / max(float(temperature), 1e-5)

        if top_k and 0 < top_k < logits.size(-1):
            cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                logits, descending=True, dim=-1
            )
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            remove = (sorted_probs.cumsum(dim=-1) - sorted_probs) > top_p
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(
                1, sorted_indices, sorted_logits
            )

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def step_learn(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Perform one supervised update, masking prompt tokens from the loss."""
        if prompt_ids.ndim != 2 or response_ids.ndim != 2:
            raise ValueError("prompt_ids and response_ids must be [batch, sequence]")
        if prompt_ids.size(0) != response_ids.size(0):
            raise ValueError("prompt and response batch sizes must match")
        if prompt_ids.size(1) == 0 or response_ids.size(1) == 0:
            raise ValueError("prompt and response must both be non-empty")

        was_training = self.training
        self.train()
        try:
            full_ids = torch.cat((prompt_ids, response_ids), dim=1)
            inputs = full_ids[:, :-1]
            targets = full_ids[:, 1:].clone()
            prompt_len = prompt_ids.size(1)
            targets[:, :max(prompt_len - 1, 0)] = -100

            # Bound the sequence without silently mismatching targets.
            if inputs.size(1) > self.config.block_size:
                inputs = inputs[:, -self.config.block_size:]
                targets = targets[:, -self.config.block_size:]

            optimizer.zero_grad(set_to_none=True)
            _, loss = self(inputs, targets=targets)
            if loss is None or not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("Non-finite loss; optimizer update cancelled")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            return float(loss.detach().cpu())
        finally:
            self.train(was_training)
