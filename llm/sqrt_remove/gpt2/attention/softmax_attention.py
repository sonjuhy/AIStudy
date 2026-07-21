from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxAttention(nn.Module):
    """표준 causal Softmax Attention (GPT-2 baseline).

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def _to_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return x.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_head, T, T) causal softmax 가중치를 반환한다 (검증/디버깅용)."""
        batch_size, seq_len, n_embd = x.shape
        q, k, _ = self.c_attn(x).split(n_embd, dim=2)
        q = self._to_heads(q, batch_size, seq_len)
        k = self._to_heads(k, batch_size, seq_len)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        return F.softmax(scores, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape

        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        q = self._to_heads(q, batch_size, seq_len)
        k = self._to_heads(k, batch_size, seq_len)
        v = self._to_heads(v, batch_size, seq_len)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        return self.c_proj(out)
