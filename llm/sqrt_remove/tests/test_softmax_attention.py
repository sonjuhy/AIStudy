from __future__ import annotations

import torch

from gpt2.attention.softmax_attention import SoftmaxAttention


def test_softmax_attention_output_shape() -> None:
    """출력 shape이 입력 shape과 동일해야 한다 (LinearAttention과 동일한 인터페이스 계약)."""
    batch_size: int = 2
    seq_len: int = 16
    n_embd: int = 32
    n_head: int = 4

    attn: SoftmaxAttention = SoftmaxAttention(n_embd=n_embd, n_head=n_head)
    x: torch.Tensor = torch.randn(batch_size, seq_len, n_embd)

    output: torch.Tensor = attn(x)

    assert output.shape == (batch_size, seq_len, n_embd)


def test_softmax_attention_causal_mask_applied() -> None:
    """causal 마스크가 적용되어 미래 토큰 정보가 과거 위치에 영향을 주지 않아야 한다."""
    attn: SoftmaxAttention = SoftmaxAttention(n_embd=16, n_head=2)
    x: torch.Tensor = torch.randn(1, 8, 16)

    x_modified: torch.Tensor = x.clone()
    x_modified[:, -1, :] = torch.randn(1, 16)  # 마지막(미래) 토큰만 변경

    out_original: torch.Tensor = attn(x)
    out_modified: torch.Tensor = attn(x_modified)

    assert torch.allclose(out_original[:, :-1, :], out_modified[:, :-1, :], atol=1e-5)


def test_softmax_attention_weights_sum_to_one() -> None:
    """각 쿼리 위치에서 attention 가중치 합이 1이어야 한다 (softmax 정규화 검증)."""
    n_embd, n_head, seq_len = 16, 2, 8
    attn: SoftmaxAttention = SoftmaxAttention(n_embd=n_embd, n_head=n_head)
    x: torch.Tensor = torch.randn(1, seq_len, n_embd)

    weights: torch.Tensor = attn.attention_weights(x)  # (B, n_head, T, T)

    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
