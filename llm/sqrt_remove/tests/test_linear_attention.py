from __future__ import annotations

import torch

from gpt2.attention.linear_attention import LinearAttention


def test_linear_attention_output_shape() -> None:
    """출력 shape이 입력 shape과 동일해야 한다."""
    batch_size: int = 2
    seq_len: int = 16
    n_embd: int = 32
    n_head: int = 4

    attn: LinearAttention = LinearAttention(n_embd=n_embd, n_head=n_head)
    x: torch.Tensor = torch.randn(batch_size, seq_len, n_embd)

    output: torch.Tensor = attn(x)

    assert output.shape == (batch_size, seq_len, n_embd)


def test_linear_attention_no_exp_or_softmax_call(monkeypatch: "pytest.MonkeyPatch") -> None:
    """구현 내부에서 torch.softmax / torch.exp를 호출하지 않아야 한다."""
    import torch.nn.functional as F

    called: dict[str, bool] = {"softmax": False, "exp": False}

    original_softmax = F.softmax
    original_exp = torch.exp

    def fake_softmax(*args: object, **kwargs: object) -> torch.Tensor:
        called["softmax"] = True
        return original_softmax(*args, **kwargs)  # type: ignore[arg-type]

    def fake_exp(*args: object, **kwargs: object) -> torch.Tensor:
        called["exp"] = True
        return original_exp(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(F, "softmax", fake_softmax)
    monkeypatch.setattr(torch, "exp", fake_exp)

    attn: LinearAttention = LinearAttention(n_embd=32, n_head=4)
    x: torch.Tensor = torch.randn(2, 16, 32)
    attn(x)

    assert called["softmax"] is False
    assert called["exp"] is False


def test_linear_attention_causal_mask_applied() -> None:
    """causal 마스크가 적용되어 미래 토큰 정보가 과거 위치에 영향을 주지 않아야 한다."""
    attn: LinearAttention = LinearAttention(n_embd=16, n_head=2)
    x: torch.Tensor = torch.randn(1, 8, 16)

    x_modified: torch.Tensor = x.clone()
    x_modified[:, -1, :] = torch.randn(1, 16)  # 마지막(미래) 토큰만 변경

    out_original: torch.Tensor = attn(x)
    out_modified: torch.Tensor = attn(x_modified)

    # 마지막 토큰을 제외한 앞부분 출력은 동일해야 함 (causal 특성)
    assert torch.allclose(out_original[:, :-1, :], out_modified[:, :-1, :], atol=1e-5)


def test_linear_attention_causal_mask_applied_across_chunks() -> None:
    """chunk 경계를 넘는 시퀀스에서도 causal 특성이 유지되어야 한다."""
    attn: LinearAttention = LinearAttention(n_embd=16, n_head=2, chunk_size=4)
    x: torch.Tensor = torch.randn(1, 20, 16)

    x_modified: torch.Tensor = x.clone()
    x_modified[:, -1, :] = torch.randn(1, 16)

    out_original: torch.Tensor = attn(x)
    out_modified: torch.Tensor = attn(x_modified)

    assert torch.allclose(out_original[:, :-1, :], out_modified[:, :-1, :], atol=1e-5)
