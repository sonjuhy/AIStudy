from __future__ import annotations

import torch

from gpt2.config import GPTConfig
from gpt2.gpt import GPT


def test_gpt_forward_with_softmax_attention() -> None:
    config: GPTConfig = GPTConfig(
        n_layer=2, n_head=2, n_embd=32, block_size=16,
        vocab_size=100, attention_type="softmax",
    )
    model: GPT = GPT(config)
    idx: torch.Tensor = torch.randint(0, 100, (2, 16))

    logits, loss = model(idx, targets=idx)

    assert logits.shape == (2, 16, 100)
    assert loss.item() > 0


def test_gpt_forward_with_linear_attention() -> None:
    config: GPTConfig = GPTConfig(
        n_layer=2, n_head=2, n_embd=32, block_size=16,
        vocab_size=100, attention_type="linear",
    )
    model: GPT = GPT(config)
    idx: torch.Tensor = torch.randint(0, 100, (2, 16))

    logits, loss = model(idx, targets=idx)

    assert logits.shape == (2, 16, 100)
    assert loss.item() > 0


def test_gpt_forward_without_targets_returns_none_loss() -> None:
    config: GPTConfig = GPTConfig(
        n_layer=2, n_head=2, n_embd=32, block_size=16,
        vocab_size=100, attention_type="softmax",
    )
    model: GPT = GPT(config)
    idx: torch.Tensor = torch.randint(0, 100, (2, 16))

    logits, loss = model(idx)

    assert logits.shape == (2, 16, 100)
    assert loss is None


def test_gpt_generate_appends_max_new_tokens() -> None:
    config: GPTConfig = GPTConfig(
        n_layer=2, n_head=2, n_embd=32, block_size=16,
        vocab_size=100, attention_type="linear",
    )
    model: GPT = GPT(config)
    model.eval()
    prompt: torch.Tensor = torch.randint(0, 100, (1, 4))

    out: torch.Tensor = model.generate(prompt, max_new_tokens=5)

    assert out.shape == (1, 9)
    assert torch.equal(out[:, :4], prompt)


def test_gpt_config_rejects_unknown_attention_type() -> None:
    import pytest

    with pytest.raises(ValueError):
        GPTConfig(attention_type="unknown")
