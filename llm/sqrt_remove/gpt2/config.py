from __future__ import annotations

from dataclasses import dataclass

ATTENTION_TYPES = ("softmax", "linear")


@dataclass
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50257  # tiktoken "gpt2" encoding
    attention_type: str = "softmax"  # "softmax" | "linear"

    def __post_init__(self) -> None:
        if self.attention_type not in ATTENTION_TYPES:
            raise ValueError(
                f"unknown attention_type: {self.attention_type!r} (expected one of {ATTENTION_TYPES})"
            )
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
