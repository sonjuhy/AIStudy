from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SmolLMConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids[:, :, None].float() * self.inv_freq[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: SmolLMConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = (
            self.q_proj(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        q, k = apply_rotary(q, k, cos, sin)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        attn = attn + causal_mask
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(
            batch, seq_len, self.num_heads * self.head_dim
        )
        return self.o_proj(out)


class SwiGLUMLP(nn.Module):
    def __init__(self, config: SmolLMConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: SmolLMConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class SmolLM(nn.Module):
    """SmolLM-135M(LlamaForCausalLM) 구조를 그대로 재작성한 PyTorch 모델.

    가중치는 코드와 분리되어 HF safetensors 체크포인트에서 로드한다 (from_pretrained_weights).
    """

    def __init__(self, config: SmolLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        cos, sin = self.rotary_emb(position_ids)

        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

    @classmethod
    def from_pretrained_weights(cls, weights_dir: str) -> "SmolLM":
        """HF 체크포인트(model.safetensors, config.json)에서 가중치만 가져와 이 재작성 모델에 채운다."""
        import json
        import os

        from safetensors.torch import load_file

        with open(os.path.join(weights_dir, "config.json"), encoding="utf-8") as f:
            hf_config = json.load(f)

        config = SmolLMConfig(
            vocab_size=hf_config["vocab_size"],
            hidden_size=hf_config["hidden_size"],
            intermediate_size=hf_config["intermediate_size"],
            num_hidden_layers=hf_config["num_hidden_layers"],
            num_attention_heads=hf_config["num_attention_heads"],
            num_key_value_heads=hf_config["num_key_value_heads"],
            max_position_embeddings=hf_config["max_position_embeddings"],
            rms_norm_eps=hf_config["rms_norm_eps"],
            rope_theta=hf_config["rope_theta"],
            tie_word_embeddings=hf_config["tie_word_embeddings"],
        )
        model = cls(config)

        state_dict = load_file(os.path.join(weights_dir, "model.safetensors"))
        mapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key[len("model.") :] if key.startswith("model.") else key
            mapped[new_key] = value
        if config.tie_word_embeddings:
            mapped.setdefault("lm_head.weight", mapped["embed_tokens.weight"])

        missing, unexpected = model.load_state_dict(mapped, strict=False)
        if unexpected:
            raise ValueError(f"unexpected keys while loading weights: {unexpected}")
        real_missing = [
            k
            for k in missing
            if k != "lm_head.weight" or not config.tie_word_embeddings
        ]
        if real_missing:
            raise ValueError(f"missing keys while loading weights: {real_missing}")

        return model
