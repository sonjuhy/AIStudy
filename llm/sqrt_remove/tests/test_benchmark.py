from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gpt2 import benchmark as benchmark_module
from gpt2.config import GPTConfig
from gpt2.gpt import GPT


@pytest.fixture()
def toy_model() -> GPT:
    config = GPTConfig(n_layer=1, n_head=1, n_embd=8, block_size=8, vocab_size=50, attention_type="linear")
    return GPT(config)


@pytest.fixture()
def toy_batch() -> torch.Tensor:
    return torch.randint(0, 50, (2, 8))


def test_measure_train_step_time_returns_positive_float(toy_model: GPT, toy_batch: torch.Tensor) -> None:
    result = benchmark_module.measure_train_step_time(toy_model, toy_batch, n_warmup=1, n_iters=2)

    assert isinstance(result, float)
    assert result > 0


def test_measure_train_step_time_raises_on_zero_iters(toy_model: GPT, toy_batch: torch.Tensor) -> None:
    with pytest.raises(ZeroDivisionError):
        benchmark_module.measure_train_step_time(toy_model, toy_batch, n_warmup=1, n_iters=0)


def test_measure_inference_tokens_per_sec_returns_positive_float(toy_model: GPT) -> None:
    prompt = torch.randint(0, 50, (1, 1))

    result = benchmark_module.measure_inference_tokens_per_sec(toy_model, prompt, max_new_tokens=4)

    assert isinstance(result, float)
    assert result > 0


def _write_toy_bin(path: Path, n_tokens: int = 500, vocab: int = 50) -> None:
    tokens = np.random.randint(0, vocab, size=n_tokens).astype(np.uint16)
    tokens.tofile(path)


def test_run_single_benchmark_returns_valid_result(tmp_path: Path) -> None:
    val_bin = tmp_path / "val.bin"
    _write_toy_bin(val_bin)

    result = benchmark_module.run_single_benchmark(
        "linear",
        n_layer=1,
        n_head=1,
        n_embd=8,
        block_size=8,
        batch_size=2,
        val_bin=val_bin,
        device="cpu",
        seed=0,
        n_warmup=1,
        n_train_iters=2,
        n_inference_tokens=4,
        eval_iters=2,
    )

    assert result.attention_type == "linear"
    assert result.train_step_time_sec > 0
    assert result.inference_tokens_per_sec > 0
    assert result.val_perplexity > 0
    assert result.extra["n_params"] > 0


def test_main_writes_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    val_bin = tmp_path / "val.bin"
    train_bin = tmp_path / "train.bin"
    _write_toy_bin(val_bin)
    _write_toy_bin(train_bin)

    monkeypatch.setattr(
        benchmark_module,
        "prepare_dataset",
        lambda name, cache_dir: {"train": train_bin, "val": val_bin},
    )

    output_path = tmp_path / "results.json"
    argv = [
        "--dataset", "toy",
        "--n-layer", "1",
        "--n-head", "1",
        "--n-embd", "8",
        "--block-sizes", "8",
        "--batch-size", "2",
        "--seeds", "0",
        "--cache-dir", str(tmp_path / "cache"),
        "--output", str(output_path),
    ]

    results = benchmark_module.main(argv)

    assert len(results) == 2  # softmax + linear
    assert output_path.exists()
    with output_path.open() as f:
        data = json.load(f)
    assert {r["attention_type"] for r in data} == {"softmax", "linear"}
