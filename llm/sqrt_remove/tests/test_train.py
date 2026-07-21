from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gpt2 import train as train_module
from gpt2.config import GPTConfig
from gpt2.gpt import GPT


def test_build_lr_lambda_warmup_then_decay() -> None:
    lr_lambda = train_module.build_lr_lambda(warmup_steps=10, max_steps=100, min_lr_ratio=0.1)

    assert lr_lambda(0) == pytest.approx(1 / 10)
    assert lr_lambda(10) == pytest.approx(1.0, abs=1e-6)
    assert lr_lambda(100) == pytest.approx(0.1, abs=1e-6)
    assert lr_lambda(200) == pytest.approx(0.1, abs=1e-6)  # max_steps 이후는 min_lr_ratio 유지


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = GPTConfig(n_layer=1, n_head=1, n_embd=8, block_size=8, vocab_size=50, attention_type="linear")
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "ckpt.pt"
    train_module.save_checkpoint(ckpt_path, model, optimizer, step=7, config=config)

    new_model = GPT(config)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    loaded_step = train_module.load_checkpoint(ckpt_path, new_model, new_optimizer, device="cpu")

    assert loaded_step == 7
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


def test_main_runs_end_to_end_with_synthetic_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train.py의 전체 루프(forward/backward/optimizer/eval/checkpoint)가
    합성 데이터로 에러 없이 끝까지 돌아가는지 확인하는 스모크 테스트."""
    bin_dir = tmp_path / "cache" / "toy"
    bin_dir.mkdir(parents=True)
    tokens = np.random.randint(0, 100, size=500).astype(np.uint16)
    train_bin = bin_dir / "train.bin"
    val_bin = bin_dir / "val.bin"
    tokens.tofile(train_bin)
    tokens.tofile(val_bin)

    monkeypatch.setattr(
        train_module,
        "prepare_dataset",
        lambda name, cache_dir: {"train": train_bin, "val": val_bin},
    )

    ckpt_dir = tmp_path / "checkpoints"
    argv = [
        "--attention-type", "linear",
        "--dataset", "toy",
        "--n-layer", "1",
        "--n-head", "1",
        "--n-embd", "8",
        "--block-size", "8",
        "--batch-size", "2",
        "--max-steps", "2",
        "--warmup-steps", "1",
        "--eval-interval", "1",
        "--eval-iters", "1",
        "--ckpt-interval", "1",
        "--cache-dir", str(tmp_path / "cache"),
        "--ckpt-dir", str(ckpt_dir),
    ]

    train_module.main(argv)

    assert (ckpt_dir / "linear" / "latest.pt").exists()


def test_main_saves_checkpoint_at_every_epoch_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--save-every-epoch를 주면 epoch 경계마다 epoch_N.pt가 별도로 저장되어야 한다."""
    bin_dir = tmp_path / "cache" / "toy"
    bin_dir.mkdir(parents=True)
    # batch_size=2, block_size=8 -> 스텝당 16 토큰 소비. 32 토큰 = 1 epoch당 2 스텝.
    tokens = np.random.randint(0, 100, size=32).astype(np.uint16)
    train_bin = bin_dir / "train.bin"
    val_bin = bin_dir / "val.bin"
    tokens.tofile(train_bin)
    tokens.tofile(val_bin)

    monkeypatch.setattr(
        train_module,
        "prepare_dataset",
        lambda name, cache_dir: {"train": train_bin, "val": val_bin},
    )

    ckpt_dir = tmp_path / "checkpoints"
    argv = [
        "--attention-type", "linear",
        "--dataset", "toy",
        "--n-layer", "1",
        "--n-head", "1",
        "--n-embd", "8",
        "--block-size", "8",
        "--batch-size", "2",
        "--max-steps", "4",
        "--warmup-steps", "1",
        "--eval-interval", "10",
        "--eval-iters", "1",
        "--ckpt-interval", "1000",  # step 간격 저장은 이번 테스트에서 끔
        "--save-every-epoch",
        "--cache-dir", str(tmp_path / "cache"),
        "--ckpt-dir", str(ckpt_dir),
    ]

    train_module.main(argv)

    assert (ckpt_dir / "linear" / "epoch_1.pt").exists()
    assert (ckpt_dir / "linear" / "epoch_2.pt").exists()
