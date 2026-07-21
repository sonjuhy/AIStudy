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


def test_load_checkpoint_rejects_mismatched_config(tmp_path: Path) -> None:
    """서로 다른 실험 단계(예: 파이프라인 검증용 소형 모델 vs 본 실험용 GPT-2 small)의
    체크포인트를 같은 --ckpt-dir로 잘못 재사용하면, PyTorch의 난해한 shape 에러 대신
    바로 원인을 알 수 있는 명확한 에러를 내야 한다."""
    small_config = GPTConfig(n_layer=1, n_head=1, n_embd=8, block_size=8, vocab_size=50, attention_type="softmax")
    small_model = GPT(small_config)
    small_optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "ckpt.pt"
    train_module.save_checkpoint(ckpt_path, small_model, small_optimizer, step=3, config=small_config)

    big_config = GPTConfig(n_layer=2, n_head=2, n_embd=16, block_size=8, vocab_size=50, attention_type="softmax")
    big_model = GPT(big_config)
    big_optimizer = torch.optim.AdamW(big_model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="체크포인트의 모델 설정이 현재 실행 설정과 다릅니다"):
        train_module.load_checkpoint(ckpt_path, big_model, big_optimizer, device="cpu", config=big_config)


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


def test_parse_args_memory_saving_flags_default_off() -> None:
    args = train_module.parse_args([])

    assert args.grad_accum_steps == 1
    assert args.no_amp is False
    assert args.grad_checkpointing is False


def _write_toy_bin_pair(bin_dir: Path, n_tokens: int = 500) -> tuple[Path, Path]:
    bin_dir.mkdir(parents=True)
    tokens = np.random.randint(0, 100, size=n_tokens).astype(np.uint16)
    train_bin = bin_dir / "train.bin"
    val_bin = bin_dir / "val.bin"
    tokens.tofile(train_bin)
    tokens.tofile(val_bin)
    return train_bin, val_bin


def test_main_runs_with_grad_accum_and_checkpointing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--grad-accum-steps와 --grad-checkpointing을 같이 켜도 학습 루프가 끝까지 돈다
    (GPU 메모리 절약 옵션이 OOM 대응 수단으로 실제로 동작하는지 확인하는 스모크 테스트)."""
    train_bin, val_bin = _write_toy_bin_pair(tmp_path / "cache" / "toy")

    monkeypatch.setattr(
        train_module,
        "prepare_dataset",
        lambda name, cache_dir: {"train": train_bin, "val": val_bin},
    )

    ckpt_dir = tmp_path / "checkpoints"
    argv = [
        "--attention-type", "softmax",
        "--dataset", "toy",
        "--n-layer", "2",
        "--n-head", "1",
        "--n-embd", "8",
        "--block-size", "8",
        "--batch-size", "2",
        "--grad-accum-steps", "3",
        "--grad-checkpointing",
        "--max-steps", "2",
        "--warmup-steps", "1",
        "--eval-interval", "1",
        "--eval-iters", "1",
        "--ckpt-interval", "1",
        "--cache-dir", str(tmp_path / "cache"),
        "--ckpt-dir", str(ckpt_dir),
    ]

    train_module.main(argv)

    assert (ckpt_dir / "softmax" / "latest.pt").exists()


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
