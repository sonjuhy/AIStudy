from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gpt2 import data


def test_get_batch_shapes_and_next_token_offset(tmp_path: Path) -> None:
    """x, y shape이 (batch_size, block_size)이고 y가 x보다 한 토큰 뒤여야 한다."""
    tokens = np.arange(1000, dtype=np.uint16)
    bin_path = tmp_path / "toy.bin"
    tokens.tofile(bin_path)

    x, y = data.get_batch(bin_path, batch_size=4, block_size=8, device="cpu")

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # y는 x를 한 칸 뒤로 민 다음 토큰 시퀀스여야 한다 (autoregressive target)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_num_tokens_returns_token_count(tmp_path: Path) -> None:
    tokens = np.arange(123, dtype=np.uint16)
    bin_path = tmp_path / "toy.bin"
    tokens.tofile(bin_path)

    assert data.num_tokens(bin_path) == 123


def test_prepare_dataset_reuses_existing_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """train.bin/val.bin이 이미 있으면 데이터셋 로더를 다시 호출하지 않아야 한다."""

    def fail_if_called() -> tuple[list[str], list[str]]:
        raise AssertionError("cache가 있는데 데이터셋 로더가 다시 호출되었다")

    monkeypatch.setitem(data.DATASET_LOADERS, "tiny_shakespeare", fail_if_called)

    cache_dir = tmp_path / "cache"
    dataset_dir = cache_dir / "tiny_shakespeare"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.bin").write_bytes(b"")
    (dataset_dir / "val.bin").write_bytes(b"")

    paths = data.prepare_dataset("tiny_shakespeare", cache_dir)

    assert paths["train"] == dataset_dir / "train.bin"
    assert paths["val"] == dataset_dir / "val.bin"


def test_prepare_dataset_rejects_unknown_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        data.prepare_dataset("not-a-real-dataset", tmp_path)
