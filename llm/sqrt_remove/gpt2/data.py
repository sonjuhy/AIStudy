from __future__ import annotations

from pathlib import Path

import numpy as np
import tiktoken
import torch

_ENC = tiktoken.get_encoding("gpt2")

DATASET_LOADERS = {
    "tiny_shakespeare": lambda: _load_tiny_shakespeare(),
    "wikitext-103": lambda: _load_wikitext103(),
}


def _load_tiny_shakespeare() -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    # 원본 "tiny_shakespeare"(HF)는 스크립트 기반 로더라 최신 datasets(>=4)에서
    # `RuntimeError: Dataset scripts are no longer supported`로 실패한다.
    # winglian/tiny-shakespeare는 동일한 텍스트를 스크립트 없이(parquet) 제공하는 미러.
    ds = load_dataset("winglian/tiny-shakespeare")
    return list(ds["train"]["text"]), list(ds["test"]["text"])


def _load_wikitext103() -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    # "wikitext"(원 경로)도 스크립트 기반이라 실패한다.
    # 데이터셋 소유자인 Salesforce가 올린 스크립트 없는(parquet) 버전을 사용한다.
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    return ds["train"]["text"], ds["validation"]["text"]


def _tokenize_texts(texts: list[str]) -> np.ndarray:
    """텍스트 목록을 GPT-2 BPE(tiktoken)로 토큰화하고, 문서 사이는 eot 토큰으로 구분한다."""
    ids: list[int] = []
    for text in texts:
        if not text:
            continue
        ids.extend(_ENC.encode_ordinary(text))
        ids.append(_ENC.eot_token)
    return np.array(ids, dtype=np.uint16)


def prepare_dataset(name: str, cache_dir: str | Path) -> dict[str, Path]:
    """`name` 데이터셋을 토큰화해 train.bin / val.bin으로 캐싱한다.

    이미 캐시가 있으면 재사용한다 (Colab/Kaggle 세션마다 재다운로드·재토큰화 방지).
    """
    if name not in DATASET_LOADERS:
        raise ValueError(f"unknown dataset: {name!r} (expected one of {list(DATASET_LOADERS)})")

    cache_path = Path(cache_dir) / name
    cache_path.mkdir(parents=True, exist_ok=True)
    train_path = cache_path / "train.bin"
    val_path = cache_path / "val.bin"

    if train_path.exists() and val_path.exists():
        return {"train": train_path, "val": val_path}

    train_texts, val_texts = DATASET_LOADERS[name]()
    _tokenize_texts(train_texts).tofile(train_path)
    _tokenize_texts(val_texts).tofile(val_path)
    return {"train": train_path, "val": val_path}


def num_tokens(bin_path: Path) -> int:
    """bin 파일에 저장된 토큰 개수를 반환한다 (1 epoch = 이 값 / (batch_size*block_size) 스텝)."""
    return Path(bin_path).stat().st_size // np.dtype(np.uint16).itemsize


def get_batch(
    bin_path: Path, batch_size: int, block_size: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """캐싱된 토큰 bin 파일에서 (x, y) = (입력, 다음 토큰 타깃) 배치를 무작위 샘플링한다."""
    # 매 호출마다 memmap을 여는 이유: 이 배열을 워커 프로세스 간 공유해도 안전하며
    # 데이터셋 전체를 메모리에 올리지 않고 디스크에서 필요한 부분만 읽는다 (nanoGPT 관례).
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )

    if device != "cpu":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
