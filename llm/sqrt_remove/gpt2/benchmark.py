from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from gpt2.config import GPTConfig
from gpt2.data import get_batch, prepare_dataset
from gpt2.gpt import GPT


@dataclass
class BenchmarkResult:
    attention_type: str
    n_layer: int
    block_size: int
    train_step_time_sec: float
    inference_tokens_per_sec: float
    val_perplexity: float
    extra: dict[str, float] = field(default_factory=dict)


def measure_train_step_time(
    model: GPT, batch: torch.Tensor, n_warmup: int = 5, n_iters: int = 20
) -> float:
    device: torch.device = next(model.parameters()).device
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for _ in range(n_warmup):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(batch, targets=batch)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    start: float = time.perf_counter()
    for _ in range(n_iters):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(batch, targets=batch)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end: float = time.perf_counter()

    return (end - start) / n_iters


def measure_inference_tokens_per_sec(
    model: GPT, prompt: torch.Tensor, max_new_tokens: int = 128
) -> float:
    device: torch.device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start: float = time.perf_counter()
        model.generate(prompt, max_new_tokens=max_new_tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end: float = time.perf_counter()

    model.train()
    return max_new_tokens / (end - start)


@torch.no_grad()
def measure_val_perplexity(
    model: GPT, val_bin: Path, batch_size: int, block_size: int, device: str, eval_iters: int = 50
) -> float:
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(val_bin, batch_size, block_size, device)
        _, loss = model(x, targets=y)
        losses[i] = loss.item()
    model.train()
    return math.exp(losses.mean().item())


def run_single_benchmark(
    attention_type: str,
    *,
    n_layer: int,
    n_head: int,
    n_embd: int,
    block_size: int,
    batch_size: int,
    val_bin: Path,
    device: str,
    seed: int,
    n_warmup: int = 5,
    n_train_iters: int = 20,
    n_inference_tokens: int = 128,
    eval_iters: int = 50,
) -> BenchmarkResult:
    """동일 세션·동일 GPU 배정 내에서 하나의 attention_type을 측정한다.

    가이드 4장의 "주의": baseline/variant는 같은 프로세스·세션에서 연달아 측정해야
    GPU 배정 편차의 영향을 최소화할 수 있다. 이 함수를 baseline -> variant 순서로
    반복 호출하는 스크립트(main)가 그 요구사항을 만족시킨다.
    """
    torch.manual_seed(seed)
    config = GPTConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        vocab_size=50257,
        attention_type=attention_type,
    )
    model = GPT(config).to(device)

    batch, _ = get_batch(val_bin, batch_size, block_size, device)
    train_step_time = measure_train_step_time(model, batch, n_warmup=n_warmup, n_iters=n_train_iters)

    prompt = batch[:1, :1]
    inference_tps = measure_inference_tokens_per_sec(model, prompt, max_new_tokens=n_inference_tokens)

    val_ppl = measure_val_perplexity(model, val_bin, batch_size, block_size, device, eval_iters)

    n_params = sum(p.numel() for p in model.parameters())
    return BenchmarkResult(
        attention_type=attention_type,
        n_layer=n_layer,
        block_size=block_size,
        train_step_time_sec=train_step_time,
        inference_tokens_per_sec=inference_tps,
        val_perplexity=val_ppl,
        extra={"n_params": n_params},
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Softmax vs Linear Attention 벤치마크")
    parser.add_argument("--dataset", default="tiny_shakespeare")
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337])
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output", default="benchmark_results.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[BenchmarkResult]:
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    paths = prepare_dataset(args.dataset, args.cache_dir)

    results: list[BenchmarkResult] = []
    for block_size in args.block_sizes:
        for seed in args.seeds:
            # baseline(softmax) -> variant(linear) 순서로 연달아 측정 (GPU 배정 편차 최소화)
            for attention_type in ("softmax", "linear"):
                result = run_single_benchmark(
                    attention_type,
                    n_layer=args.n_layer,
                    n_head=args.n_head,
                    n_embd=args.n_embd,
                    block_size=block_size,
                    batch_size=args.batch_size,
                    val_bin=paths["val"],
                    device=device,
                    seed=seed,
                )
                print(
                    f"[{attention_type}] block_size={block_size} seed={seed} "
                    f"train_step={result.train_step_time_sec * 1000:.2f}ms "
                    f"infer={result.inference_tokens_per_sec:.1f}tok/s "
                    f"val_ppl={result.val_perplexity:.2f}"
                )
                results.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    main()
