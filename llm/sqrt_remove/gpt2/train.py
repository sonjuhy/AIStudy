from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from gpt2.config import GPTConfig
from gpt2.data import get_batch, num_tokens, prepare_dataset
from gpt2.gpt import GPT


def build_lr_lambda(warmup_steps: int, max_steps: int, min_lr_ratio: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if step > max_steps:
            return min_lr_ratio
        decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr_ratio + coeff * (1.0 - min_lr_ratio)

    return lr_lambda


@torch.no_grad()
def estimate_val_loss(
    model: GPT, val_bin: Path, batch_size: int, block_size: int, device: str, eval_iters: int = 50
) -> float:
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(val_bin, batch_size, block_size, device)
        _, loss = model(x, targets=y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def save_checkpoint(
    path: Path, model: GPT, optimizer: torch.optim.Optimizer, step: int, config: GPTConfig
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(path: Path, model: GPT, optimizer: torch.optim.Optimizer, device: str) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-2 Softmax/Linear Attention 비교 학습 스크립트")
    parser.add_argument("--attention-type", choices=["softmax", "linear"], default="softmax")
    parser.add_argument("--dataset", default="tiny_shakespeare")
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--ckpt-interval", type=int, default=200)
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="1 epoch(=train.bin 토큰 수 // (batch_size*block_size) 스텝)마다 "
        "epoch_N.pt로 별도 체크포인트를 저장",
    )
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--log-path", default=None, help="지정하면 CSV 형식으로 스텝별 로그를 append")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", action="store_true", help="ckpt-dir/{attention-type}/latest.pt에서 재개")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    paths = prepare_dataset(args.dataset, args.cache_dir)
    steps_per_epoch = max(1, num_tokens(paths["train"]) // (args.batch_size * args.block_size))
    if args.save_every_epoch:
        print(f"1 epoch ≈ {steps_per_epoch} steps (train tokens // (batch_size*block_size))")

    config = GPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        vocab_size=50257,
        attention_type=args.attention_type,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.warmup_steps, args.max_steps, args.min_lr_ratio)
    )

    ckpt_path = Path(args.ckpt_dir) / args.attention_type / "latest.pt"
    start_step = 0
    if args.resume and ckpt_path.exists():
        start_step = load_checkpoint(ckpt_path, model, optimizer, device) + 1
        print(f"resumed from step {start_step}")

    log_file = None
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not log_path.exists()
        log_file = log_path.open("a")
        if is_new:
            log_file.write("step,attention_type,train_loss,val_loss,val_ppl,elapsed_sec\n")

    model.train()
    t0 = time.perf_counter()
    for step in range(start_step, args.max_steps):
        x, y = get_batch(paths["train"], args.batch_size, args.block_size, device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        is_last_step = step == args.max_steps - 1
        epoch = step // steps_per_epoch
        is_epoch_boundary = args.save_every_epoch and (step + 1) % steps_per_epoch == 0

        if step % args.eval_interval == 0 or is_last_step:
            val_loss = estimate_val_loss(
                model, paths["val"], args.batch_size, args.block_size, device, args.eval_iters
            )
            elapsed = time.perf_counter() - t0
            val_ppl = math.exp(val_loss)
            print(
                f"step {step} (epoch {epoch}): train_loss={loss.item():.4f} val_loss={val_loss:.4f} "
                f"val_ppl={val_ppl:.2f} elapsed={elapsed:.1f}s"
            )
            if log_file:
                log_file.write(
                    f"{step},{args.attention_type},{loss.item():.6f},{val_loss:.6f},"
                    f"{val_ppl:.6f},{elapsed:.3f}\n"
                )
                log_file.flush()

        if step > start_step and (step % args.ckpt_interval == 0 or is_last_step):
            save_checkpoint(ckpt_path, model, optimizer, step, config)

        if is_epoch_boundary:
            epoch_num = (step + 1) // steps_per_epoch
            epoch_ckpt_path = ckpt_path.parent / f"epoch_{epoch_num}.pt"
            save_checkpoint(epoch_ckpt_path, model, optimizer, step, config)
            save_checkpoint(ckpt_path, model, optimizer, step, config)  # resume용 latest도 갱신
            print(f"saved epoch {epoch_num} checkpoint -> {epoch_ckpt_path}")

    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
