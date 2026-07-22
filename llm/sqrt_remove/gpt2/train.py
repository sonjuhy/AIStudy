from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
# Colab에서도 노트북 셀이 아닌 `!python -m gpt2.train ...` subprocess로 실행되므로
# ipywidgets 기반 tqdm.notebook이 아니라 텍스트 기반 tqdm.std를 그대로 쓴다.
from tqdm import tqdm

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
    model: GPT,
    val_bin: Path,
    batch_size: int,
    block_size: int,
    device: str,
    eval_iters: int = 50,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> float:
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(val_bin, batch_size, block_size, device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
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


def load_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    device: str,
    config: GPTConfig | None = None,
) -> int:
    ckpt = torch.load(path, map_location=device)
    ckpt_config = ckpt.get("config")
    if config is not None and ckpt_config is not None and ckpt_config != asdict(config):
        raise ValueError(
            "체크포인트의 모델 설정이 현재 실행 설정과 다릅니다 (다른 실험 단계의 체크포인트를 "
            "잘못 재사용하려는 경우 흔히 발생합니다 — 예: 파이프라인 검증용 소형 모델과 본 실험용 "
            "GPT-2 small이 같은 --ckpt-dir을 공유한 경우).\n"
            f"  체크포인트 config: {ckpt_config}\n"
            f"  현재 config:       {asdict(config)}\n"
            "실험 단계별로 --ckpt-dir을 분리하거나, --resume 없이 새로 시작하세요."
        )
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]


def epoch_progress_bar(step: int, steps_per_epoch: int, max_steps: int, disable: bool = False) -> tqdm:
    """현재 step이 속한 epoch 전용 tqdm 진행바를 만든다.

    학습 루프는 epoch 경계(step // steps_per_epoch가 바뀔 때)마다 이 함수로 새 진행바를
    받아 이전 것과 교체한다 — "epoch 1/5", "epoch 2/5"처럼 매 epoch마다 0%로 리셋되는
    진행바가 된다. --resume으로 epoch 중간부터 시작하면 `initial`로 이미 지난 진행률을
    반영한다. 마지막 epoch은 max_steps에 걸려 steps_per_epoch보다 짧을 수 있다.
    """
    epoch = step // steps_per_epoch
    total_epochs = max(1, math.ceil(max_steps / steps_per_epoch))
    epoch_start_step = epoch * steps_per_epoch
    epoch_total = min(steps_per_epoch, max_steps - epoch_start_step)
    return tqdm(
        total=epoch_total,
        initial=step - epoch_start_step,
        desc=f"epoch {epoch + 1}/{total_epochs}",
        unit="step",
        disable=disable,
        leave=True,
    )


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
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="이만큼의 micro-batch를 누적한 뒤 한 번 optimizer.step() — batch-size를 줄이고 "
        "이 값을 늘리면 유효 batch size는 유지하면서 GPU 메모리 사용량을 줄일 수 있다",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="CUDA에서 기본으로 켜지는 mixed precision(bf16/fp16)을 끄고 fp32로 학습",
    )
    parser.add_argument(
        "--grad-checkpointing",
        action="store_true",
        help="블록마다 forward를 다시 계산해 활성화 메모리를 줄인다 (속도는 느려짐, block_size가 "
        "클 때 OOM 방지용)",
    )
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
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="epoch 단위로 리셋되는 tqdm 진행바를 끈다 (로그 파일로 출력을 리다이렉트할 때 유용)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    paths = prepare_dataset(args.dataset, args.cache_dir)
    steps_per_epoch = max(1, num_tokens(paths["train"]) // (args.batch_size * args.block_size))
    epochs_covered = args.max_steps / steps_per_epoch
    print(
        f"1 epoch ≈ {steps_per_epoch} steps (train tokens // (batch_size*block_size)) — "
        f"--max-steps {args.max_steps}은 약 {epochs_covered:.2f} epoch에 해당"
    )
    if epochs_covered < 1:
        print(
            "주의: max-steps가 1 epoch보다 작아 학습 중 학습 데이터 전체를 한 번도 다 보지 않습니다. "
            "WikiText-103처럼 큰 코퍼스에서는 흔한 일이며(LLM 사전학습은 보통 여러 epoch을 돌리지 "
            "않습니다), 이 실험의 목적(softmax vs linear 속도·정확도 비교)에는 문제가 없습니다. "
            "전체 데이터를 여러 번 보고 싶다면 --max-steps를 늘리거나(steps_per_epoch의 배수), "
            "더 작은 데이터셋(tiny_shakespeare)으로 바꾸세요."
        )

    config = GPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        vocab_size=50257,
        attention_type=args.attention_type,
    )
    model = GPT(config).to(device)
    model.gradient_checkpointing = args.grad_checkpointing
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.warmup_steps, args.max_steps, args.min_lr_ratio)
    )

    use_amp = device == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp and amp_dtype == torch.float16)
    if use_amp:
        print(f"mixed precision: {amp_dtype}")
    if args.grad_checkpointing:
        print("gradient checkpointing: on")

    ckpt_path = Path(args.ckpt_dir) / args.attention_type / "latest.pt"
    start_step = 0
    if args.resume and ckpt_path.exists():
        start_step = load_checkpoint(ckpt_path, model, optimizer, device, config) + 1
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
    pbar_epoch = start_step // steps_per_epoch
    pbar = epoch_progress_bar(start_step, steps_per_epoch, args.max_steps, disable=args.no_progress_bar)
    try:
        for step in range(start_step, args.max_steps):
            epoch = step // steps_per_epoch
            if epoch != pbar_epoch:
                pbar.close()
                pbar = epoch_progress_bar(step, steps_per_epoch, args.max_steps, disable=args.no_progress_bar)
                pbar_epoch = epoch

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _ in range(args.grad_accum_steps):
                x, y = get_batch(paths["train"], args.batch_size, args.block_size, device)
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                    _, micro_loss = model(x, targets=y)
                micro_loss = micro_loss / args.grad_accum_steps
                scaler.scale(micro_loss).backward()
                loss_accum += micro_loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss_accum:.4f}")

            is_last_step = step == args.max_steps - 1
            is_epoch_boundary = args.save_every_epoch and (step + 1) % steps_per_epoch == 0

            if step % args.eval_interval == 0 or is_last_step:
                val_loss = estimate_val_loss(
                    model, paths["val"], args.batch_size, args.block_size, device, args.eval_iters,
                    use_amp=use_amp, amp_dtype=amp_dtype,
                )
                elapsed = time.perf_counter() - t0
                val_ppl = math.exp(val_loss)
                pbar.set_postfix(loss=f"{loss_accum:.4f}", val_loss=f"{val_loss:.4f}", val_ppl=f"{val_ppl:.1f}")
                tqdm.write(
                    f"step {step} (epoch {epoch}): train_loss={loss_accum:.4f} val_loss={val_loss:.4f} "
                    f"val_ppl={val_ppl:.2f} elapsed={elapsed:.1f}s"
                )
                if log_file:
                    log_file.write(
                        f"{step},{args.attention_type},{loss_accum:.6f},{val_loss:.6f},"
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
                tqdm.write(f"saved epoch {epoch_num} checkpoint -> {epoch_ckpt_path}")
    finally:
        pbar.close()

    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
