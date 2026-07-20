"""
train.py가 남기는 metrics.jsonl(JsonlMetricsCallback)을 실시간으로 tail하며
진행률/loss/accuracy를 터미널에 시각화한다.

사용법:
    python llm/gemma-3-270/monitor_train.py <output_dir>/metrics.jsonl
"""

import json
import sys
import time

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(
        SPARK_CHARS[min(len(SPARK_CHARS) - 1, int((v - lo) / span * (len(SPARK_CHARS) - 1)))]
        for v in values
    )


def fmt_secs(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render(history: list) -> str:
    lines = ["\033[H\033[J", "=== Gemma-3-270M LoRA 학습 모니터 (v3) ===\n"]

    if not history:
        lines.append("아직 기록된 로그가 없습니다. (첫 logging_steps 도달 대기 중...)")
        lines.append("\n(Ctrl+C로 종료, 2초 간격 갱신)")
        return "\n".join(lines)

    last = history[-1]
    step, max_steps = last.get("step", 0), last.get("max_steps", 1)
    pct = 100 * step / max_steps if max_steps else 0
    elapsed = last.get("elapsed_sec", 0)
    eta = elapsed / step * (max_steps - step) if step else 0

    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "#" * filled + "-" * (bar_len - filled)
    lines.append(f"[{bar}] {pct:5.1f}%  ({step}/{max_steps} step)")
    lines.append(f"경과: {fmt_secs(elapsed)}   예상 남은 시간: {fmt_secs(eta)}\n")

    lines.append(f"epoch          : {last.get('epoch')}")
    lines.append(f"loss           : {last.get('loss')}")
    lines.append(f"mean_token_acc : {last.get('mean_token_accuracy')}")
    lines.append(f"learning_rate  : {last.get('learning_rate')}")
    lines.append(f"grad_norm      : {last.get('grad_norm')}")

    losses = [h["loss"] for h in history if "loss" in h][-40:]
    accs = [h["mean_token_accuracy"] for h in history if "mean_token_accuracy" in h][-40:]
    lines.append("")
    lines.append(f"loss 추이 (최근 {len(losses)}개, min={min(losses):.3f} max={max(losses):.3f}):")
    lines.append(f"  {sparkline(losses)}")
    if accs:
        lines.append(f"acc  추이 (최근 {len(accs)}개, min={min(accs):.3f} max={max(accs):.3f}):")
        lines.append(f"  {sparkline(accs)}")

    lines.append("\n(Ctrl+C로 종료, 2초 간격 갱신)")
    return "\n".join(lines)


def main(jsonl_path: str) -> None:
    history = []
    pos = 0

    while True:
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                f.seek(pos)
                new_lines = f.readlines()
                pos = f.tell()
        except FileNotFoundError:
            new_lines = []

        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                pass

        sys.stdout.write(render(history))
        sys.stdout.flush()
        time.sleep(2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python monitor_train.py <metrics.jsonl 경로>")
        sys.exit(1)
    try:
        main(sys.argv[1])
    except KeyboardInterrupt:
        print("\n모니터링을 종료합니다.")
