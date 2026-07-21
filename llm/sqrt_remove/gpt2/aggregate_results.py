from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

GROUP_KEYS = ("attention_type", "n_layer", "block_size")
METRIC_KEYS = ("train_step_time_sec", "inference_tokens_per_sec", "val_perplexity")


def load_results(paths: list[Path]) -> list[dict]:
    """benchmark.py --output으로 만든 JSON 파일(들)을 하나의 리스트로 합친다."""
    results: list[dict] = []
    for path in paths:
        with Path(path).open() as f:
            results.extend(json.load(f))
    return results


def aggregate(results: list[dict]) -> list[dict]:
    """(attention_type, n_layer, block_size)로 묶어 seed들에 대한 평균/표준편차를 계산한다."""
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        key = tuple(r[k] for k in GROUP_KEYS)
        groups.setdefault(key, []).append(r)

    aggregated: list[dict] = []
    for key, rows in groups.items():
        entry: dict = dict(zip(GROUP_KEYS, key))
        entry["n_seeds"] = len(rows)
        for metric in METRIC_KEYS:
            values = [row[metric] for row in rows]
            entry[f"{metric}_mean"] = statistics.mean(values)
            entry[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        aggregated.append(entry)

    aggregated.sort(key=lambda e: (e["n_layer"], e["block_size"], e["attention_type"]))
    return aggregated


def to_markdown_table(aggregated: list[dict]) -> str:
    """가이드 7장 '결과 정리 템플릿'과 동일한 컬럼의 markdown 표를 만든다."""
    header = (
        "| attention_type | n_layer | block_size | val_ppl | train_step_time(s) | "
        "inference_tok/s | n_seeds |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = [
        f"| {e['attention_type']} | {e['n_layer']} | {e['block_size']} | "
        f"{e['val_perplexity_mean']:.2f} ± {e['val_perplexity_std']:.2f} | "
        f"{e['train_step_time_sec_mean']:.4f} ± {e['train_step_time_sec_std']:.4f} | "
        f"{e['inference_tokens_per_sec_mean']:.1f} ± {e['inference_tokens_per_sec_std']:.1f} | "
        f"{e['n_seeds']} |"
        for e in aggregated
    ]
    return header + "\n".join(rows) + "\n"


def compute_speedup_table(aggregated: list[dict]) -> str:
    """block_size별 softmax 대비 linear 속도/정확도 차이를 계산한다 (가설 H1·H3 검증용).

    train_step_speedup, inference_speedup 모두 1보다 크면 linear가 더 빠르다는 뜻.
    """
    by_block: dict[int, dict[str, dict]] = {}
    for e in aggregated:
        by_block.setdefault(e["block_size"], {})[e["attention_type"]] = e

    header = (
        "| block_size | train_step_speedup (softmax/linear) | "
        "inference_speedup (linear/softmax) | val_ppl_gap (linear - softmax) |\n"
        "|---|---|---|---|\n"
    )
    rows = []
    for block_size in sorted(by_block):
        pair = by_block[block_size]
        if "softmax" not in pair or "linear" not in pair:
            continue
        softmax, linear = pair["softmax"], pair["linear"]
        train_speedup = softmax["train_step_time_sec_mean"] / linear["train_step_time_sec_mean"]
        infer_speedup = (
            linear["inference_tokens_per_sec_mean"] / softmax["inference_tokens_per_sec_mean"]
        )
        ppl_gap = linear["val_perplexity_mean"] - softmax["val_perplexity_mean"]
        rows.append(f"| {block_size} | {train_speedup:.2f}x | {infer_speedup:.2f}x | {ppl_gap:+.2f} |")

    return header + "\n".join(rows) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="benchmark.py 결과 JSON을 집계해 결과 표를 생성")
    parser.add_argument(
        "--input", nargs="+", required=True, help="benchmark.py --output으로 만든 JSON 파일(들)"
    )
    parser.add_argument("--output", default="results_table.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> str:
    args = parse_args(argv)
    results = load_results([Path(p) for p in args.input])
    aggregated = aggregate(results)

    content = (
        "# 실험 결과 정리\n\n"
        "## 가이드 7장 템플릿\n\n"
        + to_markdown_table(aggregated)
        + "\n## Softmax vs Linear 속도/정확도 비교 (H1, H3 검증)\n\n"
        + compute_speedup_table(aggregated)
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(content)
    return content


if __name__ == "__main__":
    main()
