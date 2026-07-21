from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpt2 import aggregate_results as agg


def _make_result(
    attention_type: str,
    block_size: int,
    train_step: float,
    infer_tps: float,
    val_ppl: float,
    n_layer: int = 12,
) -> dict:
    return {
        "attention_type": attention_type,
        "n_layer": n_layer,
        "block_size": block_size,
        "train_step_time_sec": train_step,
        "inference_tokens_per_sec": infer_tps,
        "val_perplexity": val_ppl,
        "extra": {"n_params": 123_000_000},
    }


def test_aggregate_averages_across_seeds() -> None:
    results = [
        _make_result("softmax", 256, 0.10, 100.0, 30.0),
        _make_result("softmax", 256, 0.12, 90.0, 32.0),
    ]

    aggregated = agg.aggregate(results)

    assert len(aggregated) == 1
    entry = aggregated[0]
    assert entry["attention_type"] == "softmax"
    assert entry["n_layer"] == 12
    assert entry["block_size"] == 256
    assert entry["n_seeds"] == 2
    assert entry["train_step_time_sec_mean"] == pytest.approx(0.11)
    assert entry["inference_tokens_per_sec_mean"] == pytest.approx(95.0)
    assert entry["val_perplexity_mean"] == pytest.approx(31.0)


def test_aggregate_keeps_different_configs_separate() -> None:
    results = [
        _make_result("softmax", 256, 0.10, 100.0, 30.0),
        _make_result("linear", 256, 0.05, 150.0, 33.0),
        _make_result("softmax", 512, 0.20, 50.0, 29.0),
    ]

    aggregated = agg.aggregate(results)

    assert len(aggregated) == 3


def test_to_markdown_table_contains_all_rows() -> None:
    results = [
        _make_result("softmax", 256, 0.10, 100.0, 30.0),
        _make_result("linear", 256, 0.05, 150.0, 33.0),
    ]
    aggregated = agg.aggregate(results)

    table = agg.to_markdown_table(aggregated)

    assert "attention_type" in table
    assert "softmax" in table
    assert "linear" in table
    assert "256" in table


def test_compute_speedup_table_reports_linear_faster() -> None:
    results = [
        _make_result("softmax", 256, 0.10, 100.0, 30.0),
        _make_result("linear", 256, 0.05, 150.0, 33.0),
    ]
    aggregated = agg.aggregate(results)

    table = agg.compute_speedup_table(aggregated)

    assert "2.00x" in table  # softmax(0.10) / linear(0.05) train speedup
    assert "1.50x" in table  # linear(150) / softmax(100) inference speedup
    assert "+3.00" in table  # ppl gap: linear(33) - softmax(30)


def test_compute_speedup_table_skips_incomplete_pairs() -> None:
    results = [_make_result("softmax", 256, 0.10, 100.0, 30.0)]
    aggregated = agg.aggregate(results)

    table = agg.compute_speedup_table(aggregated)

    data_lines = [line for line in table.splitlines() if line.startswith("| 256")]
    assert data_lines == []


def test_load_results_merges_multiple_json_files(tmp_path: Path) -> None:
    file1 = tmp_path / "a.json"
    file2 = tmp_path / "b.json"
    file1.write_text(json.dumps([_make_result("softmax", 256, 0.1, 100.0, 30.0)]))
    file2.write_text(json.dumps([_make_result("linear", 256, 0.05, 150.0, 33.0)]))

    results = agg.load_results([file1, file2])

    assert len(results) == 2


def test_main_writes_markdown_file(tmp_path: Path) -> None:
    input_path = tmp_path / "bench.json"
    input_path.write_text(
        json.dumps(
            [
                _make_result("softmax", 256, 0.10, 100.0, 30.0),
                _make_result("linear", 256, 0.05, 150.0, 33.0),
            ]
        )
    )
    output_path = tmp_path / "table.md"

    agg.main(["--input", str(input_path), "--output", str(output_path)])

    assert output_path.exists()
    content = output_path.read_text()
    assert "softmax" in content
    assert "linear" in content
