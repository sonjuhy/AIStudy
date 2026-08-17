"""1단계 — 재작성 모델의 평문 검증.

재작성한 SmolLM 코드가 원본 HuggingFace 모델과 암호화 없이(평문) 동일한 출력을
내는지 확인한다. 이 검증 없이 HE를 적용하면, 이후 오차가 재작성 버그 때문인지
HE 근사 오차 때문인지 구분할 수 없다.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.model import SmolLM

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "SmolLM-135M")


def _load_models() -> tuple[SmolLM, AutoModelForCausalLM]:
    ours = SmolLM.from_pretrained_weights(WEIGHTS_DIR)
    ours.eval()

    reference = AutoModelForCausalLM.from_pretrained(
        WEIGHTS_DIR, dtype=torch.float32, attn_implementation="eager"
    )
    reference.eval()

    return ours, reference


def test_reimplemented_model_matches_huggingface_logits() -> None:
    ours, reference = _load_models()

    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    input_ids = tokenizer(
        "Homomorphic encryption allows computation on", return_tensors="pt"
    ).input_ids

    with torch.no_grad():
        ours_logits = ours(input_ids)
        reference_logits = reference(input_ids).logits

    max_abs_diff = (ours_logits - reference_logits).abs().max().item()
    print(f"max abs diff: {max_abs_diff:.3e}")

    assert ours_logits.shape == reference_logits.shape
    assert torch.allclose(ours_logits, reference_logits, atol=1e-3, rtol=1e-3), (
        f"logits diverge from reference (max abs diff={max_abs_diff:.3e}) — "
        "재작성 코드에 버그가 있을 가능성이 높다"
    )


def test_reimplemented_model_matches_huggingface_greedy_tokens() -> None:
    ours, reference = _load_models()

    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids

    with torch.no_grad():
        ours_next_token = ours(input_ids)[0, -1].argmax().item()
        reference_next_token = reference(input_ids).logits[0, -1].argmax().item()

    assert ours_next_token == reference_next_token


if __name__ == "__main__":
    ours, reference = _load_models()

    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    input_ids = tokenizer(
        "Homomorphic encryption allows computation on", return_tensors="pt"
    ).input_ids

    with torch.no_grad():
        ours_logits = ours(input_ids)
        reference_logits = reference(input_ids).logits

    diff = (ours_logits - reference_logits).abs()
    print(f"max abs diff:  {diff.max().item():.3e}")
    print(f"mean abs diff: {diff.mean().item():.3e}")
    print(f"ours argmax:      {ours_logits[0, -1].argmax().item()}")
    print(f"reference argmax: {reference_logits[0, -1].argmax().item()}")

# max abs diff:  0.000e+00
# mean abs diff: 0.000e+00
# ours argmax:      253
# reference argmax: 253
