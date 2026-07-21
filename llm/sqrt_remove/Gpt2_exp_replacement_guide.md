# GPT-2 구조에서 지수(Softmax) 연산 대체 실험 가이드

> 목적: GPT-2 규모의 Transformer에서 Softmax Attention(지수 연산 포함)을
> 하드웨어 친화적 연산(Linear Attention 등)으로 대체했을 때
> **정확도 / 학습 속도 / 추론 속도**가 어떻게 달라지는지 비교 실험한다.

---

## 0. 배경 및 원리

### 0.1 왜 지수 연산이 문제인가

Softmax Attention의 핵심 수식:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

`softmax(x_i) = e^{x_i} / Σ e^{x_j}` 안에 지수함수 `e^x`가 들어있습니다.
CPU/NPU는 덧셈·곱셈(MAC 연산)에는 최적화되어 있지만, `e^x`, `sqrt`, 나눗셈 같은
초월함수는 룩업테이블이나 반복 근사(Newton-Raphson 등)로 계산되어 상대적으로 느립니다.

### 0.2 대체 전략의 원리

| 전략 | 핵심 아이디어 | 대표 연구 |
|---|---|---|
| Linear Attention (커널 근사) | `softmax(QK^T)`를 `φ(Q)φ(K)^T` 형태의 커널 함수로 근사 → 곱셈 순서를 바꿔 O(N²)→O(N) | GLA, RetNet |
| Softmax-free (L1-norm) | Softmax 대신 단순 L1 정규화 사용 | SimA |
| 정수 다항식 근사 | GELU/LayerNorm/Softmax를 정수 다항식으로 치환 | i-BERT |

이번 실험은 **가장 단순하고 재현이 쉬운 "Linear Attention (elu(x)+1 커널)"**을
1차 대체 대상으로 삼는다. 필요 시 SimA(L1-norm) 버전도 추가 비교한다.

### 0.3 검증 가설

- H1: Linear Attention은 학습/추론 속도(특히 긴 시퀀스에서)가 더 빠르다.
- H2: Linear Attention은 정확도(perplexity)가 baseline보다 다소 떨어진다.
- H3: 레이어 수/시퀀스 길이가 늘어날수록 두 방식의 차이가 더 벌어진다.

---

## 1. 실험 환경

### 1.1 로컬 (개발·디버깅 전용)

- 로컬 GPU(GTX 1650 Super 등, VRAM 4GB급)는 **코드 개발과 단위 테스트(TDD) 전용**으로 사용한다.
- Tensor Core 미지원 GPU는 두 attention 방식의 실제 하드웨어 이득 차이가 흐려지므로
  **정식 측정에는 사용하지 않는다.**

### 1.2 Colab / Kaggle (본 실험용)

| 항목 | Colab 무료 | Kaggle 무료 |
|---|---|---|
| GPU | T4 (16GB, Tensor Core O) | T4 / P100 (16GB, Tensor Core O) |
| 세션 제한 | ~12시간, 유휴 시 끊김 | 주당 30시간 |
| 용도 | 짧은 실험 반복 | 긴 학습 1회 실행 |

**체크포인트 저장은 필수**다. Colab은 Google Drive 마운트, Kaggle은 Kaggle Dataset/Output에
주기적으로 저장하도록 코드에 반영한다.

---

## 2. 데이터셋

| 단계 | 데이터셋 | 용도 |
|---|---|---|
| 1단계 (파이프라인 검증) | TinyShakespeare (~1MB) | attention 스위치 로직, loss 감소 여부 빠르게 확인 |
| 2단계 (본 실험) | WikiText-103 (~500MB) | perplexity 비교, 기존 논문과 비교 가능한 표준 벤치마크 |
| 3단계 (선택, 정성 평가) | TinyStories | 생성 품질을 눈으로 비교 |

```python
from datasets import load_dataset

# 1단계 — 주의: 원본 "tiny_shakespeare"와 "wikitext"는 스크립트 기반 로더라
# 최신 datasets(>=4)에서 `RuntimeError: Dataset scripts are no longer supported`로 실패한다.
# gpt2/data.py는 스크립트 없는(parquet) 미러를 사용하도록 이미 구현되어 있다.
tiny_shakespeare = load_dataset("winglian/tiny-shakespeare")  # train/test, "text" 컬럼

# 2단계
wikitext103 = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")  # 원 소유자가 올린 미러
```

> 실제 로딩/토큰화/캐싱 로직은 `gpt2/data.py`의 `prepare_dataset()`을 사용한다
> (위 코드는 참고용 스니펫).

---

## 3. 모델 스케일 권장값

| 항목 | 로컬 디버깅용 | Colab/Kaggle 본 실험용 |
|---|---|---|
| n_layer | 4 | 12 (원본 GPT-2 small) |
| n_head | 4 | 12 |
| n_embd | 128 | 768 |
| block_size | 128 | 256 / 512 / 1024 (단계적 비교) |
| batch_size | 8 | 16~32 (gradient accumulation 병행) |

> block_size를 여러 단계로 바꿔가며 실험하는 이유: 긴 시퀀스일수록 Linear Attention의
> 이론적 이점(O(N) vs O(N²))이 실측에서도 드러나는지 확인하기 위함 (가설 H3).

---

## 4. 비교 지표

| 구분 | 지표 | 측정 방법 |
|---|---|---|
| 정확도 | Validation Perplexity | 고정 검증셋에서 loss → `exp(loss)` |
| 학습 속도 | steps/sec, epoch당 wall-clock time | `time.perf_counter()`로 step 구간 측정 |
| 추론 속도 | tokens/sec (batch=1, batch=N 각각) | 고정 길이 생성 시간 측정, warmup 후 평균 |
| 공정성 확보 | 동일 total FLOPs / 파라미터 수 | 레이어 수·차원으로 파라미터 수 맞추기 |

**주의**: 학습/추론 속도는 같은 세션·같은 GPU 배정 내에서 baseline과 variant를
연달아 측정해야 한다(클라우드 GPU 배정 편차 때문).

---

## 5. 구현 절차 (TDD 기반)

TDD 원칙에 따라 **테스트 → 실패 확인(Red) → 최소 구현(Green) → 리팩터링** 순서로 진행한다.

### 5.1 디렉토리 구조 (예시)

```
project/
├── src/
│   ├── attention/
│   │   ├── softmax_attention.py      # baseline
│   │   └── linear_attention.py       # variant
│   ├── model.py                      # GPT-2 구조 (attention 모듈 스위치 가능)
│   └── config.py
├── tests/
│   ├── test_softmax_attention.py
│   ├── test_linear_attention.py
│   └── test_model_forward.py
├── scripts/
│   ├── train.py
│   └── benchmark.py
└── notebooks/
    └── colab_experiment.ipynb
```

### 5.2 1단계: Attention 모듈 단위 테스트부터 작성

```python
# tests/test_linear_attention.py
from __future__ import annotations

import torch

from src.attention.linear_attention import LinearAttention


def test_linear_attention_output_shape() -> None:
    """출력 shape이 입력 shape과 동일해야 한다."""
    batch_size: int = 2
    seq_len: int = 16
    n_embd: int = 32
    n_head: int = 4

    attn: LinearAttention = LinearAttention(n_embd=n_embd, n_head=n_head)
    x: torch.Tensor = torch.randn(batch_size, seq_len, n_embd)

    output: torch.Tensor = attn(x)

    assert output.shape == (batch_size, seq_len, n_embd)


def test_linear_attention_no_exp_or_softmax_call(monkeypatch: "pytest.MonkeyPatch") -> None:
    """구현 내부에서 torch.softmax / torch.exp를 호출하지 않아야 한다."""
    import torch.nn.functional as F

    called: dict[str, bool] = {"softmax": False, "exp": False}

    original_softmax = F.softmax
    original_exp = torch.exp

    def fake_softmax(*args: object, **kwargs: object) -> torch.Tensor:
        called["softmax"] = True
        return original_softmax(*args, **kwargs)  # type: ignore[arg-type]

    def fake_exp(*args: object, **kwargs: object) -> torch.Tensor:
        called["exp"] = True
        return original_exp(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(F, "softmax", fake_softmax)
    monkeypatch.setattr(torch, "exp", fake_exp)

    attn: LinearAttention = LinearAttention(n_embd=32, n_head=4)
    x: torch.Tensor = torch.randn(2, 16, 32)
    attn(x)

    assert called["softmax"] is False
    assert called["exp"] is False


def test_linear_attention_causal_mask_applied() -> None:
    """causal 마스크가 적용되어 미래 토큰 정보가 과거 위치에 영향을 주지 않아야 한다."""
    attn: LinearAttention = LinearAttention(n_embd=16, n_head=2)
    x: torch.Tensor = torch.randn(1, 8, 16)

    x_modified: torch.Tensor = x.clone()
    x_modified[:, -1, :] = torch.randn(1, 16)  # 마지막(미래) 토큰만 변경

    out_original: torch.Tensor = attn(x)
    out_modified: torch.Tensor = attn(x_modified)

    # 마지막 토큰을 제외한 앞부분 출력은 동일해야 함 (causal 특성)
    assert torch.allclose(out_original[:, :-1, :], out_modified[:, :-1, :], atol=1e-5)
```

이 테스트가 **먼저 실패하는 것을 확인**한 뒤(Red), `LinearAttention` 모듈을 구현해
통과시킨다(Green). Baseline인 `SoftmaxAttention`도 동일한 인터페이스 계약(shape,
causal mask)을 검증하는 테스트를 먼저 작성한다.

### 5.3 2단계: 모델 forward 통합 테스트

```python
# tests/test_model_forward.py
from __future__ import annotations

import torch

from src.config import GPTConfig
from src.model import GPT


def test_gpt_forward_with_softmax_attention() -> None:
    config: GPTConfig = GPTConfig(
        n_layer=2, n_head=2, n_embd=32, block_size=16,
        vocab_size=100, attention_type="softmax",
    )
    model: GPT = GPT(config)
    idx: torch.Tensor = torch.randint(0, 100, (2, 16))

    logits, loss = model(idx, targets=idx)

    assert logits.shape == (2, 16, 100)
    assert loss.item() > 0


def test_gpt_forward_with_linear_attention() -> None:
    config: GPTConfig = GPTConfig(
        n_layer=2, n_head=2, n_embd=32, block_size=16,
        vocab_size=100, attention_type="linear",
    )
    model: GPT = GPT(config)
    idx: torch.Tensor = torch.randint(0, 100, (2, 16))

    logits, loss = model(idx, targets=idx)

    assert logits.shape == (2, 16, 100)
    assert loss.item() > 0
```

### 5.4 3단계: 벤치마크 스크립트 (측정 자동화)

```python
# scripts/benchmark.py
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from src.config import GPTConfig
from src.model import GPT


@dataclass
class BenchmarkResult:
    attention_type: str
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

    return max_new_tokens / (end - start)
```

> 벤치마크 함수도 동일하게 **먼저 테스트를 작성**한다(예: 반환값이 `float`이고
> 0보다 큰지, `n_iters=0`일 때 예외를 던지는지 등).

---

## 6. Colab / Kaggle 실행 체크리스트

- [ ] `pip install torch datasets` (Colab/Kaggle 기본 이미지에 torch는 보통 포함)
- [ ] GPU 런타임 선택 확인 (`torch.cuda.is_available()`)
- [ ] Google Drive 마운트(Colab) 또는 Kaggle Output 경로 설정 → 체크포인트 저장 경로 지정
- [ ] WikiText-103 다운로드 후 토큰화 결과를 캐시 파일로 저장 (재다운로드 방지)
- [ ] baseline(softmax) → linear 순서로 **같은 세션 내에서 연속 실행** (GPU 배정 편차 최소화)
- [ ] 시드 3~5개로 반복 실행 (`torch.manual_seed`)
- [ ] 결과를 CSV/JSON으로 저장 → 이후 시각화

---

## 7. 결과 정리 템플릿

| attention_type | n_layer | block_size | val_ppl | train_step_time(s) | inference_tok/s |
|---|---|---|---|---|---|
| softmax | 12 | 256 | | | |
| linear | 12 | 256 | | | |
| softmax | 12 | 512 | | | |
| linear | 12 | 512 | | | |
| softmax | 12 | 1024 | | | |
| linear | 12 | 1024 | | | |

block_size(시퀀스 길이)가 늘어날수록 두 방식의 속도 차이가 벌어지는지가
가설 H3의 핵심 검증 포인트다.

---

## 8. 참고 문헌

- Yang, S. et al., *Gated Linear Attention Transformers with Hardware-Efficient Training*, ICML 2024.
- SimA: *Simple Softmax-free Attention For Vision Transformers*, OpenReview 2022.
- i-BERT: *Integer-only BERT Quantization*.
- Sun, Y. et al., *Retentive Network: A Successor to Transformer for Large Language Models*, 2023.