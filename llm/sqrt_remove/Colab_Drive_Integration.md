# Colab + Google Drive 연동 가이드

> `gpt2/train.py`를 Colab 무료 GPU에서 돌리면서, Google Drive에 데이터 캐시·체크포인트·로그를
> 영구 저장하고 세션이 끊겨도(~12시간, 유휴 시) 이어서 학습하는 방법을 정리한다.
> 관련 문서: `Gpt2_exp_replacement_guide.md`, `Kaggle_Colab_Prep_Checklist.md`
>
> **바로 실행하려면 `notebooks/colab_experiment.ipynb`를 Colab에 업로드**하면 아래 셀들이
> 이미 순서대로 구성되어 있다. 이 문서는 그 안의 각 단계를 설명하는 참고용이다.

---

## 0. 핵심 아이디어

Colab 세션은 종료되면 로컬 디스크(`/content/`)가 초기화된다. 반면 `/content/drive/MyDrive/`는
Google Drive에 마운트된 경로라 세션이 끊겨도 유지된다. 따라서 `train.py`의 세 경로 인자를
전부 Drive 하위로 지정하면, 코드 변경 없이 그대로 영속화된다.

| 인자 | 역할 | Drive에 둬야 하는 이유 |
|---|---|---|
| `--cache-dir` | 토큰화된 `train.bin`/`val.bin` 캐시 | 세션마다 데이터셋 재다운로드·재토큰화 방지 |
| `--ckpt-dir` | 모델/옵티마이저 체크포인트 | 세션 끊김 후 `--resume`으로 이어서 학습 |
| `--log-path` | 스텝별 CSV 로그 | 세션이 여러 번 끊겨도 학습 기록이 하나로 이어짐 |

> **주의**: `--ckpt-dir`은 `{ckpt-dir}/{attention-type}/latest.pt`로 저장되므로, **모델
> 규모(n_layer/n_embd 등)가 다른 실행끼리 같은 --ckpt-dir을 공유하면 안 된다.** 예를 들어
> 가이드 1단계(파이프라인 검증, 소형 모델)와 2단계(본 실험, GPT-2 small)를 같은 디렉터리에
> 저장하면, 2단계에서 `--resume`이 1단계의 작은 체크포인트를 불러오려다 `size mismatch`
> 에러로 실패한다. 단계별로 `checkpoints/pipeline_check/`, `checkpoints/main/`처럼 하위
> 디렉터리를 분리하자 (아래 예시와 `notebooks/colab_experiment.ipynb`에 이미 반영됨).
> 설정이 다른 체크포인트를 실수로 불러오면 `train.py`가 shape 에러 대신 어떤 config가
> 충돌했는지 알려주는 명확한 에러를 낸다.

---

## 1. Drive 마운트 및 코드 준비

```python
# Cell 1: GPU 확인
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

```python
# Cell 2: Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 3: 코드 가져오기 (git clone 권장 — 파일 업로드보다 세션마다 반복하기 편함)
!git clone <레포 URL> repo
%cd repo/llm/sqrt_remove
!pip install -q -r requirements.txt
```

> 레포를 매번 clone하기 번거로우면 `repo/` 자체를 Drive에 clone해두고
> `%cd /content/drive/MyDrive/repo/llm/sqrt_remove`로 바로 이동해도 된다.
> 이 경우 코드 수정분도 Drive에 남는다는 장점이 있다.

---

## 2. 실행 경로를 전부 Drive로 지정

```python
# Cell 4: 경로 상수 (한 곳에서만 정의)
DRIVE_ROOT = "/content/drive/MyDrive/gpt2_exp"
CACHE_DIR = f"{DRIVE_ROOT}/data_cache"
CKPT_DIR = f"{DRIVE_ROOT}/checkpoints"
LOG_DIR = f"{DRIVE_ROOT}/logs"
```

```python
# Cell 5: 학습 실행 (baseline: softmax)
!python -m gpt2.train \
  --attention-type softmax \
  --dataset tiny_shakespeare \
  --n-layer 12 --n-head 12 --n-embd 768 --block-size 512 \
  --batch-size 16 --max-steps 5000 \
  --eval-interval 200 --ckpt-interval 200 \
  --save-every-epoch \
  --cache-dir "{CACHE_DIR}" \
  --ckpt-dir "{CKPT_DIR}" \
  --log-path "{LOG_DIR}/softmax.csv" \
  --resume
```

- `--save-every-epoch`: 1 epoch(≈ train.bin 토큰 수 // (batch_size×block_size) 스텝)마다
  `checkpoints/softmax/epoch_N.pt`를 덮어쓰지 않고 누적 저장한다.
- `--resume`: `checkpoints/softmax/latest.pt`가 있으면 그 지점(step)부터 자동으로 이어서 학습한다.
  **매번 이 플래그를 켜두면, 세션이 언제 끊기든 같은 셀을 다시 실행하는 것만으로 재개된다.**

같은 세션에서 이어서 linear attention도 연달아 실행한다 (GPU 배정 편차 최소화, 가이드 4장 참고):

```python
# Cell 6: 학습 실행 (variant: linear)
!python -m gpt2.train \
  --attention-type linear \
  --dataset tiny_shakespeare \
  --n-layer 12 --n-head 12 --n-embd 768 --block-size 512 \
  --batch-size 16 --max-steps 5000 \
  --eval-interval 200 --ckpt-interval 200 \
  --save-every-epoch \
  --cache-dir "{CACHE_DIR}" \
  --ckpt-dir "{CKPT_DIR}" \
  --log-path "{LOG_DIR}/linear.csv" \
  --resume
```

---

## 3. 세션이 끊긴 뒤 재개하는 법

1. Colab 런타임이 끊기면 상단 메뉴에서 런타임을 다시 연결한다.
2. **Cell 1 → 2 → 3 → 4**를 순서대로 다시 실행한다 (GPU 재확인, Drive 재마운트, 코드/의존성
   재설치 — `/content/`는 초기화되지만 Drive 내용은 그대로 남아있다).
3. **Cell 5 (또는 6)을 그대로 다시 실행**한다. `--resume` 플래그 덕분에 `latest.pt`에 저장된
   step부터 자동으로 이어서 학습한다.

체크포인트가 실제로 쌓였는지 확인:

```python
!ls -la "{CKPT_DIR}/softmax"
# latest.pt, epoch_1.pt, epoch_2.pt, ...
```

---

## 4. 벤치마크도 동일하게 Drive 경로 사용

```python
# Cell 7: 벤치마크 (softmax vs linear, block_size별)
!python -m gpt2.benchmark \
  --dataset tiny_shakespeare \
  --n-layer 12 --n-head 12 --n-embd 768 \
  --block-sizes 256 512 1024 \
  --seeds 1337 42 7 \
  --batch-size 16 \
  --cache-dir "{CACHE_DIR}" \
  --output "{DRIVE_ROOT}/benchmark_results.json"
```

결과 JSON도 Drive에 저장되므로, 세션이 끊겨도 지금까지의 측정값은 안전하게 남는다.

---

## 5. 결과 집계 (가이드 7장 표 자동 생성)

`gpt2/aggregate_results.py`가 여러 seed의 벤치마크 결과를 평균 내서, 가이드 7장의 결과
정리 템플릿과 softmax/linear 속도·정확도 비교 표(가설 H1·H3 검증용)를 markdown으로 만든다.

```python
# Cell 8: 결과 집계
!python -m gpt2.aggregate_results \
  --input "{DRIVE_ROOT}/benchmark_results.json" \
  --output "{DRIVE_ROOT}/results_table.md"
```

```python
# Cell 9: 노트북에서 바로 표 보기
from IPython.display import Markdown, display
with open(f"{DRIVE_ROOT}/results_table.md") as f:
    display(Markdown(f.read()))
```

`--input`에 파일을 여러 개(공백으로 구분) 주면 서로 다른 세션에서 저장한
`benchmark_results.json`들을 하나로 합쳐서 집계할 수 있다.

---

## 6. 주의사항

- **Drive I/O는 로컬 디스크보다 느리다.** `--ckpt-interval`을 너무 짧게(예: 10 스텝마다) 잡으면
  저장 자체가 병목이 될 수 있다. 200~500 스텝 또는 epoch 단위 저장을 기본으로 권장.
- **Drive 용량**: GPT-2 small(12층, 768차원) 체크포인트는 옵티마이저 상태(AdamW, 파라미터당 2개
  모멘텀) 포함 시 파라미터 파일 크기의 약 3배. epoch마다 누적 저장하면 용량이 빠르게 늘어나므로,
  주기적으로 오래된 `epoch_N.pt`를 정리하거나 `--save-every-epoch` 없이 `latest.pt`(덮어쓰기)만
  쓰는 것도 고려한다.
- **동일 세션 내 baseline→variant 연속 실행**은 GPU 배정 편차를 줄이기 위한 것이므로, Cell 5와 6은
  가능하면 런타임을 재시작하지 않고 이어서 실행한다.
