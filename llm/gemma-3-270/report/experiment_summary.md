# 🚀 Gemma-3-270M LoRA 파인튜닝 종합 정리

`llm/gemma-3-270` 폴더에서 진행된 **일정 추출기(Schedule Extractor)** LoRA 파인튜닝 실험을 시작부터 현재까지 시간 순으로 정리한 문서입니다. 각 실험 리포트(`report/report_step_1.md`, `report_step_2.md`, `train_vaild_1.md` ~ `train_valid_5.md`, `gemma_schedule_extractor_v2/train_record.md`)와 소스 코드를 근거로 작성했습니다.

---

## 1. 프로젝트 목적

메신저/알림톡 텍스트에서 **일정 정보(날짜, 시간, 장소, 참석자)** 를 JSON으로 추출하는 On-Device SLM을 만드는 것이 목표입니다.

- **Base Model**: `google/gemma-3-270m-it` (2.7억 파라미터, 경량 instruction-tuned 모델)
- **Fine-tuning 방식**: LoRA (PEFT) + TRL `SFTTrainer`
- **학습 환경**: 1~4차는 로컬 CPU, **5차부터는 로컬 GPU**(NVIDIA GeForce GTX 1650 Super, 4GB VRAM, bf16)
- **핵심 라이브러리**: TRL 0.28.0 / Transformers 4.57.5 / PyTorch 2.9.1 / Datasets 4.5.0

## 2. 파이프라인 구성 파일

| 파일 | 역할 |
| :--- | :--- |
| `utils.py` | `snapshot_download`로 `gemma-3-270m-it` 베이스 모델을 로컬에 다운로드 |
| `create_data_from_gemini.py` | Gemini API로 일상 대화체 일정 데이터 생성 (날짜를 **직접 계산**하여 `date`/`time`로 저장) → `schedule_dataset.jsonl` |
| `create_data_without_calc_from_gemini.py` | Gemini API로 일정 데이터 생성, 단 날짜/시간을 계산하지 않고 **원문 표현 그대로**(`date_text`/`time_text`) 추출 → `schedule_dataset_v2.jsonl` |
| `create_data_notice_from_gemini.py` | 알림톡/공지/예약문자 스타일(기간 표현 포함)의 데이터를 v2 스키마로 추가 생성 → `schedule_dataset_v2.jsonl`에 이어붙임 |
| `train.py` | LoRA 설정 및 `SFTTrainer`로 파인튜닝 실행(CUDA 자동 감지, GPU면 bf16), LoRA 가중치 저장 |
| `monitor_train.py` | `train.py`가 남기는 `metrics.jsonl`을 tail하며 진행률/loss/accuracy를 터미널에 실시간 시각화 (5차부터 도입) |
| `valid_model.py` / `inference_valid.ipynb` | 학습된 LoRA 가중치를 베이스 모델에 얹어(`PeftModel`) 검증. 비대화형 실행 시 고정 회귀 테스트 케이스 자동 수행 |
| `merge_lora.py` | LoRA 가중치를 베이스 모델에 병합(`merge_and_unload`)하여 단일 배포 모델 생성 |
| `inference_model.py` / `inference_test.ipynb` | 파인튜닝 이전 베이스 모델 자체의 생성 능력 확인용 |

## 3. 데이터셋 스키마 변천

| 버전 | 파일 | 샘플 수(현재 기준) | 스키마 | 특징 |
| :--- | :--- | :--- | :--- | :--- |
| v1 | `schedule_dataset.jsonl` | 40 | `date`(YYYY-MM-DD), `time`(HH:MM), `location`, `attendees` | 모델이 "내일", "다음 주 수요일" 같은 상대 날짜를 **직접 계산**해서 절대 날짜로 출력해야 함 |
| v2 | `schedule_dataset_v2.jsonl` | 1,000 | `date_text`, `time_text`, `location`, `attendees` | 계산 없이 **원문 표현을 그대로 추출**하도록 스키마 변경 (연산 부담 제거 목적) + 알림톡/공지 스타일 데이터 및 기간(Range) 표현 추가 |
| v3 | `schedule_dataset_v3.jsonl` | 1,200 | v2와 동일 스키마 | v2에 알림톡류 데이터를 더 보강한 확장판. **5차 실험에서 학습 완료** |

> v1 → v2 전환 이유: 270M급 초경량 모델은 "상대 날짜 → 절대 날짜 계산" 같은 추론을 안정적으로 수행하지 못해, 스키마 자체를 "원문 그대로 추출"로 단순화함 (`train_valid_4.md` 상단 메모: *"모델이 수학 계산을 하지 않고 순수 추출만 하도록"*).

## 4. 실험 타임라인 및 수치 결과

### 1차 실험 — v1 스키마, 소규모 데이터 (`gemma_schedule_extractor/`)

| 항목 | 값 |
| :--- | :--- |
| 데이터셋 | 40 samples |
| Epochs | 3 |
| Learning Rate | 8e-05 |
| 최종 Train Loss | **3.578** |
| 최종 Mean Token Accuracy | **52.4%** |
| Runtime | 661초 (약 11분) |

**검증 결과**: JSON이 아니라 평서문("네, 모레 저녁에 판교 카카오 본사에서 미팅 있음입니다.")을 출력 — instruction following 실패.

### 2차 실험 — v1 스키마, 데이터/Epoch 확대 (`gemma_schedule_extractor/`)

| 항목 | 값 |
| :--- | :--- |
| 데이터셋 | 240 samples (1차 대비 +140%) |
| Epochs | 10 (1차 대비 3.3배) |
| Learning Rate | 시작 1.64e-4 |
| Loss 추이 | 3.681 (2 epoch) → **1.7516** (10 epoch, -52.4%) |
| Token Accuracy 추이 | 51.55% (2 epoch) → **69.36%** (10 epoch, +17.8%p) |
| Runtime | 1,649초 (약 27분) |

**검증 결과**: 두 테스트 입력 모두 `null` 반환 — JSON 형식은 안정화됐지만 EOS 조기 트리거 또는 파싱 실패로 추정되는 오작동 발생.

### 3차 실험 — v1 스키마, 대규모 확장 (1,000 samples / 20 epochs)

| 항목 | 값 |
| :--- | :--- |
| 데이터셋 | 1,000 samples |
| Epochs | 20 |
| Learning Rate | 시작 1.82e-4 |
| 최종 Train Loss | **0.892** |
| 최종 Mean Token Accuracy | **91.8%** |
| Runtime | 3,214초 (약 53분) |

**검증 결과**: 처음으로 완전한 JSON 정상 출력 확인.
```
{"date": "2026-03-04", "time": "19:00", "location": "판교 본사", "attendees": null}
{"date": "2026-02-25", "time": "14:00", "location": "강남역", "attendees": ["영희"]}
```

### 4차 실험 — v2 스키마 전환, 1,000 samples / 20 epochs (`gemma_schedule_extractor_v2/`)

| 항목 | 값 |
| :--- | :--- |
| 데이터셋 | `schedule_dataset_v2.jsonl` 1,000 samples |
| Epochs | 20 (2,500 steps) |
| Learning Rate | 2e-4 → 0 (선형 감쇠) |
| 최종 Train Loss | **0.239** |
| 최종 Mean Token Accuracy | **96.5%** |
| Runtime | 74,339초 (약 20시간 39분 — 이전 실험 대비 환경/부하 영향으로 대폭 증가) |

**검증 결과 (2건 모두 확인)**:
- `valid_model.py` 실행 로그: `{"date_no_text": "모레", "time_no_text": "저녁", "location": "판교 카카오 본사", "attendees": null}` — 값은 정확하지만 **키 이름이 학습 스키마(`date_text`)와 다르게 환각**(`date_no_text`) 생성됨.
- `inference_valid.ipynb` 실행 결과: `{"모레", "time_text": "저녁", ...}` — **키 이름이 아예 누락된 JSON 파싱 불가 형태**로 출력됨.
- 다만 알림톡(가스 자가검침 안내문) 입력에 대해서는 `{null, "time_text": null, "location": null, "attendees": null}`로 **일정 없음을 올바르게 판단**하는 일반화 능력은 확인됨.

> 종합: Loss/Accuracy 지표는 4차 실험이 가장 우수하지만, 키 이름 환각·누락 등 **JSON 구조적 안정성 문제가 여전히 남아있음**.

### 5차 실험 — v3 스키마, 최초 GPU 학습 (`gemma_schedule_extractor_v3/`)

| 항목 | 값 |
| :--- | :--- |
| 데이터셋 | `schedule_dataset_v3.jsonl` 1,200 samples |
| Epochs | 20 (3,000 steps) |
| GPU | NVIDIA GeForce GTX 1650 Super (4GB, 데스크톱과 VRAM 공유) |
| 정밀도 | bf16 (fp16은 NaN 발생으로 폐기, 아래 참고) |
| Batch | `batch_size=1`, `gradient_accumulation_steps=8` (유효 배치 8), `gradient_checkpointing=True` |
| 최종 Loss | **0.1418** (최저 0.1249) |
| 최종 Mean Token Accuracy | **96.1%** (최고 96.6%) |
| Runtime | 4,891초 (**약 1시간 22분** — 4차 CPU 대비 약 1/15) |

**시행착오**:
1. VRAM 부족(OOM) — 6GB 카드 기준으로 잡았던 `batch_size=8`이 4GB 카드+데스크톱 GPU 사용량과 충돌. `batch_size=1`, `gradient_checkpointing`, 실측 기반 `max_length` 축소(256)로 해결.
2. **fp16 NaN 붕괴** — 모델을 fp16으로 로드한 상태에서 Trainer의 fp16 AMP까지 중복으로 걸어 `grad_norm: NaN`, `loss`/`accuracy` 0으로 붕괴. bf16으로 전환하고 AMP 중복 설정을 제거해 해결. 부수 효과로 dtype 캐스팅 오버헤드가 사라져 속도도 **4.0s/it → 1.6s/it**로 개선됨.
3. 표준출력으로는 `logging_steps` 지표가 안 찍히는 문제를 발견해 `JsonlMetricsCallback`으로 `metrics.jsonl`에 직접 기록하도록 변경, `monitor_train.py`로 실시간 시각화.

**검증 결과 (회귀 비교용 고정 케이스 3종 전부 통과)**:
```
{"date_text": "모레", "time_text": "저녁", "location": "판교 카카오 본사", "attendees": null}
{"date_text": "내일", "time_text": "오후 3시", "location": "강남역", "attendees": ["영희"]}
{"date_text": "03월 05일 부터 03월 10일 까지", "time_text": null, "location": null, "attendees": null}
```
4차 실험의 **키 이름 누락/환각 문제가 전부 해결**됨. `merge_lora.py`로 병합한 단일 모델도 `gemma_schedule_extractor_v3/merged_model`에 생성 완료.

> 상세 내용은 [`report/train_valid_5.md`](train_valid_5.md) 참고.

## 5. 학습 하이퍼파라미터 (LoRA / SFT 공통 설정, `train.py` 기준 · 5차/GPU 버전)

```python
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 모델 로드: device_map="cuda", torch_dtype=torch.bfloat16 (CUDA 없으면 cpu/float32로 자동 폴백)

training_args = SFTConfig(
    max_length=256,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```

## 6. 트러블슈팅 이력

| 실험 | 증상 | 추정 원인 |
| :--- | :--- | :--- |
| 1차 | JSON 대신 평서문 출력 | 데이터/Epoch 부족으로 instruction following 미학습 |
| 2차 | 항상 `null` 반환 | Overfitting 가능성, EOS 토큰 조기 트리거, 후처리 파싱 실패 가능성 |
| 4차 | 키 이름 환각/누락으로 JSON 파싱 불가 | v2 데이터셋/설정만으로는 구조적 출력 안정성이 부족했던 것으로 추정 — 정량 지표(Loss/Accuracy) 개선만으로는 완전히 해결되지 않음. **5차(v3 데이터셋 + bf16)에서 해결됨** |
| 5차 | GPU 전환 초기 VRAM OOM | 6GB 카드 기준 배치 설정이 실제 4GB 카드+데스크톱 GPU 사용량과 충돌 — batch_size/gradient_checkpointing/max_length 조정으로 해결 |
| 5차 | fp16에서 `grad_norm: NaN` 붕괴 | Gemma의 fp16 표현범위 초과 이슈 + fp16 가중치 위에 AMP 중복 적용 — bf16 전환으로 해결 (속도도 개선되는 부수 효과) |

## 7. 다음 단계 제안

1. `gemma_schedule_extractor_v3/merged_model`(병합 완료된 단일 모델)의 실제 배포/추론 테스트.
2. 알림톡류 데이터를 더 다양화하거나, v3 이후 스키마를 개선할 필요가 있는지 추가 실사용 케이스로 검증.
3. 다음 학습부터는 5차에서 확정된 구성(bf16 + `gradient_checkpointing` + `JsonlMetricsCallback` 실시간 모니터링)을 기본값으로 사용.
