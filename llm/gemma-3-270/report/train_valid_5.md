# 🚀 Gemma-3-270M LoRA 파인튜닝 5차 실험 리포트

4차 실험까지는 전부 로컬 **CPU** 환경에서 진행됐다(4차만 20시간 39분 소요). 5차 실험은 처음으로 로컬 **GPU**(NVIDIA GeForce GTX 1650 Super, 4GB VRAM)에서 진행했고, 그 과정에서 겪은 시행착오와 최종 결과를 정리한다.

---

## 1. 실험 환경

| 항목 | 값 |
| :--- | :--- |
| 데이터셋 | `schedule_dataset_v3.jsonl` — 1,200 samples (v2 스키마 + 알림톡류 데이터 보강) |
| Epochs | 20 (3,000 steps) |
| GPU | NVIDIA GeForce GTX 1650 Super (4GB VRAM, Turing, Tensor Core 없음) — 데스크톱 UI(Chrome/VS Code)와 VRAM 공유 |
| 정밀도 | bf16 (아래 3번 참고) |
| Batch | `per_device_train_batch_size=1`, `gradient_accumulation_steps=8` (유효 배치 8) |
| max_length | 256 (v3 데이터셋 실측 최대 토큰 길이 210 확인 후 결정) |
| 기타 | `gradient_checkpointing=True`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

---

## 2. CPU → GPU 전환 과정에서 만난 문제와 해결

### 2-1. VRAM 부족 (OOM)
4GB 카드가 데스크톱 환경(Chrome/VS Code GPU 프로세스)과 VRAM을 나눠 써서, 처음 설정한 `batch_size=8`(6GB 카드 기준으로 잡았던 값)은 즉시 OOM. `batch_size`를 1까지 낮추고 `gradient_checkpointing`, `max_length` 축소(384→256, 실측 기반), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 적용해 여유 VRAM을 74MB→715MB 수준까지 확보했다.

### 2-2. fp16에서 NaN 발생 (핵심 이슈)
VRAM 문제를 해결하고 fp16으로 첫 실행했을 때, step 10부터 `grad_norm: NaN`, `mean_token_accuracy: 0.0`이 관측됐다. 원인은 두 가지가 겹친 것으로 파악:
1. **Gemma 계열의 fp16 불안정성**: 임베딩 스케일 값이 fp16 표현범위(~65504)를 넘어서기 쉬워 NaN이 잘 발생하는 것으로 알려진 문제.
2. **정밀도 설정 중복**: 모델 가중치를 이미 `torch_dtype=torch.float16`으로 로드해놓고, `SFTConfig(fp16=True)`로 Trainer의 자동 혼합정밀도(AMP)까지 이중으로 걸어놓은 상태였음.

**해결**: 모델 로드 dtype을 `bfloat16`으로 변경하고, Trainer의 `fp16`/`bf16` AMP 플래그는 켜지 않음(가중치가 이미 bf16이므로 추가 캐스팅 불필요). GTX 1650 Super는 Turing이라 bf16 텐서코어 가속은 없지만(연산이 fp32로 폴백), 지수범위가 fp32와 동일해 안정적으로 동작한다.

**부수 효과**: 예상과 달리 속도가 오히려 크게 빨라졌다. fp16 가중치 위에 AMP를 중복으로 걸었을 때 발생하던 반복적인 dtype 캐스팅 오버헤드가 사라지면서 **4.0s/it → 1.6s/it**로 개선됨.

### 2-3. 실시간 모니터링을 위한 구조 변경
`logging_steps=10`으로 찍히는 학습 지표가 표준출력(nohup 리다이렉트) 환경에서 정상적으로 남지 않는 것을 확인(로깅 레벨/버퍼링 문제로 추정). `TrainerCallback`을 커스텀 구현(`JsonlMetricsCallback`)해 `on_log`마다 `metrics.jsonl`에 한 줄씩 직접 기록하도록 변경, 이를 tail하는 `monitor_train.py`로 별도 터미널에서 실시간 진행률/loss/accuracy 스파크라인을 확인할 수 있게 했다.

---

## 3. 학습 지표 추이

| Epoch | Loss | Mean Token Accuracy | Learning Rate | Grad Norm |
| :---: | :---: | :---: | :---: | :---: |
| 2  | 0.4001 | 91.39% | 1.80e-04 | 0.828 |
| 4  | 0.3181 | 92.56% | 1.60e-04 | 0.821 |
| 6  | 0.2707 | 93.61% | 1.40e-04 | 1.038 |
| 8  | 0.2368 | 94.15% | 1.20e-04 | 0.873 |
| 10 | 0.2158 | 94.17% | 1.00e-04 | 0.918 |
| 12 | 0.1893 | 94.95% | 8.01e-05 | 1.084 |
| 14 | 0.1767 | 95.05% | 6.01e-05 | 1.218 |
| 16 | 0.1562 | 95.81% | 4.01e-05 | 1.122 |
| 18 | 0.1449 | 95.78% | 2.01e-05 | 0.933 |
| 20 | **0.1418** | **96.09%** | 6.67e-08 | 1.270 |

- 전체 300개 로그(steps 10~3000) 중 **NaN/Inf 0건** — bf16 전환 이후 학습 내내 안정적.
- 최종 `train_loss`(전체 평균): 0.2516 / 최저 스텝 loss: 0.1249 / 최고 accuracy: 96.55%
- **총 학습 시간: 4,891초 (1시간 22분)** — 4차 실험(CPU, 20시간 39분) 대비 **약 1/15**로 단축.

---

## 4. 검증 결과 (`valid_model.py`, 회귀 비교용 고정 케이스 3종)

4차 실험에서 문제였던 **JSON 키 이름 누락/환각**이 전부 해결되고, 문법적으로 완전한 JSON을 안정적으로 출력한다.

| 입력 | 출력 |
| :--- | :--- |
| "모레 저녁에 판교 카카오 본사에서 미팅 있음" | `{"date_text": "모레", "time_text": "저녁", "location": "판교 카카오 본사", "attendees": null}` |
| "내일 오후 3시에 강남역에서 영희랑 커피 마시기로 함" | `{"date_text": "내일", "time_text": "오후 3시", "location": "강남역", "attendees": ["영희"]}` |
| 가스 자가검침 안내문(알림톡 스타일, 기간 표현 포함) | `{"date_text": "03월 05일 부터 03월 10일 까지", "time_text": null, "location": null, "attendees": null}` |

세 케이스 모두 키 이름 정확, 값 정확, 기간(Range) 표현도 원문 그대로 잘 추출됨. 알림톡류 데이터를 보강한 v3 데이터셋의 효과가 명확히 드러난다.

---

## 5. 병합(Merge)

`merge_lora.py`로 LoRA 가중치를 베이스 모델에 병합해 단일 배포 모델을 생성했다.
- 저장 위치: `gemma_schedule_extractor_v3/merged_model` (545MB, safetensors)

---

## 6. 결론

| | 4차 (CPU, v2) | 5차 (GPU, v3) |
| :--- | :---: | :---: |
| 정밀도 | fp32 | bf16 |
| 소요 시간 | 20시간 39분 | **1시간 22분** |
| 최종 Loss | 0.239 | 0.142 |
| 최종 Accuracy | 96.5% | 96.1% |
| JSON 출력 품질 | 키 이름 누락/환각 | **정상** |

지표(Loss/Accuracy)는 비슷한 수준이지만, **실제 JSON 출력 품질과 학습 속도 모두에서 5차 실험이 확실히 개선**됐다. 다음 실험부터는 이번에 확정된 구성(bf16 + `gradient_checkpointing` + `JsonlMetricsCallback` 실시간 모니터링)을 기본값으로 사용하면 된다.
