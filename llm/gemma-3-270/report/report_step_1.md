# 🚀 Gemma-3-270M LoRA 파인튜닝 실험 리포트

본 리포트는 초경량 모델인 **Gemma-3-270M**을 활용하여 자연어 메시지에서 일정을 추출하고, 이를 정형화된 데이터로 변환하는 **On-Device AI** 기술 실증 과정을 담고 있습니다.

---

## 1. 실험 환경 및 모델 구성 (Environment)
* **Base Model**: `google/gemma-3-270m-it`
* **Training Method**: LoRA (Low-Rank Adaptation)
* **Hardware**: Local CPU Environment (VRAM ~1GB 사용)
* **Dataset**: `schedule_dataset.jsonl` (Train samples: 100 / Epochs: 3)

---

## 2. 학습 로그 분석 (Training Metrics)
학습 과정에서 도출된 주요 지표 및 성능 데이터입니다.

| 항목 (Metric) | 수치 (Value) | 분석 및 근거 |
| :--- | :--- | :--- |
| **Train Loss** | **3.5780** | 학습 데이터 대비 손실값이 높아 추가 에폭(Epoch) 필요 |
| **Mean Token Accuracy** | **52.38%** | 토큰 예측 정확도가 낮아 JSON 구조 생성에 어려움 발생 |
| **Runtime** | **661.07s** | CPU 환경에서도 11분 내외로 학습 가능한 경량성 확인 |
| **Learning Rate** | **8e-05** | 안정적인 수렴을 위해 설정된 최적 학습률 |

---

## 3. 결과 검증 (Validation)
추출 봇 테스트 결과, 자연어 이해도는 높으나 **출력 형식(JSON)** 준수율에서 개선점이 발견되었습니다.

* **Input**: "모레 저녁에 판교 카카오 본사에서 미팅 있음"
* **Output**: "네, 모레 저녁에 판교 카카오 본사에서 미팅 있음입니다." (평서문 출력)
* **Issue**: 모델이 JSON 구조 대신 대화형 응답을 우선시함 (Instruction Following 미흡)

---

## 4. 개선을 위한 기술적 제언 (Technical Insights)

### 🛠️ 데이터셋 및 학습 epoch 최적화 (Python Type Hinting 적용)
JSON 형태의 엄격한 출력을 위해 다음과 같은 개선이 필요합니다.

1. **데이터셋 규모 확장 (Dataset Scaling)**:
   - 현재 100개의 샘플로는 복잡한 JSON 스키마를 완벽히 학습하기에 부족함이 확인되었습니다.
   - 다양한 문장 패턴과 예외 상황을 포함하여 데이터셋을 **1,000개 이상**으로 확장함으로써 모델의 일반화 성능을 높여야 합니다.

2. **학습 에폭 증가 (Increasing Epochs)**:
   - 현재 3 Epoch에서 측정된 Accuracy(52.38%)는 모델이 출력 형식을 충분히 내재화하지 못한 상태임을 나타냅니다.
   - 학습 횟수를 **20 Epoch 이상**으로 대폭 늘려 손실값(Loss)을 1.0 미만으로 낮추고, JSON 구조에 대한 강한 제약 조건을 학습시켜야 합니다.

