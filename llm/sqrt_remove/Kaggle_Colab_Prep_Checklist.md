# GPT-2 지수 연산 대체 실험 — Kaggle/Colab 무료버전 준비 체크리스트

> `Gpt2_exp_replacement_guide.md` 실험을 Colab/Kaggle **무료 GPU(T4/P100)** 위에서
> 세션 끊김·시간 제한 없이 돌리기 위해 사전에 준비해야 할 것들을 정리한다.

---

## 1. 무료 티어 제약 조건 요약

| 항목 | Colab 무료 | Kaggle 무료 |
|---|---|---|
| GPU | T4 (16GB) | T4×2 또는 P100 (16GB) |
| 세션 최대 길이 | ~12시간, **유휴 시 연결 끊김** | 9시간/세션 |
| 주간 한도 | 없음(단, 사용량 많으면 우선순위 하락) | **주당 30시간** |
| 디스크 | 세션 종료 시 초기화 | 세션 종료 시 초기화 (Output만 유지) |
| 백그라운드 실행 | 브라우저 탭 닫으면 위험 | 창 닫아도 커널 계속 실행 가능(Kaggle이 더 안전) |

**결론**: 하나의 실험(예: block_size=1024, n_layer=12)을 12시간 안에 못 끝낼 수 있으므로
**중단 후 재개(resume)가 가능한 구조**가 필수다. Kaggle이 세션 안정성 면에서 더 유리하므로
본 실험(baseline vs linear 비교, 긴 학습)은 Kaggle을, 짧은 파이프라인 검증은 Colab을 쓰는 것을 권장.

---

## 2. 코드 준비 (실험 전에 로컬에서 끝내야 할 것)

- [ ] `src/attention/softmax_attention.py`, `src/attention/linear_attention.py` 구현 및
      단위 테스트(`tests/test_*.py`) **로컬에서 전부 통과** 확인 (TDD Green 상태로 업로드)
- [ ] `src/model.py`의 `GPTConfig(attention_type=...)` 스위치가 두 방식 모두에서
      `test_model_forward.py` 통과하는지 확인
- [ ] `scripts/train.py`, `scripts/benchmark.py`에 **체크포인트 저장/재개 로직** 추가
      - `torch.save({"model": ..., "optimizer": ..., "step": ...}, path)`
      - 시작 시 체크포인트 존재하면 자동으로 이어서 학습(resume)
      - N step마다 저장 (예: 500 step 또는 10분마다) — 세션이 중간에 끊겨도 손실 최소화
- [ ] 로그를 CSV/JSON으로 append 방식 저장 (매 스텝/eval마다) → 세션 재시작해도 기록 이어짐
- [ ] 코드 전체를 **git repo 또는 GitHub gist**에 올려서 노트북에서 `git clone` 한 줄로 받을 수 있게 준비
      (Colab/Kaggle 파일 업로드는 세션마다 반복해야 해서 번거로움)

---

## 3. 데이터셋 준비

- [ ] 1단계 TinyShakespeare는 매번 `load_dataset`으로 즉시 받아도 무방 (용량 작음)
- [ ] 2단계 WikiText-103(~500MB)은 **토큰화 결과를 미리 캐싱**
      - Kaggle: 토큰화된 데이터를 **Kaggle Dataset**으로 한 번 업로드해두고 노트북에서 Add Data로 불러오기
        (매 세션 재다운로드/재토큰화 방지 → 시간 절약, 주 30시간 한도 안에서 특히 중요)
      - Colab: Google Drive에 토큰화 결과(.bin/.pt) 저장 후 마운트해서 재사용
- [ ] 토크나이저는 GPT-2 기본 tokenizer(tiktoken/HuggingFace GPT2Tokenizer)로 고정,
      버전 명시(라이브러리 버전 차이로 토큰 ID 달라지는 것 방지)

---

## 4. 체크포인트 & 출력 저장 전략

- [ ] **Colab**: `from google.colab import drive; drive.mount('/content/drive')` →
      `/content/drive/MyDrive/gpt2_exp/checkpoints/` 경로에 저장
- [ ] **Kaggle**: `/kaggle/working/` 에 저장 → 노트북 실행 종료 시 **Output**으로 자동 보존
      (다음 세션에서 "Add Data" > 이전 노트북 output 버전으로 이어받기)
- [ ] baseline(softmax)과 linear 각각 별도 체크포인트 디렉토리로 분리
- [ ] 실험 결과(perplexity, step time, tokens/sec) 저장 파일명에
      `{attention_type}_{n_layer}_{block_size}_{seed}.json` 형태로 메타정보 포함
      → 나중에 결과 정리 템플릿(가이드 7장 표)에 자동 매핑 가능

---

## 5. GPU 배정 편차 최소화 (가이드 4장 "주의" 대응)

- [ ] baseline → linear를 **같은 노트북 세션 안에서 연속 실행** (커널 재시작 없이)
- [ ] 실행 전 `torch.cuda.get_device_name(0)`으로 실제 배정된 GPU 종류(T4/P100) 로그 기록
      → Kaggle은 세션마다 T4 또는 P100이 랜덤 배정될 수 있어, 결과 비교 시 GPU 종류가
      동일했는지 반드시 확인해야 함
- [ ] `torch.backends.cudnn.benchmark = True` 등 설정을 두 방식에 동일하게 적용

---

## 6. 시간·자원 예산 계획 (주 30시간 한도 대비)

| 실험 | 예상 소요(추정, T4 기준) | 비고 |
|---|---|---|
| 1단계 파이프라인 검증 (TinyShakespeare, n_layer=4) | ~30분 | Colab에서 반복 가능 |
| 2단계 본 실험 1세트 (block_size 256/512/1024 × softmax/linear × seed 3~5개) | 수 시간~10시간+ | Kaggle 세션 분할 필요 가능성 높음 |

- [ ] 본 실험은 **block_size 단계별로 노트북을 분리**해서 실행
      (하나의 노트북에서 256→512→1024를 순차 실행하면 12시간 초과 위험)
- [ ] 시드 3~5개 반복은 우선 **seed 1개로 파이프라인 전체를 끝까지 검증**한 뒤
      나머지 시드를 추가 세션에서 진행 (한 번에 다 돌리다가 시간 초과로 날리는 것 방지)
- [ ] 주간 30시간을 넘길 것 같으면 Colab(무료, 세션 제한만 있고 주간 총량 제한 없음)으로
      일부 짧은 반복 실험을 분산

---

## 7. 노트북(`notebooks/colab_experiment.ipynb`) 셀 구성 준비

- [ ] Cell 1: GPU 확인 (`torch.cuda.is_available()`, `get_device_name`)
- [ ] Cell 2: 저장소 clone + 의존성 설치 (`pip install -r requirements.txt`)
- [ ] Cell 3: Drive 마운트(Colab) / Dataset 연결(Kaggle) — 체크포인트·데이터 캐시 경로 설정
- [ ] Cell 4: 체크포인트 존재 여부 확인 → 있으면 resume, 없으면 처음부터 시작
- [ ] Cell 5: 학습 루프 실행 (주기적 저장 포함)
- [ ] Cell 6: benchmark.py 실행 → 결과 JSON/CSV 저장
- [ ] Cell 7: 결과를 Drive/Kaggle Output에 최종 백업

---

## 8. requirements.txt / 환경 고정

- [ ] `torch`, `datasets`, `tiktoken`(또는 `transformers`) 버전을 명시적으로 고정
      (Colab/Kaggle 기본 이미지 버전이 수시로 바뀌어 재현성 깨질 수 있음)
- [ ] 로컬 개발 환경과 버전 맞추기 어렵다면 최소한 **실험 시작 시점에 버전을 로그로 기록**
