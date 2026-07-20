import os

# CUDA 컨텍스트 초기화 전에 설정해야 적용됨: 단편화로 인한 OOM 완화
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # -> AIStudy 폴더


class JsonlMetricsCallback(TrainerCallback):
    """on_log 시점마다 loss/accuracy 등을 metrics.jsonl에 한 줄씩 기록.
    콘솔 출력(로그 레벨/버퍼링)에 의존하지 않고 실시간 모니터링용 데이터를 남기기 위함."""

    def __init__(self, path: str):
        self.path = path
        self._start = time.time()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(self.path, "w", encoding="utf-8").close()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        record = dict(logs)
        record["step"] = state.global_step
        record["max_steps"] = state.max_steps
        record["elapsed_sec"] = round(time.time() - self._start, 1)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def finetune_gemma(
    dataset_path: str,
    output_dir: str = os.path.join(BASE_DIR, "gemma_schedule_extractor_v3"),
    model_id: str = os.path.join(PROJECT_ROOT, "models", "gemma-3-270m-it"),
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_seq_length: int = 256,
):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    # Gemma는 임베딩 스케일 값이 fp16 표현범위(~65504)를 넘어 NaN이 잘 터지는 것으로
    # 알려져 있어 bf16을 사용한다. GTX 16시리즈(Turing)는 bf16 텐서코어 가속은 없지만
    # (연산이 fp32로 폴백) 지수범위가 fp32와 같아 안정적으로 동작한다.
    dtype = torch.bfloat16 if use_cuda else torch.float32

    print(f" [{model_id}] 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        print(f" GPU({gpu_name})에 모델 로드 중 (bf16)...")
    else:
        print(" CUDA를 사용할 수 없어 CPU에 모델 로드 중 (메모리 약 1GB 필요)...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=dtype,
    )
    if use_cuda:
        # gradient checkpointing과 PEFT(베이스 모델 동결)를 함께 쓰려면
        # 입력 임베딩에 grad가 흐르도록 명시적으로 켜줘야 함
        model.enable_input_require_grads()

    # LoRA 설정 객체만 생성합니다. (get_peft_model 호출 삭제)
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f" 데이터셋 [{dataset_path}] 로드 중...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # GPU(4GB, 데스크톱 환경과 VRAM을 공유) 기준 학습 인자:
    # 배치는 작게, gradient checkpointing으로 활성화 메모리를 아껴 OOM을 방지
    training_args = SFTConfig(
        dataset_text_field="text",
        output_dir=output_dir,
        max_length=max_seq_length,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=num_epochs,
        use_cpu=not use_cuda,
        # 가중치를 이미 bf16으로 로드했으므로 Trainer의 fp16/bf16 AMP는 켜지 않는다.
        # (fp16 AMP를 이중으로 걸면 NaN grad_norm이 발생하는 걸 확인함)
        gradient_checkpointing=use_cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_cuda else None,
        dataloader_pin_memory=use_cuda,
        remove_unused_columns=False,
    )

    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    print(f" 실시간 모니터링용 지표를 [{metrics_path}]에 기록합니다...")

    # SFTTrainer가 peft_config를 받아 내부적으로 LoRA를 적용합니다.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        callbacks=[JsonlMetricsCallback(metrics_path)],
    )

    print(f" {device.upper()} 환경에서 파인튜닝을 시작합니다...")
    trainer.train()

    final_model_path = os.path.join(output_dir, "final_lora_weights")
    print(f"학습 완료. 모델 가중치와 토크나이저를 [{final_model_path}]에 저장합니다.")

    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    return final_model_path


if __name__ == "__main__":
    DATASET_FILE = os.path.join(BASE_DIR, "schedule_dataset_v3.jsonl")

    try:
        saved_path = finetune_gemma(
            dataset_path=DATASET_FILE, num_epochs=20, batch_size=1
        )
        print(f"\n 모든 과정이 성공적으로 끝났습니다. 저장 경로: {saved_path}")
    except Exception as e:
        print(f"\n 에러가 발생했습니다: {e}")
