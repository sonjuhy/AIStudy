import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR)) # -> AIStudy 폴더

def finetune_gemma_cpu(
    dataset_path: str,
    output_dir: str = os.path.join(BASE_DIR, "gemma_schedule_extractor_v2"),
    model_id: str = os.path.join(PROJECT_ROOT, "models", "gemma-3-270m-it"),
    num_epochs: int = 3,
    batch_size: int = 2,
    max_seq_length: int = 256
):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")

    print(f" [{model_id}] 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f" CPU에 모델 로드 중 (메모리 약 1GB 필요)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cpu",
        torch_dtype=torch.float32 # 수정됨: dtype -> torch_dtype
    )

    # LoRA 설정 객체만 생성합니다. (get_peft_model 호출 삭제)
    peft_config = LoraConfig(
        r=32, 
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print(f" 데이터셋 [{dataset_path}] 로드 중...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # CPU 맞춤형 학습 인자 설정
    training_args = SFTConfig(
        dataset_text_field="text",
        output_dir=output_dir,
        max_length=max_seq_length,         
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=num_epochs,
        use_cpu=True,                      
        remove_unused_columns=False,
    )

    # SFTTrainer가 peft_config를 받아 내부적으로 LoRA를 적용합니다.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
    )

    print(" CPU 환경에서 파인튜닝을 시작합니다...")
    trainer.train()

    final_model_path = os.path.join(output_dir, "final_lora_weights")
    print(f"학습 완료. 모델 가중치와 토크나이저를 [{final_model_path}]에 저장합니다.")
    
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    return final_model_path

if __name__ == "__main__":
    DATASET_FILE = os.path.join(BASE_DIR, "schedule_dataset_v2.jsonl")
    
    try:
        saved_path = finetune_gemma_cpu(
            dataset_path=DATASET_FILE,
            num_epochs=20,
            batch_size=2
        )
        print(f"\n 모든 과정이 성공적으로 끝났습니다. 저장 경로: {saved_path}")
    except Exception as e:
        print(f"\n 에러가 발생했습니다: {e}")