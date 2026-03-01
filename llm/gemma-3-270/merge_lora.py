import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 경로 설정 ---
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(BASE_DIR)) # -> AIStudy 폴더

# 1. 베이스 모델과 학습된 V2 LoRA 가중치 경로
BASE_MODEL_ID: str = os.path.join(PROJECT_ROOT, "models", "gemma-3-270m-it")
LORA_PATH: str = os.path.join(BASE_DIR, "gemma_schedule_extractor_v2", "final_lora_weights")

# 2. 병합된 최종 모델이 저장될 폴더
MERGED_OUTPUT_DIR: str = os.path.join(BASE_DIR, "gemma_schedule_extractor_v2", "merged_model")

def merge_and_save() -> None:
    """베이스 모델과 LoRA 가중치를 병합하여 단일 모델로 저장합니다."""
    
    if not os.path.exists(LORA_PATH):
        print(f" LoRA 가중치를 찾을 수 없습니다: {LORA_PATH}")
        return

    print(" 베이스 모델을 메모리에 로드 중... (시간이 조금 걸릴 수 있습니다)")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="cpu",
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True
    )
    
    # 토크나이저는 학습 과정에서 PAD 토큰 등이 업데이트되었을 수 있으므로 LoRA 폴더에서 가져옵니다.
    print(" 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

    print(" LoRA 가중치를 불러와 베이스 모델에 덧씌웁니다...")
    peft_model = PeftModel.from_pretrained(base_model, LORA_PATH)

    print("융합(Merge) 연산을 시작합니다. CPU 메모리를 많이 사용합니다...")
    # 원리: merge_and_unload()는 LoRA의 행렬(A*B) 연산 결과를 원본 베이스 모델 행렬(W)에 직접 더해(W' = W + A*B) 
    # 완전히 독립적인 하나의 모델로 만든 뒤, 더 이상 필요 없는 LoRA 객체를 메모리에서 해제합니다.
    merged_model = peft_model.merge_and_unload()
    
    print(f" 병합 완료! 온전한 단일 모델을 [{MERGED_OUTPUT_DIR}]에 저장합니다.")
    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    
    # 일반적인 Hugging Face 모델 포맷(safetensors)으로 최종 저장
    merged_model.save_pretrained(MERGED_OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
    
    print("\n 축하합니다! 독립 실행이 가능한 완벽한 단일 모델이 만들어졌습니다.")

if __name__ == "__main__":
    merge_and_save()