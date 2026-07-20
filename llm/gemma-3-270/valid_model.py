import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 경로 설정 (사용자 환경에 맞춤) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR)) # -> AIStudy 폴더

# 베이스 모델 경로 (이전에 사용하신 로컬 경로 혹은 Hugging Face ID)
BASE_MODEL_ID = os.path.join(PROJECT_ROOT, "models", "gemma-3-270m-it")
# 방금 학습이 완료된 LoRA 가중치 경로
LORA_PATH = os.path.join(BASE_DIR, "gemma_schedule_extractor_v3", "final_lora_weights")

# 회귀 비교용 고정 테스트 케이스 (1~4차 검증 때와 동일한 입력)
FIXED_TEST_CASES = [
    "모레 저녁에 판교 카카오 본사에서 미팅 있음",
    "내일 오후 3시에 강남역에서 영희랑 커피 마시기로 함",
    """[자가검침 기간 안내]
고객님 자가검침 기간이 도래하였습니다.
03월 05일 부터 03월 10일 까지 검침이 가능하며 아래 [자가검침 바로가기] 메뉴를 클릭하여 등록하여 주시기 바랍니다.
검침 숫자는 실제 계량기 숫자로 입력해 주시고 잘못 입력하시면 요금상 불이익이 발생할 수 있습니다.""",
]

def test_fine_tuned_model():
    print(" 토크나이저와 베이스 모델을 로드합니다...")
    # 토크나이저는 파인튜닝 폴더에 저장된 것을 사용합니다
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32

    # 1. 베이스 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map=device,
        torch_dtype=dtype
    )

    # 2. 베이스 모델에 LoRA 가중치 덧씌우기 (PeftModel 사용)
    print(" 학습된 LoRA 가중치를 모델에 결합합니다...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval() # 평가(추론) 모드로 전환

    # 3. 테스트할 프롬프트 준비 (학습할 때와 완벽히 동일한 포맷이어야 함)
    SYSTEM_PROMPT_V1 = "당신은 일정 추출기입니다. 오늘 날짜는 2026-02-22입니다. 사용자의 텍스트에서 date(YYYY-MM-DD), time(HH:MM), location, attendees(배열)를 JSON으로만 출력하세요. 없으면 null입니다."
    SYSTEM_PROMPT_V2 = "당신은 일정 정보 추출기입니다. 사용자의 텍스트에서 날짜와 시간과 관련된 '원문 표현(Raw text)'을 그대로 추출하여 date_text, time_text, location, attendees(배열)를 JSON으로만 출력하세요. 없으면 null입니다."

    def run_inference(user_input: str) -> str:
        prompt = (
            f"<start_of_turn>system\n{SYSTEM_PROMPT_V2}<end_of_turn>\n"
            f"<start_of_turn>user\n{user_input}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,      # JSON 출력이 길지 않으므로 100이면 충분
                temperature=0.1,         # 사실 기반 출력을 위해 낮게 설정
                repetition_penalty=1.1   # 반복 출력 방지
            )
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("\n" + "="*50)
    print(" 일정 추출 봇이 준비되었습니다. (종료하려면 'q' 입력)")
    print("="*50)

    if not sys.stdin.isatty():
        # 비대화형(스크립트/파이프) 실행 시: 회귀 비교용 고정 케이스로 자동 검증
        for case in FIXED_TEST_CASES:
            print(f"\n메시지 입력: {case}")
            print("\n[추출된 JSON 결과]")
            print(run_inference(case))
        return

    while True:
        user_input = input("\n메시지 입력: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            break

        print("\n[추출된 JSON 결과]")
        print(run_inference(user_input))

if __name__ == "__main__":
    test_fine_tuned_model()