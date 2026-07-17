import os
import json
import dotenv

from google import genai
from google.genai import types

dotenv.load_dotenv(".env")

# 1. API 키 설정 (환경 변수 또는 직접 입력)
# os.environ["GEMINI_API_KEY"] = "당신의_API_키" 
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 2. 데이터를 저장할 파일
OUTPUT_FILE = "schedule_dataset.jsonl"
SYSTEM_PROMPT = "당신은 일정 추출기입니다. 오늘 날짜는 2026-02-22입니다. 사용자의 텍스트에서 date(YYYY-MM-DD), time(HH:MM), location, attendees(배열)를 JSON으로만 출력하세요. 없으면 null입니다."

def generate_synthetic_data(batch_size=10):
    """Gemini를 사용해 가상의 한국어 일정 대화와 정답 JSON을 생성합니다."""
    
    prompt = f"""
    당신은 AI 학습용 데이터셋 생성기입니다. 
    한국어로 된 일상적인 메신저 대화나 음성 메모 스타일의 텍스트와, 거기서 일정을 추출한 JSON 결과를 {batch_size}개 생성해주세요.
    기준 날짜는 2026-02-22로 가정하고 상대적인 날짜(예: 내일, 다음 주 금요일)를 정확한 YYYY-MM-DD로 계산하세요.
    
    [조건]
    1. 텍스트는 오타, 구어체, 반말, 존댓말, 음성 인식 오류 스타일 등을 다양하게 섞으세요.
    2. 일정이 포함된 문장 70%, 일정이 없는 일상 대화(null 반환용) 30% 비율로 만드세요.
    3. 결과는 반드시 아래 구조의 JSON 배열(Array) 형식으로만 출력하세요. 코드 블록(```json) 없이 순수 JSON 텍스트만 출력하세요.
    
    [출력 형식 예시]
    [
      {{
        "user_input": "아 담주 수욜 저녁 8시에 홍대 1번출구에서 재석이형 보기로함",
        "expected_json": {{"date": "2026-02-25", "time": "20:00", "location": "홍대 1번출구", "attendees": ["재석"]}}
      }},
      {{
        "user_input": "나 지금 막 버스 탔어. 금방 갈게!",
        "expected_json": {{"date": null, "time": null, "location": null, "attendees": null}}
      }}
    ]
    """

    print("Gemini API 호출 중... 데이터를 생성하고 있습니다.")
    
    # Gemini 2.5 Flash 모델 사용 (빠르고 저렴하며 데이터 생성에 탁월함)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7, # 다양성을 위해 약간의 창의성 허용
        )
    )
    
    return response.text

def save_to_jsonl(raw_json_string):
    """생성된 JSON 문자열을 Gemma 학습 포맷(JSONL)으로 변환하여 저장합니다."""
    try:
        data_list = json.loads(raw_json_string)
        
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for item in data_list:
                user_input = item["user_input"]
                expected_json = json.dumps(item["expected_json"], ensure_ascii=False)
                
                # Gemma 3 SFT 프롬프트 포맷
                gemma_format = (
                    f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
                    f"<start_of_turn>user\n{user_input}<end_of_turn>\n"
                    f"<start_of_turn>model\n{expected_json}<end_of_turn>"
                )
                
                jsonl_line = json.dumps({"text": gemma_format}, ensure_ascii=False)
                f.write(jsonl_line + "\n")
                
        print(f" 성공적으로 {len(data_list)}개의 데이터를 {OUTPUT_FILE}에 추가했습니다.")
        
    except json.JSONDecodeError as e:
        print(f" JSON 파싱 에러 발생: {e}")
        print("Gemini 응답 원본:\n", raw_json_string)

# 실행 부 (예: 10개씩 5번 반복하여 총 50개 생성 테스트)
if __name__ == "__main__":
    # 데이터 생성 루프
    for i in range(76):
        print(f"\n--- Batch {i+1} ---")
        generated_text = generate_synthetic_data(batch_size=10)
        save_to_jsonl(generated_text)