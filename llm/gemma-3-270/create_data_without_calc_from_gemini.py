import os
import json
import time
from typing import Optional
import dotenv

from google import genai
from google.genai import types

dotenv.load_dotenv(".env")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 1. 새로운 파일명과 완전히 달라진 시스템 프롬프트
OUTPUT_FILE: str = "schedule_dataset_v2.jsonl"
SYSTEM_PROMPT: str = "당신은 일정 정보 추출기입니다. 사용자의 텍스트에서 날짜와 시간과 관련된 '원문 표현(Raw text)'을 그대로 추출하여 date_text, time_text, location, attendees(배열)를 JSON으로만 출력하세요. 없으면 null입니다."

def generate_synthetic_data(batch_size: int = 10) -> str:
    """Gemini를 사용해 원문 텍스트 추출 방식의 일정 대화와 정답 JSON을 생성합니다."""
    
    prompt: str = f"""
    당신은 AI 학습용 데이터셋 생성기입니다. 
    한국어로 된 일상적인 메신저 대화나 음성 메모 텍스트와, 거기서 일정을 추출한 JSON 결과를 {batch_size}개 생성해주세요.
    
    [핵심 조건: 원문 추출]
    절대로 날짜를 계산하거나 임의의 YYYY-MM-DD로 변환하지 마세요. 
    사용자가 말한 "내일", "다음 주 수요일", "저녁 8시", "낼모레" 같은 원문(Raw text) 텍스트 자체를 그대로 추출해야 합니다.
    
    [일반 조건]
    1. 텍스트는 오타, 구어체, 반말, 존댓말 등을 다양하게 섞으세요.
    2. 일정이 포함된 문장 70%, 일정이 없는 일상 대화(null 반환용) 30% 비율로 만드세요.
    3. 결과는 반드시 아래 구조의 JSON 배열(Array) 형식으로만 출력하세요.
    
    [출력 형식 예시]
    [
      {{
        "user_input": "아 담주 수욜 저녁 8시에 홍대 1번출구에서 재석이형 보기로함",
        "expected_json": {{"date_text": "담주 수욜", "time_text": "저녁 8시", "location": "홍대 1번출구", "attendees": ["재석"]}}
      }},
      {{
        "user_input": "모레 점심쯤에 판교 카카오 본사에서 미팅 있음",
        "expected_json": {{"date_text": "모레", "time_text": "점심쯤", "location": "판교 카카오 본사", "attendees": null}}
      }},
      {{
        "user_input": "나 지금 막 버스 탔어. 금방 갈게!",
        "expected_json": {{"date_text": null, "time_text": null, "location": null, "attendees": null}}
      }}
    ]
    """

    print("Gemini API 호출 중... 데이터를 생성하고 있습니다.")
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7, 
        )
    )
    
    return response.text
def save_to_jsonl(raw_json_string: str) -> None:
    """생성된 JSON 문자열을 파싱하여 JSONL 파일에 누적 저장합니다."""
    
    # [핵심 추가] 모델이 고집부린 마크다운 코드 블록(```json, ```)을 강제로 제거합니다.
    cleaned_string = raw_json_string.replace("```json", "").replace("```", "").strip()
    
    try:
        # [수정] 원본 대신 깨끗해진 문자열을 파싱합니다.
        data_list: list[dict] = json.loads(cleaned_string)
        
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for item in data_list:
                user_input: str = item["user_input"]
                expected_json: str = json.dumps(item["expected_json"], ensure_ascii=False)
                
                gemma_format: str = (
                    f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
                    f"<start_of_turn>user\n{user_input}<end_of_turn>\n"
                    f"<start_of_turn>model\n{expected_json}<end_of_turn>"
                )
                
                jsonl_line: str = json.dumps({"text": gemma_format}, ensure_ascii=False)
                f.write(jsonl_line + "\n")
                
        print(f" 성공적으로 {len(data_list)}개의 데이터를 {OUTPUT_FILE}에 추가했습니다.")
        
    except json.JSONDecodeError as e:
        print(f" JSON 파싱 에러 발생: {e}")
        print("Gemini 응답 원본:\n", raw_json_string)

if __name__ == "__main__":
    # 총 1,000개 생성 (10개씩 100번 반복)
    total_iterations: int = 100
    
    for i in range(total_iterations):
        print(f"\n--- Batch {i+1} / {total_iterations} ---")
        generated_text: str = generate_synthetic_data(batch_size=10)
        save_to_jsonl(generated_text)
        
        if i < total_iterations - 1:
            print("API 과부하 방지를 위해 2초 대기 중...")
            time.sleep(2)