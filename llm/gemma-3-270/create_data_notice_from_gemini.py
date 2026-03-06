import os
import json
import time
from typing import List, Dict
import dotenv

from google import genai
from google.genai import types

dotenv.load_dotenv(".env")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 기존에 1,000개를 모았던 파일에 그대로 덧붙입니다.
OUTPUT_FILE: str = "schedule_dataset_v2.jsonl"
SYSTEM_PROMPT: str = (
    "당신은 일정 정보 추출기입니다. 사용자의 텍스트에서 날짜와 시간과 관련된 '원문 표현(Raw text)'을 그대로 추출하여 date_text, time_text, location, attendees(배열)를 JSON으로만 출력하세요. 없으면 null입니다."
)


def generate_notice_data(batch_size: int = 10) -> str:
    """Gemini를 사용해 [알림톡/예약문자/공지사항] 스타일의 일정 데이터를 생성합니다."""

    prompt: str = f"""
    당신은 AI 학습용 데이터셋 생성기입니다. 
    한국어로 된 '공공기관, 병원, 택배, 가스검침, 예약확인 등의 알림톡이나 단체 문자' 텍스트와, 거기서 일정을 추출한 JSON 결과를 {batch_size}개 생성해주세요.
    
    [핵심 조건]
    1. 텍스트는 '[Web발신]', '고객님', '안내드립니다' 같은 딱딱하고 형식적인 문체를 사용하세요.
    2. 단일 날짜뿐만 아니라 "03월 05일 부터 03월 10일 까지" 같은 **기간(Range) 표현**도 많이 포함시키세요. 기간 표현도 그대로 date_text에 추출합니다.
    3. 참석자(attendees)는 주로 없거나(null) 본인이므로 null 처리하세요.
    4. 절대로 날짜를 계산하지 말고 원문(Raw text)을 추출하세요.
    
    [출력 형식 예시]
    [
      {{
        "user_input": "[자가검침 기간 안내] 고객님 자가검침 기간이 도래하였습니다. 03월 05일 부터 03월 10일 까지 검침이 가능하며...",
        "expected_json": {{"date_text": "03월 05일 부터 03월 10일 까지", "time_text": null, "location": null, "attendees": null}}
      }},
      {{
        "user_input": "[예약확인] 김철수님, 10월 25일 오전 10시 30분에 서울대병원 내과 진료 예약이 완료되었습니다.",
        "expected_json": {{"date_text": "10월 25일", "time_text": "오전 10시 30분", "location": "서울대병원 내과", "attendees": null}}
      }}
    ]
    코드 블록(```json) 없이 순수 JSON 배열만 출력하세요.
    """

    print("Gemini API 호출 중... 알림톡 데이터를 생성하고 있습니다.")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.8),
    )
    return response.text


def save_to_jsonl(raw_json_string: str) -> None:
    """생성된 JSON 문자열을 파싱하여 기존 JSONL 파일에 누적 저장합니다."""
    cleaned_string: str = (
        raw_json_string.replace("```json", "").replace("```", "").strip()
    )

    try:
        data_list: List[Dict] = json.loads(cleaned_string)

        # 'a' (Append) 모드로 기존 1,000개 데이터 밑에 추가합니다.
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for item in data_list:
                user_input: str = item["user_input"]
                expected_json: str = json.dumps(
                    item["expected_json"], ensure_ascii=False
                )

                gemma_format: str = (
                    f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
                    f"<start_of_turn>user\n{user_input}<end_of_turn>\n"
                    f"<start_of_turn>model\n{expected_json}<end_of_turn>"
                )

                jsonl_line: str = json.dumps({"text": gemma_format}, ensure_ascii=False)
                f.write(jsonl_line + "\n")

        print(f"✅ 알림톡 데이터 {len(data_list)}개를 {OUTPUT_FILE}에 추가했습니다.")

    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 에러 발생: {e}")


if __name__ == "__main__":
    # 알림톡 데이터 200개 추가 생성 (10개씩 20번 반복)
    total_iterations: int = 20

    for i in range(total_iterations):
        print(f"\n--- Batch {i+1} / {total_iterations} ---")
        generated_text: str = generate_notice_data(batch_size=10)
        save_to_jsonl(generated_text)

        if i < total_iterations - 1:
            time.sleep(2)
