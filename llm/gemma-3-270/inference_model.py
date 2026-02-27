import torch
import dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# .env 파일에서 환경변수 로드
dotenv.load_dotenv(".env")

class GemmaInference:
    def __init__(self, model_path: str = "./models/gemma-3-270m-it"):
        """
        로컬에 다운로드된 Gemma 모델을 로드하여 추론을 준비하는 클래스
        
        Args:
            model_path (str): 로컬에 저장된 가중치 모델의 경로
        """
        self.model_path = model_path
        print(f"'{self.model_path}' 경로에서 모델 로드 중...")
        
        try:
            # 로컬 경로에서 토크나이저 및 모델 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                dtype="auto", # 사용하는 하드웨어에 맞춰 자동으로 dtype 설정
            )
            print("모델 로드 성공!")
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            print(f"'{self.model_path}' 경로에 모델이 올바르게 다운로드되었는지 확인해주세요.")
            raise e

    def generate(self, prompt_text: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        주어진 내용에 대해 모델을 실행하여 답변을 생성합니다.
        
        Args:
            prompt_text (str): 생성할 텍스트의 입력 내용 (유저의 질문)
            max_new_tokens (int): 생성할 최대 토큰 수
            temperature (float): 생성 다양성 조절치 (높을수록 창의적)
            top_p (float): 누적 확률 컷오프 (nucleus sampling)
            
        Returns:
            str: 모델이 프롬프트를 바탕으로 생성한 텍스트 결과
        """
        messages = [
            {
                "role": "user", 
                "content": prompt_text
            }
        ]
        
        # 챗 템플릿 적용 및 텐서 변환
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        print("추론(Inference) 진행 중입니다...\n")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            
        # 입력으로 들어간 프롬프트 부분은 제외하고 생성된 토큰만 추출 후 디코딩
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text

if __name__ == "__main__":
    # 1. 로컬에 저장된 모델을 사용하여 클래스 인스턴스 생성
    # 앞서 utils.py를 통해 다운로드 받았던 디렉토리 경로를 지정합니다.
    local_model_directory = "./models/gemma-3-270m-it" 
    
    try:
        # 모델을 메모리에 로드
        gemma_infer = GemmaInference(model_path=local_model_directory)
        
        # 2. 실행할 프롬프트(질문) 지정
        user_question = "Introduce yourself in English"
        
        # 3. 텍스트 생성(추론) 실행
        response = gemma_infer.generate(prompt_text=user_question)
        
        # 4. 결과 출력
        print("=" * 50)
        print(response)
        print("=" * 50)

        # 2. 실행할 프롬프트(질문) 지정
        user_question = "한국어로 자신을 소개해 주세요"
        
        # 3. 텍스트 생성(추론) 실행
        response = gemma_infer.generate(prompt_text=user_question)
        
        # 4. 결과 출력
        print("=" * 50)
        print(response)
        print("=" * 50)
        
    except Exception as e:
        print("에러가 발생하여 추론 실행을 중단합니다.")
