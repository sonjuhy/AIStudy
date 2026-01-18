from tokenizers import ByteLevelBPETokenizer
from transformers import Qwen2TokenizerFast
import os


def train_tokenizer_with_qwen(corpus_file_path: str, save_path: str):
    # 1. 토크나이저 초기화 및 학습
    tokenizer = ByteLevelBPETokenizer()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 학습 시작 (20억 어절 기준 약 10~30분 소요)
    tokenizer.train(
        files=[corpus_file_path],
        vocab_size=50257,  # Qwen2와 동일한 사전 크기
        min_frequency=2,  # 최소 2번 이상 나온 단어만 등록
        show_progress=True,
        special_tokens=[
            "<|endoftext|>",  # 텍스트의 끝
            "<|im_start|>",  # 대화 시작
            "<|im_end|>",  # 대화 끝
            "<pad>",  # 길이 맞춤용 패딩
        ],
    )

    # 2. 저장 (HuggingFace 포맷으로 변환)
    tokenizer.save_model(save_path)

    # 테스트 로드
    fast_tokenizer = Qwen2TokenizerFast.from_pretrained(save_path)
    test_text = "안녕하세요, 한국어 Pre-training을 시작합니다."
    print("토큰화 결과:", fast_tokenizer.tokenize(test_text))


if __name__ == "__main__":
    dataset_path = "./DataSets/AIHub/total_corpus.txt"
    save_path = "./korean_qwen_tokenizer"
    train_tokenizer_with_qwen(dataset_path, save_path)
