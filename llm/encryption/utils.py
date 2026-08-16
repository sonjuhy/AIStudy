import os
from huggingface_hub import snapshot_download


def download_huggingface_model(
    model_id: str = "HuggingFaceTB/SmolLM-135M",
    save_directory: str = "./models/SmolLM-135M",
):
    """
    Hugging Face 허브에서 모델 가중치 및 설정 파일들을 로컬 디렉토리로 다운로드합니다.
    (메모리에 모델을 로드하지 않고 파일만 다운로드하므로 안전하고 빠릅니다.)

    Args:
        model_id (str): 다운로드할 Hugging Face 모델 ID
        save_directory (str): 모델을 저장할 로컬 경로

    Returns:
        str: 다운로드가 완료된 로컬 디렉토리 경로
    """
    print(f"'{model_id}' 모델 다운로드 시작...")

    os.makedirs(save_directory, exist_ok=True)

    try:
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=save_directory,
            local_dir_use_symlinks=False,
        )
        print(f"다운로드 완료! 모델이 '{downloaded_path}' 경로에 안전하게 저장되었습니다.")
        return downloaded_path

    except Exception as e:
        print(f"다운로드 중 오류가 발생했습니다: {e}")
        print("Hugging Face 인증(HF_TOKEN)이 제대로 설정되어 있는지 확인해주세요.")
        raise e


if __name__ == "__main__":
    download_huggingface_model()
