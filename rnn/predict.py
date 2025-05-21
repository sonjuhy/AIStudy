from rnn_model import RNNModel

import torch


def predict(input_size: int, hidden_size: int, output_size: int):
    # 모델 로드
    loaded_model = RNNModel(input_size, hidden_size, output_size)
    loaded_model.load_state_dict(torch.load("rnn_model.pth"))
    loaded_model.eval()  # 추론 모드로 설정
    print("Model loaded successfully.")

    # 새로운 데이터로 추론
    sample_data = [[1, 0], [1, 1], [0, 0]]
    new_data = torch.tensor(data=sample_data, dtype=torch.float32).unsqueeze(1)
    # (batch_size, seq_length=1, input_size)

    with torch.no_grad():
        predictions = loaded_model(new_data)
        binary_predictions = (predictions > 0.5).float()  # 0.5 기준 이진화
        print("Predictions:", predictions.squeeze().numpy())
        print("Binary Predictions:", binary_predictions.squeeze().numpy())

