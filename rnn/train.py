from rnn_model import RNNModel

import torch
import torch.nn as nn
import torch.optim as optim

from tensor_dto import TensorDto


def train(tensor_dto: TensorDto):
    # 모델 초기화
    input_size = 2  # 조건1, 조건2
    hidden_size = 16
    output_size = 1  # 결과값
    model = RNNModel(input_size, hidden_size, output_size)

    # 손실 함수와 옵티마이저
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # RNN 입력 형태로 변환 (배치, 타임스텝, 특징)
    x_train_rnn = tensor_dto.x_train_tensor.unsqueeze(1)  # (batch_size, seq_length=1, input_size)
    x_test_rnn = tensor_dto.x_test_tensor.unsqueeze(1)

    # 학습
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_rnn)
        loss = criterion(outputs, tensor_dto.y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # 모델 저장
    torch.save(model.state_dict(), "rnn_model.pth")
    print("Model saved successfully.")

