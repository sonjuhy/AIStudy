import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensor_dto import TensorDto


def make_data_about_weather() -> dict:

    # 랜덤 데이터 생성
    np.random.seed(42)
    data_size = 100

    # 조건 생성
    time = np.random.randint(0, 24, data_size)  # 시간 (0~23)
    wind_speed = np.random.randint(0, 11, data_size)  # 풍속 (0~10)
    cloud_cover = np.random.randint(0, 3, data_size)  # 구름양 (0, 1, 2)
    precipitation = np.random.randint(0, 11, data_size)  # 강수량 (0~10)

    # 결과 생성: 날씨 좋음(1) 또는 나쁨(0) 가정 규칙
    # 날씨가 좋음: 풍속 < 5, 구름양 < 2, 강수량 == 0
    result = ((wind_speed < 5) & (cloud_cover < 2) & (precipitation == 0)).astype(int)

    # 데이터프레임 생성
    data = pd.DataFrame({
        "Time": time,
        "WindSpeed": wind_speed,
        "CloudCover": cloud_cover,
        "Precipitation": precipitation,
        "WeatherGood": result
    })

    data.head(), data.tail()

    return {
        'head': data.head(),
        'tail': data.tail()
    }


def preprocess_to_tensor() -> TensorDto:
    # 가상의 데이터 생성
    np.random.seed(42)
    data_size = 1000
    time_index = np.arange(data_size)
    condition1 = np.random.randint(0, 2, data_size)
    condition2 = np.random.randint(0, 2, data_size)
    result = (condition1 & condition2).astype(int)  # 단순한 논리 AND로 결과 생성
    
    # 데이터 합치기 (시간 순 인덱스는 학습에 사용하지 않음)
    data = np.column_stack((condition1, condition2, result))
    
    # Train-Test Split
    x = data[:, :-1]  # 조건1, 조건2
    y = data[:, -1]   # 결과값
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Tensor 변환
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # (N, 1) 형태
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return TensorDto(
        x_train=x_train_tensor,
        y_train=y_train_tensor,
        x_test=x_test_tensor,
        y_test=y_test_tensor
    )
    
    
