import torch.nn as nn


# MobileNet V1 논문 URL : https://arxiv.org/pdf/1704.04861


def explain_to_conv2d_and_avgpool():
    # Conv / s2 3 × 3 × 3 × 32 224 × 224 × 3
    # 입력 크기 : 224 x 224 x 3 -> (H, W, C)
    # 커널 크기 : 3 x 3 ('3 x 3' x 3 x 32) -> 2D 필터 크기
    # 입력 채널 : 3 (224 x 224 x '3')
    # 출력 채널 : 32 (3 x 3 x 3 x '32')
    # stride : 2 (s'2')
    # padding : k = 2p + 1 -> 3 = 2p + 1 -> p = 1
    # 출력 채널 공식 : Hout = (Hin - K + 2p) / S + 1
    conv_2d_1 = nn.Conv2d(
        in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1
    )

    # Conv dw / s1 3 × 3 × 32 dw 112 × 112 × 32
    # 입력 크기 : 112 x 112 x 32 -> (H, W, C)
    # 커널 크기 : 3 x 3 ('3 x 3' x 32) -> 2D 필터 크기
    # 입력 채널 : 3 (112 x 112 x '32')
    # 출력 채널 : 32 (3 x 3 x '32')
    # stride : 2 (s'1')
    # 타입	            groups 값	               의미
    # 일반 Conv	        1	                      모든 채널이 서로 연결됨
    # Depthwise Conv   groups = in_channels	     채널별로 독립적인 필터
    # Grouped Conv	   1 < groups < in_channels	 일부 채널끼리만 연결
    conv_depth_wise_2 = nn.Conv2d(
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=32,  # depth wise의 경우, in channels와 동일 값
    )
    # Conv / s1 1 × 1 × 32 × 64 112 × 112 × 32
    # padding : k = 2p + 1 -> 1 = 2p + 1 -> p = 0
    conv_2d_3 = nn.Conv2d(
        in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0
    )

    # Avg Pool / s1 Pool 7 × 7 7 × 7 × 1024
    # 여기선 feature map을 추출하는게 아니기에 padding값을 0으로 설정
    avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
    # 만약 입력 값이 7 x 7이 아닐 경우 대비, 자동으로 입력값 크기 상관없이 1x1로 결과물을 해주는 함수(일반적)
    avg_pool_adaptive = nn.AdaptiveAvgPool2d(output_size=(1, 1))


def conv(
    in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,  # 편향 가중치 사용 여부. 뒤의 BatchNorm 에서 편향을 사용하기에 중복이라서 비활성화
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU6(inplace=True),
    )


def conv_dw(
    in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,  # 편향 가중치 사용 여부. 뒤의 BatchNorm 에서 편향을 사용하기에 중복이라서 비활성화
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU6(inplace=True),
    )


class MobileNetV1(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.model = nn.Sequential(
            conv(3, 32, 3, 2, 1),  # Conv / s2 3 × 3 × 3 × 32 224 × 224 × 3
            conv_dw(32, 32, 3, 1, 1),  # Conv dw / s1 3 × 3 × 32 dw 112 × 112 × 32
            conv(32, 64, 1, 1, 0),  # Conv / s1 1 × 1 × 32 × 64 112 × 112 × 32
            conv_dw(64, 64, 3, 2, 1),  # Conv dw / s2 3 × 3 × 64 dw 112 × 112 × 64
            conv(64, 128, 1, 1, 0),  # Conv / s1 1 × 1 × 64 × 128 56 × 56 × 64
            conv_dw(128, 128, 3, 1, 1),  # Conv dw / s1 3 × 3 × 128 dw 56 × 56 × 128
            conv(128, 128, 1, 1, 0),  # Conv / s1 1 × 1 × 128 × 128 56 × 56 × 128
            conv_dw(128, 128, 3, 2, 1),  # Conv dw / s2 3 × 3 × 128 dw 56 × 56 × 128
            conv(128, 256, 1, 1, 0),  # Conv / s1 1 × 1 × 128 × 256 28 × 28 × 128
            conv_dw(256, 256, 3, 1, 1),  # Conv dw / s1 3 × 3 × 256 dw 28 × 28 × 256
            conv(256, 256, 1, 1, 0),  # Conv / s1 1 × 1 × 256 × 256 28 × 28 × 256
            conv_dw(256, 256, 3, 2, 1),  # Conv dw / s2 3 × 3 × 256 dw 28 × 28 × 256
            conv(256, 512, 1, 1, 0),  # Conv / s1 1 × 1 × 256 × 512 14 × 14 × 256
            # 5×
            *[
                layer
                for _ in range(5)
                for layer in (
                    conv_dw(512, 512, 3, 1, 1),
                    conv(512, 512, 1, 1, 0),
                )
            ],
            conv_dw(512, 512, 3, 2, 1),  # Conv dw / s2 3 × 3 × 512 dw 14 × 14 × 512
            conv(512, 1024, 1, 1, 0),  # Conv / s1 1 × 1 × 512 × 1024 7 × 7 × 512
            conv_dw(1024, 1024, 3, 2, 1),  # Conv dw / s2 3 × 3 × 1024 dw 7 × 7 × 1024
            conv(1024, 1024, 1, 1, 0),  # Conv / s1 1 × 1 × 1024 × 1024 7 × 7 × 1024
            # Avg Pool / s1 Pool 7 × 7 7 × 7 × 1024
            # 여기선 feature map을 추출하는게 아니기에 padding값을 0으로 설정
            # nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        # FC / s1 1024 × 1000 1 × 1 × 1024
        # out_feature는 class의 갯수만큼
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(
            -1, 1024
        )  # -1 : batch 크기 자동화 설정, 1024 : feature channel를 평탄화 하는 사이즈
        x = self.fc(x)
        return x