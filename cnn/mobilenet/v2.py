import torch.nn as nn


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


class BottleNeck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ):
        super().__init__()
        # Input Operator Output
        # h × w × k | 1x1 conv2d , ReLU6 | h × w × (tk)
        # h × w × tk | 3x3 dwise s=s, ReLU6 | h/s × w/s × (tk)
        # h/s × w/s × (tk) | linear 1x1 conv2d | h/s × w/s × k'
        hidden_dim = in_channels * expand_ratio
        self.use_res = stride == 1 and in_channels == out_channels

        pointwise_1x1_expansion_layers = []
        if expand_ratio != 1:
            pointwise_1x1_expansion_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            )
            pointwise_1x1_expansion_layers.append(
                nn.BatchNorm2d(num_features=hidden_dim)
            )
            pointwise_1x1_expansion_layers.append(nn.ReLU6(inplace=True))

        depthwise_3x3_conv_layers = []
        depthwise_3x3_conv_layers.append(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False,
            )
        )
        depthwise_3x3_conv_layers.append(nn.BatchNorm2d(num_features=hidden_dim))
        depthwise_3x3_conv_layers.append(nn.ReLU6(inplace=True))

        pointwise_1x1_projection_layers = []
        pointwise_1x1_projection_layers.append(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        )
        pointwise_1x1_projection_layers.append(
            nn.BatchNorm2d(num_features=out_channels)
        )

        self.residual = nn.Sequential(
            *pointwise_1x1_expansion_layers,
            *depthwise_3x3_conv_layers,
            *pointwise_1x1_projection_layers,
        )

    def forward(self, x):
        out = self.residual(x)
        return x + out if self.use_res else out


# MobileNet V2 논문 URL : https://arxiv.org/pdf/1801.04381
class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # input, Operator, expansion factor, output, repeated times, stride
        # All aptial conv use 3 x 3 kernels
        # ReLU로 소실되는 데이터를 최소화 하고자, BottleNeck에서 기존 데이터(ReLU 적용 전)을 합성

        self.model = nn.Sequential(
            conv(3, 32, 3, 2, 1),  # 224 x 224 × 3 conv2d - 32 1 2
            BottleNeck(32, 16, 1, 1),  # 112 x 112 × 32 bottleneck 1 16 1 1
            # 112 x 112 × 16 bottleneck 6 24 2 2
            BottleNeck(16, 24, 2, 6),
            BottleNeck(24, 24, 1, 6),
            # 56 x 56 × 24 bottleneck 6 32 3 2
            BottleNeck(24, 32, 2, 6),
            BottleNeck(32, 32, 1, 6),
            BottleNeck(32, 32, 1, 6),
            # 28 x 28 × 32 bottleneck 6 64 4 2
            BottleNeck(32, 64, 2, 6),
            BottleNeck(64, 64, 1, 6),
            BottleNeck(64, 64, 1, 6),
            BottleNeck(64, 64, 1, 6),
            # 14 x 14 × 64 bottleneck 6 96 3 1
            BottleNeck(64, 96, 1, 6),
            BottleNeck(96, 96, 1, 6),
            BottleNeck(96, 96, 1, 6),
            # 14 x 14 × 96 bottleneck 6 160 3 2
            BottleNeck(96, 160, 2, 6),
            BottleNeck(160, 160, 1, 6),
            BottleNeck(160, 160, 1, 6),
            # 7 x 7 × 160 bottleneck 6 320 1 1
            BottleNeck(160, 320, 1, 6),
            # 7 x 7 × 320 conv2d 1x1 - 1280 1 1
            conv(320, 1280, 1, 1, 0),
            nn.AdaptiveAvgPool2d(
                output_size=(1, 1)
            ),  # 7 x 7 × 1280 avgpool 7x7 - - 1 -
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(1280, num_classes)
        )  # 1 × 1 × 1280 conv2d 1x1 - k -

    def forward(self, x):
        x = self.model(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x