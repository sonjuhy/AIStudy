import torch
import torch.nn as nn


# SEBlock 논문 주소 : https://arxiv.org/pdf/1709.01507
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super(SEBlock, self).__init__()
        reduced_channels = max(1, channels // reduction)  # 채널 축소 비율 (보통 4)

        self.pool = nn.AdaptiveAvgPool2d(1)  # [B, C, H, W] → [B, C, 1, 1]
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=True),
            nn.Hardsigmoid(inplace=True),  # MobileNetV3 논문 기준 활성함수
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)  # [B, C, 1, 1] → [B, C]
        y = self.fc(y).view(b, c, 1, 1)  # [B, C] → [B, C, 1, 1]
        return x * y.expand_as(x)  # 채널별 scale (⊗)


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    groups: int,
    non_linearity_type: str,
):

    if non_linearity_type == "HS":
        nl = nn.Hardswish(inplace=True)
    else:  # RE
        nl = nn.ReLU(inplace=True)
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,  # 편향 가중치 사용 여부. 뒤의 BatchNorm 에서 편향을 사용하기에 중복이라서 비활성화
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nl,
    )


class BottleNeck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int,
        expansion_channels: int,
        se_block_activated: bool,
        non_linearity_type: str,
    ):
        super().__init__()
        self.use_res = stride == 1 and in_channels == out_channels
        hidden_dim = expansion_channels if expansion_channels != 0 else in_channels
        pad = kernel_size // 2

        conv_layer = []
        if expansion_channels != in_channels:
            conv_layer.append(
                conv(
                    in_channels,
                    hidden_dim,
                    1,
                    1,
                    0,
                    1,
                    non_linearity_type,
                )
            )
        else:
            hidden_dim = in_channels

        dw_conv_list = [
            conv(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride,
                pad,
                hidden_dim,
                non_linearity_type,
            ),
        ]
        if se_block_activated:
            dw_conv_list.append(SEBlock(hidden_dim))

        conv_1x1_projection = [
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ]

        self.model = nn.Sequential(*conv_layer, *dw_conv_list, *conv_1x1_projection)

    def forward(self, x):
        out = self.model(x)
        return x + out if self.use_res else out


# MobileNet V3 논문 URL : https://arxiv.org/pdf/1905.02244
class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Input : 입력
        # Operator : 연산 종류
        # exp_size : DepSepConv 이전 1x1 Conv가 expansion 실행 할 횟수
        # #out : bot_neck의 out channle
        # SE : SE-block 적용 여부
        # NL : 사용 할  NonLinearity 함수 종류(HardSwish(Sigmoid * ReLU) or ReLU)
        # s : DepSepConv의 Depthwise layer에 적용할 stride
        # ※ HardSwish란? Sigmoid에 ReLU 를 곱한 형태. 무리수는 모바일에서 연산하기 힘들어 유리수로 처리하는 함수
        # ※ SEBlock: Squeeze, Excitation 연산을 통해 특징을 잘 갖고있는 채널들에게 가중치를 부여하는 Self-attention Block
        # SEBlock 논문 주소 : https://arxiv.org/pdf/1709.01507
        self.model = nn.Sequential(
            conv(3, 16, 3, 2, 1, 1, "HS"),  # 224 x 224 × 3 conv2d - 16 - HS 2
            BottleNeck(
                16, 16, 1, 3, 16, True, "RE"
            ),  # 112 x 112 × 16 bneck, 3x3 16 16 - RE 1
            BottleNeck(
                16, 24, 2, 3, 64, False, "RE"
            ),  # 112 x 112 × 16 bneck, 3x3 64 24 - RE 2
            BottleNeck(
                24, 24, 1, 3, 72, False, "RE"
            ),  # 56 x 56 × 24 bneck, 3x3 72 24 - RE 1
            BottleNeck(
                24, 40, 2, 5, 72, True, "RE"
            ),  # 56 x 56 × 24 bneck, 5x5 72 40 X RE 2
            BottleNeck(
                40, 40, 1, 5, 120, True, "RE"
            ),  # 28 x 28 × 40 bneck, 5x5 120 40 X RE 1
            BottleNeck(
                40, 40, 1, 5, 120, True, "RE"
            ),  # 28 x 28 × 40 bneck, 5x5 120 40 X RE 1
            BottleNeck(
                40, 80, 2, 3, 240, False, "HS"
            ),  # 28 x 28 × 40 bneck, 3x3 240 80 - HS 2
            BottleNeck(
                80, 80, 1, 3, 200, False, "HS"
            ),  # 14 x 14 × 80 bneck, 3x3 200 80 - HS 1
            BottleNeck(
                80, 80, 1, 3, 184, False, "HS"
            ),  # 14 x 14 × 80 bneck, 3x3 184 80 - HS 1
            BottleNeck(
                80, 80, 1, 3, 184, False, "HS"
            ),  # 14 x 14 × 80 bneck, 3x3 184 80 - HS 1
            BottleNeck(
                80, 112, 1, 3, 480, True, "HS"
            ),  # 14 x 14 × 80 bneck, 3x3 480 112 X HS 1
            BottleNeck(
                112, 112, 1, 3, 672, True, "HS"
            ),  # 14 x 14 × 112 bneck, 3x3 672 112 X HS 1
            BottleNeck(
                112, 160, 2, 5, 672, True, "HS"
            ),  # 14 x 14 × 112 bneck, 5x5 672 160 X HS 2
            BottleNeck(
                160, 160, 1, 5, 960, True, "HS"
            ),  # 7 × 7 × 160 bneck, 5x5 960 160 X HS 1
            BottleNeck(
                160, 160, 1, 5, 960, True, "HS"
            ),  # 7 × 7 × 160 bneck, 5x5 960 160 X HS 1
            conv(160, 960, 1, 1, 0, 1, "HS"),  # 7 × 7 × 160 conv2d, 1x1 - 960 - HS 1
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # 7 × 7 × 960 pool, 7x7 - - - - 1
        )
        # self.head = nn.Sequential(
        #     # 1 × 1 × 960 conv2d 1x1, NBN - 1280 - HS 1
        #     nn.Conv2d(960, 1280, 1, 1, 0, bias=True),
        #     nn.Hardswish(inplace=True),
        #     # 1 × 1 × 1280 conv2d 1x1, NBN - k - - 1
        #     nn.Conv2d(1280, num_classes, 1, 1, 0, bias=True),
        # )
        self.head = nn.Sequential(
            # 1 × 1 × 960 conv2d 1x1, NBN - 1280 - HS 1
            nn.Linear(960, 1280),
            nn.Hardswish(inplace=True),
            # 1 × 1 × 1280 conv2d 1x1, NBN - k - - 1
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        # x = x.flatten(1)
        return x


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.model = nn.Sequential(
            conv(3, 16, 3, 2, 1, 1, "HS"),  # 224 × 224 × 3 conv2d, 3x3 - 16 - HS 2
            BottleNeck(
                16, 16, 2, 3, 16, True, "RE"
            ),  # 112 × 112 × 16 bneck, 3x3 16 16 X RE 2
            BottleNeck(
                16, 24, 2, 3, 72, False, "RE"
            ),  # 56 × 56 × 16 bneck, 3x3 72 24 - RE 2
            BottleNeck(
                24, 24, 1, 3, 88, False, "RE"
            ),  # 28 × 28 × 24 bneck, 3x3 88 24 - RE 1
            BottleNeck(
                24, 40, 2, 5, 96, True, "HS"
            ),  # 28 × 28 × 24 bneck, 5x5 96 40 X HS 2
            BottleNeck(
                40, 40, 1, 5, 240, True, "HS"
            ),  # 14 × 14 × 40 bneck, 5x5 240 40 X HS 1
            BottleNeck(
                40, 40, 1, 5, 240, True, "HS"
            ),  # 14 × 14 × 40 bneck, 5x5 240 40 X HS 1
            BottleNeck(
                40, 48, 1, 5, 120, True, "HS"
            ),  # 14 × 14 × 40 bneck, 5x5 120 48 X HS 1
            BottleNeck(
                48, 48, 1, 5, 144, True, "HS"
            ),  # 14 × 14 × 48 bneck, 5x5 144 48 X HS 1
            BottleNeck(
                48, 96, 2, 5, 288, True, "HS"
            ),  # 14 × 14 × 48 bneck, 5x5 288 96 X HS 2
            BottleNeck(
                96, 96, 1, 5, 576, True, "HS"
            ),  # 7 × 2 × 96 bneck, 5x5 576 96 X HS 1
            BottleNeck(
                96, 96, 1, 5, 576, True, "HS"
            ),  # 7 × 2 × 96 bneck, 5x5 576 96 X HS 1
            conv(96, 576, 1, 1, 0, 1, "HS"),  # 7 × 2 × 96 conv2d, 1x1 - 576 X HS 1
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # 7 × 2 × 576 pool, 7x7 - - - - 1
        )
        self.head = nn.Sequential(
            # 1 × 2 × 576 conv2d 1x1, NBN - 1024 - HS 1
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            # 1 × 2 × 1024 conv2d 1x1, NBN - k - - 1
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x