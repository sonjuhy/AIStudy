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


class FFN(nn.Module):
    def __init__(
        self,
        channels: int,
        expand_ratio: float = 2.0,
        nl_type: str = "HS",
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = int(round(channels * expand_ratio))
        act = nn.Hardswish(inplace=True) if nl_type == "HS" else nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, 1, 0, bias=True),
            act,
            nn.Dropout(p=dropout, inplace=True) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden, channels, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        return x + self.block(x)


class FusedIB(nn.Module):  # FusedInvertedBottleneck
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int,
        expansion_channels: int,
        non_linearity_type: str = "HS",
    ):
        super().__init__()
        hidden_dim = expansion_channels
        pad = kernel_size // 2

        # 1) Fused Conv: expansion + DWConv 합친 일반 Conv
        self.fused_conv = conv(
            in_channels,
            hidden_dim,
            kernel_size,
            stride,
            pad,
            groups=1,  # 일반 conv
            non_linearity_type=non_linearity_type,
        )

        # 2) Projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.use_res = stride == 1 and in_channels == out_channels

    def forward(self, x):
        out = self.fused_conv(x)
        out = self.project(out)
        if self.use_res:
            out = x + out
        return out


class MobileMQA(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, dropout=0.0):
        super().__init__()
        """
        Self-Attension 을 모바일용으로 최적화한 내용
        """
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)  # [B, N, heads, dim_head]

        # Multi-query: K,V shared across heads
        k = k.mean(2, keepdim=True)
        v = v.mean(2, keepdim=True)

        attn = (q @ k.transpose(-2, -1)) / (C**0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


class UIB(nn.Module):  # UniversalInvertedBottleneck
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int,
        expansion_channels: int,
        se_block_activated: bool,
        ffn_block_activated: bool,
        pre_dw_activated: bool = False,  # ✅ DWConv before expansion
        extra_dw_activated: bool = False,  # ✅ Extra DW after main DW
        ffn_expand_ratio: float = 2.0,  # ✅ FFN 확장 비율
        ffn_dropout: float = 0.0,
        non_linearity_type: str = "HS",
    ):
        """
        InvertedBottleneck, Extra DW, DW, SE or FFN 를 한번에 관리 및 제어하는 범용 IB

        IB(InvertedBottleneck) : 기존 버전에서 사용해 온 bottleneck
        Extra DW : DW를 실행 후 stride=1로 설정 후 한번 더 실행(특징 강조)
        DW : DepthWise Conv
        SE or FFN : ConvNext 효과와 유사한 효과를 내는 블럭(연산량 낮추고 정확도 올림)
        """
        super().__init__()
        self.use_res = stride == 1 and in_channels == out_channels
        hidden_dim = expansion_channels if expansion_channels != 0 else in_channels
        pad = kernel_size // 2

        conv_layer = []
        # DWConv before expansion [Optional]
        if pre_dw_activated:
            conv_layer.append(
                conv(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride,
                    pad,
                    in_channels,
                    non_linearity_type,
                )
            )
            stride_for_main_dw = 1
        else:
            stride_for_main_dw = stride

        # 1x1 Conv expansion
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

        # 3x3 DWConv
        conv_layer.append(
            conv(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride_for_main_dw,
                pad,
                hidden_dim,
                non_linearity_type,
            )
        )

        # Extra DW [Optional]
        if extra_dw_activated:
            conv_layer.append(
                conv(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    1,
                    pad,
                    hidden_dim,
                    non_linearity_type,
                )
            )

        # SE or FFN [Optional]
        if se_block_activated:
            conv_layer.append(SEBlock(hidden_dim))
        if ffn_block_activated:
            conv_layer.append(
                FFN(hidden_dim, ffn_expand_ratio, non_linearity_type, ffn_dropout)
            )

        # 1x1 Conv proejction
        conv_layer.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        conv_layer.append(nn.BatchNorm2d(out_channels))

        self.model = nn.Sequential(*conv_layer)

    def forward(self, x):
        out = self.model(x)
        return x + out if self.use_res else out


# Mobile Net V4 논문 URL : https://arxiv.org/pdf/2404.10518
class MobileNetV4ConvSmall(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Input Block DW K1 DW K2 Expanded Dim Output Dim Stride
        self.model = nn.Sequential(
            conv(3, 32, 3, 2, 1, 1, "HS"),  # 224 × 224 × 3 Conv2D - 3 × 3 - 32 2
            FusedIB(32, 32, 2, 3, 32, "HS"),  # 112 × 122 × 32 FusedIB - 3 × 3 32 32 2
            FusedIB(32, 64, 2, 3, 96, "HS"),  # 56 × 56 × 32 FusedIB - 3 × 3 96 64 2
            UIB(
                64, 96, 2, 5, 192, True, False, False, True
            ),  # 28 × 28 × 64 ExtraDW 5 × 5 5 × 5 192 96 2
            UIB(
                96, 96, 1, 3, 192, True, False, False, False
            ),  # 14 × 14 × 96 IB - 3 × 3 192 96 1
            UIB(
                96, 96, 1, 3, 192, True, False, False, False
            ),  # 14 × 14 × 96 IB - 3 × 3 192 96 1
            UIB(
                96, 96, 1, 3, 192, True, False, False, False
            ),  # 14 × 14 × 96 IB - 3 × 3 192 96 1
            UIB(
                96, 96, 1, 3, 192, True, False, False, False
            ),  # 14 × 14 × 96 IB - 3 × 3 192 96 1
            UIB(
                96, 96, 1, 3, 384, False, True, False, False
            ),  # 14 × 14 × 96 ConvNext 3 × 3 - 384 96 1
            UIB(
                96, 128, 2, 3, 576, True, False, False, True
            ),  # 14 × 14 × 96 ExtraDW 3 × 3 3 × 3 576 128 2
            UIB(
                128, 128, 1, 5, 512, True, False, False, True
            ),  # 7 × 2 × 128 ExtraDW 5 × 5 5 × 5 512 128 1
            UIB(
                128, 128, 1, 5, 512, True, False, False, False
            ),  # 7 × 2 × 128 IB - 5 × 5 512 128 1
            UIB(
                128, 128, 1, 5, 384, True, False, False, False
            ),  # 7 × 2 × 128 IB - 5 × 5 384 128 1
            UIB(
                128, 128, 1, 3, 512, True, False, False, False
            ),  # 7 × 2 × 128 IB - 3 × 3 512 128 1
            UIB(
                128, 128, 1, 3, 512, True, False, False, False
            ),  # 7 × 2 × 128 IB - 3 × 3 512 128 1
            conv(128, 960, 1, 1, 0, 1, "HS"),  # 7 × 2 × 128 Conv2D - 1 × 1 - 960 1
        )

        # self.head = nn.Sequential(
        #     # 7 × 2 × 960 AvgPool - 7 × 7 - 960 1
        #     nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        #     # 1 × 2 × 960 Conv2D - 1 × 1 - 1280 1
        #     nn.Linear(960, 1280),
        #     nn.Hardswish(inplace=True),
        #     # 1 × 2 × 1280 Conv2D - 1 × 1 - 1000 1
        #     nn.Linear(1280, num_classes),
        #     nn.Hardswish(inplace=True),
        # )
        self.head = nn.Sequential(
            # 8 × 2 × 960 AvgPool - 8 × 8 - 960 1
            nn.AdaptiveAvgPool2d(1),
            # 1 × 2 × 960 Conv2D - 1 × 1 - 1280 1
            nn.Flatten(),
            nn.Linear(960, 1280),
            # 1 × 2 × 1280 Conv2D - 1 × 1 - 1000 1
            nn.Hardswish(inplace=True),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x


class MobileNetV4ConvMedium(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Input Block DW K1 DW K2 Expanded Dim Output Dim Stride
        self.model = nn.Sequential(
            conv(3, 32, 3, 2, 1, 1, "HS"),  # 256 × 256 × 3 Conv2D - 3 × 3 - 32 2
            FusedIB(32, 48, 2, 3, 128),  # 128 × 128 × 32 FusedIB - 3 × 3 128 48 2
            # 64² × 48 ExtraDW 3 × 3 5 × 5 192 80 2
            UIB(48, 80, 2, 3, 192, True, False, extra_dw_activated=True),
            # 32² × 80 ExtraDW 3 × 3 3 × 3 160 80 1
            UIB(80, 80, 1, 3, 160, True, False, extra_dw_activated=True),
            # 32² × 80 ExtraDW 3 × 3 5 × 5 480 160 2
            UIB(80, 160, 2, 5, 480, True, False, extra_dw_activated=True),
            # 16² × 160 ExtraDW 3 × 3 3 × 3 640 160 1
            UIB(160, 160, 1, 3, 640, True, False, extra_dw_activated=True),
            # 16² × 160 ExtraDW 3 × 3 3 × 3 640 160 1
            UIB(160, 160, 1, 3, 640, True, False, extra_dw_activated=True),
            # 16² × 160 ExtraDW 3 × 3 5 × 5 640 160 1
            UIB(160, 160, 1, 5, 640, True, False, extra_dw_activated=True),
            # 16² × 160 ExtraDW 3 × 3 3 × 3 640 160 1
            UIB(160, 160, 1, 3, 640, True, False, extra_dw_activated=True),
            # 16² × 160 ConvNext 3 × 3 - 640 160 1
            UIB(160, 160, 1, 3, 640, False, True, extra_dw_activated=False),
            # 16² × 160 FFN - - 320 160 1
            UIB(160, 160, 1, 3, 320, False, True, extra_dw_activated=False),
            # 16² × 160 ConvNext 3 × 3 - 640 160 1
            UIB(160, 160, 1, 3, 640, False, True, extra_dw_activated=False),
            # 8² × 256 ExtraDW 5 × 5 5 × 5 960 256 2
            UIB(160, 256, 2, 5, 960, True, False, extra_dw_activated=True),
            # 8² × 256 ExtraDW 5 × 5 5 × 5 1024 256 1
            UIB(256, 256, 1, 5, 1024, True, False, extra_dw_activated=True),
            # 8² × 256 ExtraDW 3 × 3 5 × 5 1024 256 1
            UIB(256, 256, 1, 5, 1024, True, False, extra_dw_activated=True),
            # 8² × 256 ExtraDW 3 × 3 5 × 5 1024 256 1
            UIB(256, 256, 1, 5, 1024, True, False, extra_dw_activated=True),
            # 8² × 256 FFN - - 1024 256 1
            UIB(256, 256, 1, 3, 1024, False, True, extra_dw_activated=False),
            # 8² × 256 ConvNext 3 × 3 - 1024 256 1
            UIB(256, 256, 1, 3, 1024, False, True, extra_dw_activated=False),
            # 8² × 256 ExtraDW 3 × 3 5 × 5 512 256 1
            UIB(256, 256, 1, 5, 512, True, False, extra_dw_activated=True),
            # 8² × 256 ExtraDW 5 × 5 5 × 5 1024 256 1
            UIB(256, 256, 1, 5, 1024, True, False, extra_dw_activated=True),
            # 8² × 256 FFN - - 1024 256 1
            UIB(256, 256, 1, 3, 1024, False, True, extra_dw_activated=False),
            # 8² × 256 FFN - - 1024 256 1
            UIB(256, 256, 1, 3, 1024, False, True, extra_dw_activated=False),
            # 8² × 256 ConvNext 5 × 5 - 512 256 1
            UIB(256, 256, 1, 5, 512, False, True, extra_dw_activated=False),
            # 8² × 256 Conv2D - 1 × 1 - 960 1
            conv(256, 960, 1, 1, 0, 1, "HS"),
        )

        self.head = nn.Sequential(
            # 8 × 2 × 960 AvgPool - 8 × 8 - 960 1
            nn.AdaptiveAvgPool2d(1),
            # 1 × 2 × 960 Conv2D - 1 × 1 - 1280 1
            nn.Flatten(),
            nn.Linear(960, 1280),
            # 1 × 2 × 1280 Conv2D - 1 × 1 - 1000 1
            nn.Hardswish(inplace=True),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        # x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class MobileNetV4HybridMedium(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            conv(3, 32, 3, 2, 1, 1, "HS"),  # 256 × 256 × 3 Conv2D - 3 × 3 - 32 2
            FusedIB(32, 48, 2, 3, 128, "HS"),  # 128 × 128 × 32 FusedIB - 3 × 3 128 48 2
            UIB(
                48, 80, 2, 3, 192, True, False, extra_dw_activated=True
            ),  # 64 × 64 × 48 ExtraDW 3 × 3 5 × 5 192 80 2
            UIB(
                80, 80, 1, 3, 160, True, False, extra_dw_activated=True
            ),  # 32 × 32 × 80 ExtraDW 3 × 3 3 × 3 160 80 1
            UIB(
                80, 160, 2, 5, 480, True, False, extra_dw_activated=True
            ),  # 32 × 32 × 80 ExtraDW 3 × 3 5 × 5 480 160 2
            UIB(
                160, 160, 1, 3, 640, True, False, extra_dw_activated=True
            ),  # 16 × 16 × 160 ExtraDW 3 × 3 3 × 3 640 160 1
            UIB(
                160, 160, 1, 3, 640, True, False, extra_dw_activated=True
            ),  # 16 × 16 × 160 ExtraDW 3 × 3 3 × 3 640 160 1
            UIB(
                160, 160, 1, 5, 640, True, False, extra_dw_activated=True
            ),  # 16 × 16 × 160 ExtraDW 3 × 3 5 × 5 640 160 1
            MobileMQA(160, num_heads=4),  # 16 × 16 × 160 Mobile-MQA - - - 160 1
            UIB(
                160, 160, 1, 3, 640, True, False, extra_dw_activated=True
            ),  # 16 × 16 × 160 ExtraDW 3 × 3 3 × 3 640 160 1
            MobileMQA(160, num_heads=4),  # 16 × 16 × 160 Mobile-MQA - - - 160 1
            UIB(
                160, 160, 1, 3, 640, False, True
            ),  # 16 × 16 × 160 ConvNext 3 × 3 - 640 160 1
            MobileMQA(160, num_heads=4),  # 16 × 16 × 160 Mobile-MQA - - - 160 1
            UIB(160, 160, 1, 3, 640, False, True),  # 16 × 16 × 160 FFN - - 640 160 1
            MobileMQA(160, num_heads=4),  # 16 × 16 × 160 Mobile-MQA - - - 160 1
            UIB(
                160, 160, 1, 3, 640, False, True
            ),  # 16 × 16 × 160 ConvNext 3 × 3 - 640 160 1
            UIB(
                160, 256, 2, 5, 960, True, False, extra_dw_activated=True
            ),  # 16 × 16 × 160 ExtraDW 5 × 5 5 × 5 960 256 2
            UIB(
                256, 256, 1, 5, 1024, True, False, extra_dw_activated=True
            ),  # 8 × 2 × 256 ExtraDW 5 × 5 5 × 5 1024 256 1
            UIB(
                256, 256, 1, 5, 1024, True, False, extra_dw_activated=True
            ),  # 8 × 2 × 256 ExtraDW 3 × 3 5 × 5 1024 256 1
            UIB(
                256, 256, 1, 5, 1024, True, False, extra_dw_activated=True
            ),  # 8 × 2 × 256 ExtraDW 3 × 3 5 × 5 1024 256 1
            UIB(256, 256, 1, 3, 1024, False, True),  # 8 × 2 × 256 FFN - - 1024 256 1
            UIB(
                256, 256, 1, 3, 1024, False, True
            ),  # 8 × 2 × 256 ConvNext 3 × 3 - 1024 256 1
            UIB(
                256, 256, 1, 5, 512, True, False, extra_dw_activated=True
            ),  # 8 × 2 × 256 ExtraDW 3 × 3 5 × 5 512 256 1
            MobileMQA(256, num_heads=4),  # 8 × 2 × 256 Mobile-MQA - - - 256 1
            UIB(
                256, 256, 1, 5, 1024, True, False, extra_dw_activated=True
            ),  # 8 × 2 × 256 ExtraDW 5 × 5 5 × 5 1024 256 1
            MobileMQA(256, num_heads=4),  # 8 × 2 × 256 Mobile-MQA - - - 256 1
            UIB(256, 256, 1, 3, 1024, False, True),  # 8 × 2 × 256 FFN - - 1024 256 1
            MobileMQA(256, num_heads=4),  # 8 × 2 × 256 Mobile-MQA - - - 256 1
            UIB(256, 256, 1, 3, 1024, False, True),  # 8 × 2 × 256 FFN - - 1024 256 1
            MobileMQA(256, num_heads=4),  # 8 × 2 × 256 Mobile-MQA - - - 256 1
            UIB(
                256, 256, 1, 5, 1024, False, True
            ),  # 8 × 2 × 256 ConvNext 5 × 5 - 1024 256 1
            conv(256, 960, 1, 1, 0, 1, "HS"),  # 8 × 2 × 256 Conv2D - 1 × 1 - 960 1
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 8 × 2 × 960 AvgPool - 8 × 8 - 960 1
            nn.Flatten(),
            nn.Linear(960, 1280),  # 1 × 2 × 960 Conv2D - 1 × 1 - 1280 1
            nn.Hardswish(inplace=True),
            nn.Linear(1280, num_classes),  # 1 × 2 × 1280 Conv2D - 1 × 1 - 1000 1
        )

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x


class MobileNetV4ConvLarge(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            conv(3, 24, 3, 2, 1, 1, "HS"),  # 384 × 384 × 2 × 3 Conv2D - 3 × 3 - 24 2
            FusedIB(
                24, 48, 2, 3, 96, "HS"
            ),  # 192 × 192 × 2 × 24 FusedIB - 3 × 3 96 48 2
            UIB(
                48, 96, 2, 3, 192, True, False, extra_dw_activated=True
            ),  # 96 × 96 × 2 × 48 ExtraDW 3 × 3 5 × 5 192 96 2
            UIB(
                96, 96, 1, 3, 384, True, False, extra_dw_activated=True
            ),  # 48 × 48 × 2 × 96 ExtraDW 3 × 3 3 × 3 384 96 1
            UIB(
                96, 192, 2, 3, 384, True, False, extra_dw_activated=True
            ),  # 48 × 48 × 2 × 96 ExtraDW 3 × 3 5 × 5 384 192 2
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 3 × 3 768 192 1
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 3 × 3 768 192 1
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 3 × 3 768 192 1
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 5 × 5 768 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            UIB(
                192, 192, 1, 3, 768, False, True
            ),  # 24 × 24 × 2 × 192 ConvNext 3 × 3 - 768 192 1
            UIB(
                192, 512, 2, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 5 × 5 768 512 2
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 3 × 3 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 3 × 3 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            conv(
                512, 960, 1, 1, 0, 1, "HS"
            ),  # 12 × 12 × 2 × 512 Conv2D - 1 × 1 - 960 1
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 12 × 12 × 2 × 960 AvgPool - 12 × 12 - 960 1
            nn.Flatten(),
            nn.Linear(960, 1280),  # 1 × 2 × 960 Conv2D - 1 × 1 - 1280 1
            nn.Hardswish(inplace=True),
            nn.Linear(1280, num_classes),  # 1 × 2 × 1280 Conv2D - 1 × 1 - 1000 1
        )

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x


class MobileNetV4HybridLarge(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            conv(3, 24, 3, 2, 1, 1, "HS"),  # 384 × 384 × 2 × 3 Conv2D - 3 × 3 - 24 2
            FusedIB(
                24, 48, 2, 3, 96, "HS"
            ),  # 192 × 192 × 2 × 24 FusedIB - 3 × 3 96 48 2
            UIB(
                48, 96, 2, 3, 192, True, False, extra_dw_activated=True
            ),  # 96 × 96 × 2 × 48 ExtraDW 3 × 3 5 × 5 192 96 2
            UIB(
                96, 96, 1, 3, 384, True, False, extra_dw_activated=True
            ),  # 48 × 48 × 2 × 96 ExtraDW 3 × 3 3 × 3 384 96 1
            UIB(
                96, 192, 2, 3, 384, True, False, extra_dw_activated=True
            ),  # 48 × 48 × 2 × 96 ExtraDW 3 × 3 5 × 5 384 192 2
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 3 × 3 768 192 1
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 3 × 3 768 192 1
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 3 × 3 768 192 1
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 5 × 5 768 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            MobileMQA(192, num_heads=4),  # 24 × 24 × 2 × 192 Mobile-MQA - - - 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            MobileMQA(192, num_heads=4),  # 24 × 24 × 2 × 192 Mobile-MQA - - - 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            MobileMQA(192, num_heads=4),  # 24 × 24 × 2 × 192 Mobile-MQA - - - 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            MobileMQA(192, num_heads=4),  # 24 × 24 × 2 × 192 Mobile-MQA - - - 192 1
            UIB(
                192, 192, 1, 3, 768, False, True
            ),  # 24 × 24 × 2 × 192 ConvNext 3 × 3 - 768 192 1
            UIB(
                192, 512, 2, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 5 × 5 768 512 2
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 3 × 3 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 3 × 3 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            MobileMQA(512, num_heads=4),  # 12 × 12 × 2 × 512 Mobile-MQA - - - 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            MobileMQA(512, num_heads=4),  # 12 × 12 × 2 × 512 Mobile-MQA - - - 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            MobileMQA(512, num_heads=4),  # 12 × 12 × 2 × 512 Mobile-MQA - - - 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            MobileMQA(512, num_heads=4),  # 12 × 12 × 2 × 512 Mobile-MQA - - - 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            conv(
                512, 960, 1, 1, 0, 1, "HS"
            ),  # 12 × 12 × 2 × 512 Conv2D - 1 × 1 - 960 1
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 12 × 12 × 2 × 960 AvgPool - 12 × 12 - 960 1
            nn.Flatten(),
            nn.Linear(960, 1280),  # 1 × 2 × 960 Conv2D - 1 × 1 - 1280 1
            nn.Hardswish(inplace=True),
            nn.Linear(1280, num_classes),  # 1 × 2 × 1280 Conv2D - 1 × 1 - 1000 1
        )

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x


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


class FFN(nn.Module):
    def __init__(
        self,
        channels: int,
        expand_ratio: float = 2.0,
        nl_type: str = "HS",
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = int(round(channels * expand_ratio))
        act = nn.Hardswish(inplace=True) if nl_type == "HS" else nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, 1, 0, bias=True),
            act,
            nn.Dropout(p=dropout, inplace=True) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden, channels, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        return x + self.block(x)


class FusedIB(nn.Module):  # FusedInvertedBottleneck
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int,
        expansion_channels: int,
        non_linearity_type: str = "HS",
    ):
        super().__init__()
        hidden_dim = expansion_channels
        pad = kernel_size // 2

        # 1) Fused Conv: expansion + DWConv 합친 일반 Conv
        self.fused_conv = conv(
            in_channels,
            hidden_dim,
            kernel_size,
            stride,
            pad,
            groups=1,  # 일반 conv
            non_linearity_type=non_linearity_type,
        )

        # 2) Projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.use_res = stride == 1 and in_channels == out_channels

    def forward(self, x):
        out = self.fused_conv(x)
        out = self.project(out)
        if self.use_res:
            out = x + out
        return out


class MobileMQA(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, dropout=0.0):
        super().__init__()
        """
        Self-Attension 을 모바일용으로 최적화한 내용
        """
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)  # [B, N, heads, dim_head]

        # Multi-query: K,V shared across heads
        k = k.mean(2, keepdim=True)
        v = v.mean(2, keepdim=True)

        attn = (q @ k.transpose(-2, -1)) / (C**0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


class UIB(nn.Module):  # UniversalInvertedBottleneck
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int,
        expansion_channels: int,
        se_block_activated: bool,
        ffn_block_activated: bool,
        pre_dw_activated: bool = False,  # ✅ DWConv before expansion
        extra_dw_activated: bool = False,  # ✅ Extra DW after main DW
        ffn_expand_ratio: float = 2.0,  # ✅ FFN 확장 비율
        ffn_dropout: float = 0.0,
        non_linearity_type: str = "HS",
    ):
        """
        InvertedBottleneck, Extra DW, DW, SE or FFN 를 한번에 관리 및 제어하는 범용 IB

        IB(InvertedBottleneck) : 기존 버전에서 사용해 온 bottleneck
        Extra DW : DW를 실행 후 stride=1로 설정 후 한번 더 실행(특징 강조)
        DW : DepthWise Conv
        SE or FFN : ConvNext 효과와 유사한 효과를 내는 블럭(연산량 낮추고 정확도 올림)
        """
        super().__init__()
        self.use_res = stride == 1 and in_channels == out_channels
        hidden_dim = expansion_channels if expansion_channels != 0 else in_channels
        pad = kernel_size // 2

        conv_layer = []
        # DWConv before expansion [Optional]
        if pre_dw_activated:
            conv_layer.append(
                conv(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride,
                    pad,
                    in_channels,
                    non_linearity_type,
                )
            )
            stride_for_main_dw = 1
        else:
            stride_for_main_dw = stride

        # 1x1 Conv expansion
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

        # 3x3 DWConv
        conv_layer.append(
            conv(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride_for_main_dw,
                pad,
                hidden_dim,
                non_linearity_type,
            )
        )

        # Extra DW [Optional]
        if extra_dw_activated:
            conv_layer.append(
                conv(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    1,
                    pad,
                    hidden_dim,
                    non_linearity_type,
                )
            )

        # SE or FFN [Optional]
        if se_block_activated:
            conv_layer.append(SEBlock(hidden_dim))
        if ffn_block_activated:
            conv_layer.append(
                FFN(hidden_dim, ffn_expand_ratio, non_linearity_type, ffn_dropout)
            )

        # 1x1 Conv proejction
        conv_layer.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        conv_layer.append(nn.BatchNorm2d(out_channels))

        self.model = nn.Sequential(*conv_layer)

    def forward(self, x):
        out = self.model(x)
        return x + out if self.use_res else out


# Mobile Net V4 논문 URL : https://arxiv.org/pdf/2404.10518
class MobileNetV4ConvSmall(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Input Block DW K1 DW K2 Expanded Dim Output Dim Stride
        self.model = nn.Sequential(
            conv(3, 32, 3, 2, 1, 1, "HS"),  # 224 × 224 × 3 Conv2D - 3 × 3 - 32 2
            FusedIB(32, 32, 2, 3, 32, "HS"),  # 112 × 122 × 32 FusedIB - 3 × 3 32 32 2
            FusedIB(32, 64, 2, 3, 96, "HS"),  # 56 × 56 × 32 FusedIB - 3 × 3 96 64 2
            UIB(
                64, 96, 2, 5, 192, True, False, False, True
            ),  # 28 × 28 × 64 ExtraDW 5 × 5 5 × 5 192 96 2
            UIB(
                96, 96, 1, 3, 192, True, False, False, False
            ),  # 14 × 14 × 96 IB - 3 × 3 192 96 1
            UIB(
                96, 96, 1, 3, 192, True, False, False, False
            ),  # 14 × 14 × 96 IB - 3 × 3 192 96 1
            UIB(
                96, 96, 1, 3, 192, True, False, False, False
            ),  # 14 × 14 × 96 IB - 3 × 3 192 96 1
            UIB(
                96, 96, 1, 3, 192, True, False, False, False
            ),  # 14 × 14 × 96 IB - 3 × 3 192 96 1
            UIB(
                96, 96, 1, 3, 384, False, True, False, False
            ),  # 14 × 14 × 96 ConvNext 3 × 3 - 384 96 1
            UIB(
                96, 128, 2, 3, 576, True, False, False, True
            ),  # 14 × 14 × 96 ExtraDW 3 × 3 3 × 3 576 128 2
            UIB(
                128, 128, 1, 5, 512, True, False, False, True
            ),  # 7 × 2 × 128 ExtraDW 5 × 5 5 × 5 512 128 1
            UIB(
                128, 128, 1, 5, 512, True, False, False, False
            ),  # 7 × 2 × 128 IB - 5 × 5 512 128 1
            UIB(
                128, 128, 1, 5, 384, True, False, False, False
            ),  # 7 × 2 × 128 IB - 5 × 5 384 128 1
            UIB(
                128, 128, 1, 3, 512, True, False, False, False
            ),  # 7 × 2 × 128 IB - 3 × 3 512 128 1
            UIB(
                128, 128, 1, 3, 512, True, False, False, False
            ),  # 7 × 2 × 128 IB - 3 × 3 512 128 1
            conv(128, 960, 1, 1, 0, 1, "HS"),  # 7 × 2 × 128 Conv2D - 1 × 1 - 960 1
        )

        # self.head = nn.Sequential(
        #     # 7 × 2 × 960 AvgPool - 7 × 7 - 960 1
        #     nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        #     # 1 × 2 × 960 Conv2D - 1 × 1 - 1280 1
        #     nn.Linear(960, 1280),
        #     nn.Hardswish(inplace=True),
        #     # 1 × 2 × 1280 Conv2D - 1 × 1 - 1000 1
        #     nn.Linear(1280, num_classes),
        #     nn.Hardswish(inplace=True),
        # )
        self.head = nn.Sequential(
            # 8 × 2 × 960 AvgPool - 8 × 8 - 960 1
            nn.AdaptiveAvgPool2d(1),
            # 1 × 2 × 960 Conv2D - 1 × 1 - 1280 1
            nn.Flatten(),
            nn.Linear(960, 1280),
            # 1 × 2 × 1280 Conv2D - 1 × 1 - 1000 1
            nn.Hardswish(inplace=True),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x


class MobileNetV4ConvMedium(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Input Block DW K1 DW K2 Expanded Dim Output Dim Stride
        self.model = nn.Sequential(
            conv(3, 32, 3, 2, 1, 1, "HS"),  # 256 × 256 × 3 Conv2D - 3 × 3 - 32 2
            FusedIB(32, 48, 2, 3, 128),  # 128 × 128 × 32 FusedIB - 3 × 3 128 48 2
            # 64² × 48 ExtraDW 3 × 3 5 × 5 192 80 2
            UIB(48, 80, 2, 3, 192, True, False, extra_dw_activated=True),
            # 32² × 80 ExtraDW 3 × 3 3 × 3 160 80 1
            UIB(80, 80, 1, 3, 160, True, False, extra_dw_activated=True),
            # 32² × 80 ExtraDW 3 × 3 5 × 5 480 160 2
            UIB(80, 160, 2, 5, 480, True, False, extra_dw_activated=True),
            # 16² × 160 ExtraDW 3 × 3 3 × 3 640 160 1
            UIB(160, 160, 1, 3, 640, True, False, extra_dw_activated=True),
            # 16² × 160 ExtraDW 3 × 3 3 × 3 640 160 1
            UIB(160, 160, 1, 3, 640, True, False, extra_dw_activated=True),
            # 16² × 160 ExtraDW 3 × 3 5 × 5 640 160 1
            UIB(160, 160, 1, 5, 640, True, False, extra_dw_activated=True),
            # 16² × 160 ExtraDW 3 × 3 3 × 3 640 160 1
            UIB(160, 160, 1, 3, 640, True, False, extra_dw_activated=True),
            # 16² × 160 ConvNext 3 × 3 - 640 160 1
            UIB(160, 160, 1, 3, 640, False, True, extra_dw_activated=False),
            # 16² × 160 FFN - - 320 160 1
            UIB(160, 160, 1, 3, 320, False, True, extra_dw_activated=False),
            # 16² × 160 ConvNext 3 × 3 - 640 160 1
            UIB(160, 160, 1, 3, 640, False, True, extra_dw_activated=False),
            # 8² × 256 ExtraDW 5 × 5 5 × 5 960 256 2
            UIB(160, 256, 2, 5, 960, True, False, extra_dw_activated=True),
            # 8² × 256 ExtraDW 5 × 5 5 × 5 1024 256 1
            UIB(256, 256, 1, 5, 1024, True, False, extra_dw_activated=True),
            # 8² × 256 ExtraDW 3 × 3 5 × 5 1024 256 1
            UIB(256, 256, 1, 5, 1024, True, False, extra_dw_activated=True),
            # 8² × 256 ExtraDW 3 × 3 5 × 5 1024 256 1
            UIB(256, 256, 1, 5, 1024, True, False, extra_dw_activated=True),
            # 8² × 256 FFN - - 1024 256 1
            UIB(256, 256, 1, 3, 1024, False, True, extra_dw_activated=False),
            # 8² × 256 ConvNext 3 × 3 - 1024 256 1
            UIB(256, 256, 1, 3, 1024, False, True, extra_dw_activated=False),
            # 8² × 256 ExtraDW 3 × 3 5 × 5 512 256 1
            UIB(256, 256, 1, 5, 512, True, False, extra_dw_activated=True),
            # 8² × 256 ExtraDW 5 × 5 5 × 5 1024 256 1
            UIB(256, 256, 1, 5, 1024, True, False, extra_dw_activated=True),
            # 8² × 256 FFN - - 1024 256 1
            UIB(256, 256, 1, 3, 1024, False, True, extra_dw_activated=False),
            # 8² × 256 FFN - - 1024 256 1
            UIB(256, 256, 1, 3, 1024, False, True, extra_dw_activated=False),
            # 8² × 256 ConvNext 5 × 5 - 512 256 1
            UIB(256, 256, 1, 5, 512, False, True, extra_dw_activated=False),
            # 8² × 256 Conv2D - 1 × 1 - 960 1
            conv(256, 960, 1, 1, 0, 1, "HS"),
        )

        self.head = nn.Sequential(
            # 8 × 2 × 960 AvgPool - 8 × 8 - 960 1
            nn.AdaptiveAvgPool2d(1),
            # 1 × 2 × 960 Conv2D - 1 × 1 - 1280 1
            nn.Flatten(),
            nn.Linear(960, 1280),
            # 1 × 2 × 1280 Conv2D - 1 × 1 - 1000 1
            nn.Hardswish(inplace=True),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        # x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class MobileNetV4HybridMedium(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            conv(3, 32, 3, 2, 1, 1, "HS"),  # 256 × 256 × 3 Conv2D - 3 × 3 - 32 2
            FusedIB(32, 48, 2, 3, 128, "HS"),  # 128 × 128 × 32 FusedIB - 3 × 3 128 48 2
            UIB(
                48, 80, 2, 3, 192, True, False, extra_dw_activated=True
            ),  # 64 × 64 × 48 ExtraDW 3 × 3 5 × 5 192 80 2
            UIB(
                80, 80, 1, 3, 160, True, False, extra_dw_activated=True
            ),  # 32 × 32 × 80 ExtraDW 3 × 3 3 × 3 160 80 1
            UIB(
                80, 160, 2, 5, 480, True, False, extra_dw_activated=True
            ),  # 32 × 32 × 80 ExtraDW 3 × 3 5 × 5 480 160 2
            UIB(
                160, 160, 1, 3, 640, True, False, extra_dw_activated=True
            ),  # 16 × 16 × 160 ExtraDW 3 × 3 3 × 3 640 160 1
            UIB(
                160, 160, 1, 3, 640, True, False, extra_dw_activated=True
            ),  # 16 × 16 × 160 ExtraDW 3 × 3 3 × 3 640 160 1
            UIB(
                160, 160, 1, 5, 640, True, False, extra_dw_activated=True
            ),  # 16 × 16 × 160 ExtraDW 3 × 3 5 × 5 640 160 1
            MobileMQA(160, num_heads=4),  # 16 × 16 × 160 Mobile-MQA - - - 160 1
            UIB(
                160, 160, 1, 3, 640, True, False, extra_dw_activated=True
            ),  # 16 × 16 × 160 ExtraDW 3 × 3 3 × 3 640 160 1
            MobileMQA(160, num_heads=4),  # 16 × 16 × 160 Mobile-MQA - - - 160 1
            UIB(
                160, 160, 1, 3, 640, False, True
            ),  # 16 × 16 × 160 ConvNext 3 × 3 - 640 160 1
            MobileMQA(160, num_heads=4),  # 16 × 16 × 160 Mobile-MQA - - - 160 1
            UIB(160, 160, 1, 3, 640, False, True),  # 16 × 16 × 160 FFN - - 640 160 1
            MobileMQA(160, num_heads=4),  # 16 × 16 × 160 Mobile-MQA - - - 160 1
            UIB(
                160, 160, 1, 3, 640, False, True
            ),  # 16 × 16 × 160 ConvNext 3 × 3 - 640 160 1
            UIB(
                160, 256, 2, 5, 960, True, False, extra_dw_activated=True
            ),  # 16 × 16 × 160 ExtraDW 5 × 5 5 × 5 960 256 2
            UIB(
                256, 256, 1, 5, 1024, True, False, extra_dw_activated=True
            ),  # 8 × 2 × 256 ExtraDW 5 × 5 5 × 5 1024 256 1
            UIB(
                256, 256, 1, 5, 1024, True, False, extra_dw_activated=True
            ),  # 8 × 2 × 256 ExtraDW 3 × 3 5 × 5 1024 256 1
            UIB(
                256, 256, 1, 5, 1024, True, False, extra_dw_activated=True
            ),  # 8 × 2 × 256 ExtraDW 3 × 3 5 × 5 1024 256 1
            UIB(256, 256, 1, 3, 1024, False, True),  # 8 × 2 × 256 FFN - - 1024 256 1
            UIB(
                256, 256, 1, 3, 1024, False, True
            ),  # 8 × 2 × 256 ConvNext 3 × 3 - 1024 256 1
            UIB(
                256, 256, 1, 5, 512, True, False, extra_dw_activated=True
            ),  # 8 × 2 × 256 ExtraDW 3 × 3 5 × 5 512 256 1
            MobileMQA(256, num_heads=4),  # 8 × 2 × 256 Mobile-MQA - - - 256 1
            UIB(
                256, 256, 1, 5, 1024, True, False, extra_dw_activated=True
            ),  # 8 × 2 × 256 ExtraDW 5 × 5 5 × 5 1024 256 1
            MobileMQA(256, num_heads=4),  # 8 × 2 × 256 Mobile-MQA - - - 256 1
            UIB(256, 256, 1, 3, 1024, False, True),  # 8 × 2 × 256 FFN - - 1024 256 1
            MobileMQA(256, num_heads=4),  # 8 × 2 × 256 Mobile-MQA - - - 256 1
            UIB(256, 256, 1, 3, 1024, False, True),  # 8 × 2 × 256 FFN - - 1024 256 1
            MobileMQA(256, num_heads=4),  # 8 × 2 × 256 Mobile-MQA - - - 256 1
            UIB(
                256, 256, 1, 5, 1024, False, True
            ),  # 8 × 2 × 256 ConvNext 5 × 5 - 1024 256 1
            conv(256, 960, 1, 1, 0, 1, "HS"),  # 8 × 2 × 256 Conv2D - 1 × 1 - 960 1
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 8 × 2 × 960 AvgPool - 8 × 8 - 960 1
            nn.Flatten(),
            nn.Linear(960, 1280),  # 1 × 2 × 960 Conv2D - 1 × 1 - 1280 1
            nn.Hardswish(inplace=True),
            nn.Linear(1280, num_classes),  # 1 × 2 × 1280 Conv2D - 1 × 1 - 1000 1
        )

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x


class MobileNetV4ConvLarge(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            conv(3, 24, 3, 2, 1, 1, "HS"),  # 384 × 384 × 2 × 3 Conv2D - 3 × 3 - 24 2
            FusedIB(
                24, 48, 2, 3, 96, "HS"
            ),  # 192 × 192 × 2 × 24 FusedIB - 3 × 3 96 48 2
            UIB(
                48, 96, 2, 3, 192, True, False, extra_dw_activated=True
            ),  # 96 × 96 × 2 × 48 ExtraDW 3 × 3 5 × 5 192 96 2
            UIB(
                96, 96, 1, 3, 384, True, False, extra_dw_activated=True
            ),  # 48 × 48 × 2 × 96 ExtraDW 3 × 3 3 × 3 384 96 1
            UIB(
                96, 192, 2, 3, 384, True, False, extra_dw_activated=True
            ),  # 48 × 48 × 2 × 96 ExtraDW 3 × 3 5 × 5 384 192 2
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 3 × 3 768 192 1
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 3 × 3 768 192 1
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 3 × 3 768 192 1
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 5 × 5 768 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            UIB(
                192, 192, 1, 3, 768, False, True
            ),  # 24 × 24 × 2 × 192 ConvNext 3 × 3 - 768 192 1
            UIB(
                192, 512, 2, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 5 × 5 768 512 2
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 3 × 3 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 3 × 3 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            conv(
                512, 960, 1, 1, 0, 1, "HS"
            ),  # 12 × 12 × 2 × 512 Conv2D - 1 × 1 - 960 1
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 12 × 12 × 2 × 960 AvgPool - 12 × 12 - 960 1
            nn.Flatten(),
            nn.Linear(960, 1280),  # 1 × 2 × 960 Conv2D - 1 × 1 - 1280 1
            nn.Hardswish(inplace=True),
            nn.Linear(1280, num_classes),  # 1 × 2 × 1280 Conv2D - 1 × 1 - 1000 1
        )

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x


class MobileNetV4HybridLarge(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            conv(3, 24, 3, 2, 1, 1, "HS"),  # 384 × 384 × 2 × 3 Conv2D - 3 × 3 - 24 2
            FusedIB(
                24, 48, 2, 3, 96, "HS"
            ),  # 192 × 192 × 2 × 24 FusedIB - 3 × 3 96 48 2
            UIB(
                48, 96, 2, 3, 192, True, False, extra_dw_activated=True
            ),  # 96 × 96 × 2 × 48 ExtraDW 3 × 3 5 × 5 192 96 2
            UIB(
                96, 96, 1, 3, 384, True, False, extra_dw_activated=True
            ),  # 48 × 48 × 2 × 96 ExtraDW 3 × 3 3 × 3 384 96 1
            UIB(
                96, 192, 2, 3, 384, True, False, extra_dw_activated=True
            ),  # 48 × 48 × 2 × 96 ExtraDW 3 × 3 5 × 5 384 192 2
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 3 × 3 768 192 1
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 3 × 3 768 192 1
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 3 × 3 768 192 1
            UIB(
                192, 192, 1, 3, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 3 × 3 5 × 5 768 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            MobileMQA(192, num_heads=4),  # 24 × 24 × 2 × 192 Mobile-MQA - - - 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            MobileMQA(192, num_heads=4),  # 24 × 24 × 2 × 192 Mobile-MQA - - - 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            MobileMQA(192, num_heads=4),  # 24 × 24 × 2 × 192 Mobile-MQA - - - 192 1
            UIB(
                192, 192, 1, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 3 × 3 768 192 1
            MobileMQA(192, num_heads=4),  # 24 × 24 × 2 × 192 Mobile-MQA - - - 192 1
            UIB(
                192, 192, 1, 3, 768, False, True
            ),  # 24 × 24 × 2 × 192 ConvNext 3 × 3 - 768 192 1
            UIB(
                192, 512, 2, 5, 768, True, False, extra_dw_activated=True
            ),  # 24 × 24 × 2 × 192 ExtraDW 5 × 5 5 × 5 768 512 2
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 3 × 3 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 3 × 3 2048 512 1
            UIB(
                512, 512, 1, 5, 2048, True, False, extra_dw_activated=True
            ),  # 12 × 12 × 2 × 512 ExtraDW 5 × 5 5 × 5 2048 512 1
            MobileMQA(512, num_heads=4),  # 12 × 12 × 2 × 512 Mobile-MQA - - - 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            MobileMQA(512, num_heads=4),  # 12 × 12 × 2 × 512 Mobile-MQA - - - 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            MobileMQA(512, num_heads=4),  # 12 × 12 × 2 × 512 Mobile-MQA - - - 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            MobileMQA(512, num_heads=4),  # 12 × 12 × 2 × 512 Mobile-MQA - - - 512 1
            UIB(
                512, 512, 1, 5, 2048, False, True
            ),  # 12 × 12 × 2 × 512 ConvNext 5 × 5 - 2048 512 1
            conv(
                512, 960, 1, 1, 0, 1, "HS"
            ),  # 12 × 12 × 2 × 512 Conv2D - 1 × 1 - 960 1
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 12 × 12 × 2 × 960 AvgPool - 12 × 12 - 960 1
            nn.Flatten(),
            nn.Linear(960, 1280),  # 1 × 2 × 960 Conv2D - 1 × 1 - 1280 1
            nn.Hardswish(inplace=True),
            nn.Linear(1280, num_classes),  # 1 × 2 × 1280 Conv2D - 1 × 1 - 1000 1
        )

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x
