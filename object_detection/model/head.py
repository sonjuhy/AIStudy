import math
from object_detection.model.block import StandardConv, DWConv

import torch
import torch.nn as nn


class DFLHead(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        reg_max: int = 16,
        in_channels: tuple = (256, 512, 512),
    ):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.no = self.nc + self.reg_max * 4  # 4 coordinates × reg_max bins
        self.stride = torch.tensor([8, 16, 32])  # 각 스케일 stride

        # detection conv heads
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for c in in_channels:
            self.cls_convs.append(
                nn.Sequential(
                    nn.Conv2d(c, c, 3, 1, 1, bias=True),
                    nn.BatchNorm2d(c),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(c, num_classes, 1),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    nn.Conv2d(c, c, 3, 1, 1, bias=True),
                    nn.BatchNorm2d(c),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(c, 4 * reg_max, 1),
                )
            )

        self.initialize_biases()

    def initialize_biases(self):
        # 초기 bias 세팅 (YOLOv8 방식)
        for cls_conv in self.cls_convs:
            b = cls_conv[-1].bias.view(self.nc)
            b.data.fill_(-4.5)
            cls_conv[-1].bias = torch.nn.Parameter(b)
        for reg_conv in self.reg_convs:
            b = reg_conv[-1].bias.view(4 * self.reg_max)
            b.data.zero_()
            reg_conv[-1].bias = torch.nn.Parameter(b)

    def forward(self, feats):
        """
        feats: (P3, P4, P5)
        return: [preds_p3, preds_p4, preds_p5]
                각 preds = [B, no, H, W]
        """
        outputs = []
        for i, x in enumerate(feats):
            cls_pred = self.cls_convs[i](x)
            reg_pred = self.reg_convs[i](x)
            out = torch.cat((reg_pred, cls_pred), 1)
            outputs.append(out)
        return outputs


class StandardDecoupledHead(nn.Module):
    def __init__(
        self, num_classes: int = 80, in_channels: tuple[int, int, int] = (256, 512, 512)
    ) -> None:
        super().__init__()
        self.num_classes: int = num_classes
        self.cls_heads: nn.ModuleList = nn.ModuleList()
        self.box_heads: nn.ModuleList = nn.ModuleList()

        for in_c in in_channels:
            self.cls_heads.append(
                nn.Sequential(
                    StandardConv(in_c, in_c, kernel_size=3, padding=1),
                    nn.Conv2d(in_c, num_classes, kernel_size=1),
                )
            )
            # DFL 제거: 무조건 4채널(l, t, r, b) 출력
            self.box_heads.append(
                nn.Sequential(
                    StandardConv(in_c, in_c, kernel_size=3, padding=1),
                    nn.Conv2d(in_c, 4, kernel_size=1),
                )
            )
        self._initialize_biases()

    def _initialize_biases(self) -> None:
        """분류 헤드의 편향을 초기화하여 초기 배경(Background) Loss 폭발을 방지합니다."""
        prior_prob: float = 0.01
        bias_value: float = -math.log((1 - prior_prob) / prior_prob)  # 약 -4.59

        for cls_head in self.cls_heads:
            # cls_heads 내의 마지막 Conv2d 레이어 편향(bias)에 적용
            cls_head[-1].bias.data.fill_(bias_value)

    def forward(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> list[torch.Tensor]:
        preds: list[torch.Tensor] = []
        for i, x in enumerate(features):
            box_out: torch.Tensor = self.box_heads[i](x)
            cls_out: torch.Tensor = self.cls_heads[i](x)
            preds.append(torch.cat([box_out, cls_out], dim=1))
        return preds


class DWConvDecoupledHead(nn.Module):
    def __init__(
        self, num_classes: int = 80, in_channels: tuple[int, int, int] = (256, 512, 512)
    ) -> None:
        super().__init__()
        self.num_classes: int = num_classes
        self.cls_heads: nn.ModuleList = nn.ModuleList()
        self.box_heads: nn.ModuleList = nn.ModuleList()

        for in_c in in_channels:
            self.cls_heads.append(
                nn.Sequential(
                    DWConv(in_c, in_c, kernel_size=3, padding=1),
                    nn.Conv2d(in_c, num_classes, kernel_size=1),
                )
            )
            self.box_heads.append(
                nn.Sequential(
                    DWConv(in_c, in_c, kernel_size=3, padding=1),
                    nn.Conv2d(in_c, 4, kernel_size=1),
                )
            )
        self._initialize_biases()

    def _initialize_biases(self) -> None:
        """분류 헤드의 편향을 초기화하여 초기 배경(Background) Loss 폭발을 방지합니다."""
        prior_prob: float = 0.01
        bias_value: float = -math.log((1 - prior_prob) / prior_prob)  # 약 -4.59

        for cls_head in self.cls_heads:
            # cls_heads 내의 마지막 Conv2d 레이어 편향(bias)에 적용
            cls_head[-1].bias.data.fill_(bias_value)

    def forward(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> list[torch.Tensor]:
        preds: list[torch.Tensor] = []
        for i, x in enumerate(features):
            box_out: torch.Tensor = self.box_heads[i](x)
            cls_out: torch.Tensor = self.cls_heads[i](x)
            preds.append(torch.cat([box_out, cls_out], dim=1))
        return preds
