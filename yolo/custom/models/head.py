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
