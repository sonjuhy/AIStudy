from mobilenetv4_backbone import MobileNetV4ConvLargeBackbone
from neck import YOLO11Neck

import torch.nn as nn

class YOLO11_MobileNetV4(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = MobileNetV4ConvLargeBackbone()
        self.neck = YOLO11Neck(in_channels=self.backbone.out_channels)

        # Detection Head
        c3 = self.neck.out_channels
        self.detect_p3 = nn.Conv2d(c3[0], num_classes + 5, 1)
        self.detect_p4 = nn.Conv2d(c3[1], num_classes + 5, 1)
        self.detect_p5 = nn.Conv2d(c3[2], num_classes + 5, 1)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        return self.detect_p3(p3), self.detect_p4(p4), self.detect_p5(p5)
