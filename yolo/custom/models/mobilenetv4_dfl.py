from mobilenetv4_backbone import (
    MobileNetV4ConvSmallBackbone,
    MobileNetV4ConvLargeBackbone,
    MobileNetV4ConvMediumBackbone,
)
from neck import YOLO11DynamicNeck
from head import DFLHead
from yolo.custom.util.utils import dist2bbox, nms

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class YOLO11_MobileNetV4_DFL(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        reg_max: int = 16,
    ):
        super().__init__()
        # self.backbone = MobileNetV4ConvLargeBackbone()
        self.backbone = MobileNetV4ConvMediumBackbone()
        # self.backbone = MobileNetV4ConvSmallBackbone()
        
        self.neck = YOLO11DynamicNeck(
            in_channels=self.backbone.out_channels, width=1.00
        )
        self.head = DFLHead(
            num_classes=num_classes, reg_max=reg_max, in_channels=self.neck.out_channels
        )

        self.nc = num_classes
        self.reg_max = reg_max
        self.stride = self.head.stride
        self.args = type("obj", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5})()

        self.model = nn.ModuleList([self.backbone, self.neck, self.head])

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        preds = self.head((p3, p4, p5))
        return preds

    @torch.no_grad()
    def predict(self, x, orig_size=None, conf_thres=0.25, iou_thres=0.45):
        """
        x: [B,3,640,640]  (학습과 동일 전처리)
        orig_size: (H_orig, W_orig)  # 원본 이미지에 그릴 때 필요
        """
        self.eval()
        preds = self.forward(x)
        bs, device = x.shape[0], x.device
        results = []

        proj = torch.arange(self.reg_max, device=device).view(1, 1, -1, 1, 1)

        for b in range(bs):
            boxes_all, scores_all, classes_all = [], [], []

            for i, p in enumerate(preds):  # P3,P4,P5
                stride = self.stride[i]
                _, _, h, w = p.shape
                p = p[b : b + 1]  # [1,C,H,W]

                # split cls / dfl
                pred_distri, pred_cls = torch.split(
                    p, [4 * self.reg_max, self.nc], dim=1
                )

                # DFL decode (l,t,r,b) in pixels
                pred_distri = pred_distri.view(1, 4, self.reg_max, h, w).softmax(2)
                pred_dist = (pred_distri * proj).sum(2) * stride  # [1,4,H,W]

                # === FIX 1: grid는 셀 중심 (i+0.5, j+0.5) ===
                gy, gx = torch.meshgrid(
                    torch.arange(h, device=device),
                    torch.arange(w, device=device),
                    indexing="ij",
                )
                anchor_points = torch.stack(
                    ((gx + 0.5) * stride, (gy + 0.5) * stride), dim=-1
                ).reshape(
                    -1, 2
                )  # [H*W,2]

                # (l,t,r,b) + center -> (x1,y1,x2,y2)
                boxes = dist2bbox(
                    pred_dist.permute(0, 2, 3, 1).reshape(-1, 4),
                    anchor_points,
                    xywh=False,
                )  # [H*W,4] in input(640) scale

                # cls
                scores = pred_cls.sigmoid().permute(0, 2, 3, 1).reshape(-1, self.nc)
                conf, cls = scores.max(1)

                mask = conf > conf_thres
                if mask.any():
                    boxes_all.append(boxes[mask])
                    scores_all.append(conf[mask])
                    classes_all.append(cls[mask])

            if len(boxes_all):
                boxes_all = torch.cat(boxes_all, 0)
                scores_all = torch.cat(scores_all, 0)
                classes_all = torch.cat(classes_all, 0)

                # === FIX 2: 모든 스케일 합친 뒤 최종 NMS ===
                keep = nms(boxes_all, scores_all, iou_thres)
                boxes_all, scores_all, classes_all = (
                    boxes_all[keep],
                    scores_all[keep],
                    classes_all[keep],
                )

                # === FIX 3: 원본 크기에 맞게 되돌리기(옵션) ===
                if orig_size is not None:
                    H0, W0 = orig_size
                    boxes_all[:, [0, 2]] *= W0 / x.shape[-1]  # 640→W0
                    boxes_all[:, [1, 3]] *= H0 / x.shape[-2]  # 640→H0

                results.append(
                    {"boxes": boxes_all, "scores": scores_all, "classes": classes_all}
                )
            else:
                results.append(
                    {
                        "boxes": torch.zeros((0, 4), device=device),
                        "scores": torch.zeros((0,), device=device),
                        "classes": torch.zeros((0,), device=device),
                    }
                )
        return results
