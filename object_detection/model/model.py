from typing import Literal
from object_detection.model.mobilenet_backbone import (
    MobileNetV4ConvSmallBackbone,
    MobileNetV4ConvLargeBackbone,
    MobileNetV4ConvMediumBackbone,
)
from object_detection.model.neck import (
    LightweightPANNeck,
    YOLO11DynamicNeck,
    YOLO11Neck,
)
from object_detection.model.head import (
    DFLHead,
    StandardDecoupledHead,
    DWConvDecoupledHead,
)
from object_detection.utils import dist2bbox, nms

import torch
import torch.nn as nn

import copy
from copy import deepcopy
import torch
import torch.nn as nn
import torchvision


class ModelEMA:
    """Simple EMA wrapper for a PyTorch model (Ultralytics 스타일)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = ""):
        # EMA용 모델 복사
        self.ema = deepcopy(model).eval()
        # EMA weight는 학습 안 하니까 grad 끔
        for p in self.ema.parameters():
            p.requires_grad_(False)

        self.decay = decay
        self.device = device

        if device:
            self.ema.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        model의 weight를 EMA 모델에 반영.
        ema = d * ema + (1 - d) * model
        """
        msd = model.state_dict()
        esd = self.ema.state_dict()

        for k, v in esd.items():
            if k in msd:
                m = msd[k].detach()
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + m * (1.0 - self.decay))
                else:
                    # float가 아닌 버퍼/정수 텐서는 그대로 복사
                    v.copy_(m)

    def update_attr(self, model, include=("nc", "names", "stride"), exclude=()):
        """
        model에 있는 몇몇 속성(nc, names, stride 등)을 EMA 모델에도 동기화.
        """
        for k in include:
            if hasattr(model, k) and not hasattr(self.ema, k) and k not in exclude:
                setattr(self.ema, k, getattr(model, k))


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


class YOLO11_MobileNetV4_DFL(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        reg_max: int = 16,
        backbone_size: Literal[1, 2, 3, 4, 5] = 1,
    ):
        super().__init__()
        if backbone_size == 1:
            self.backbone = MobileNetV4ConvSmallBackbone()
        elif backbone_size == 2:
            self.backbone = MobileNetV4ConvMediumBackbone()
        elif backbone_size == 3:
            self.backbone = MobileNetV4ConvLargeBackbone()
        else:
            # Default to Medium if unknown or explicitly 4
            self.backbone = MobileNetV4ConvMediumBackbone()
        # self.neck = YOLO11Neck(in_channels=self.backbone.out_channels, width=1.00)
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
    def predict(self, x, orig_size=None, conf_thres=0.001, iou_thres=0.6):
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


class StandardEMA:
    """
    Ultralytics 종속성이 없는 순수 가중치 이동 평균(EMA) 클래스입니다.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.ema: nn.Module = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay: float = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd: dict[str, torch.Tensor] = model.state_dict()
        esd: dict[str, torch.Tensor] = self.ema.state_dict()
        for k, v in esd.items():
            if k in msd:
                m_val: torch.Tensor = msd[k].detach()
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + m_val * (1.0 - self.decay))
                else:
                    v.copy_(m_val)


class MobileVisionNet(nn.Module):
    """
    Ultralytics 종속성이 전혀 없는 완전한 독자적 모바일 객체 탐지 모델입니다.
    """

    def __init__(
        self,
        num_classes: int = 80,
        backbone_size: Literal[1, 2, 3, 4, 5] = 1,
    ) -> None:
        super().__init__()
        self.nc: int = num_classes
        self.stride: list[int] = [8, 16, 32]

        # 1. Backbone
        if backbone_size == 1:
            self.backbone = MobileNetV4ConvSmallBackbone()
        elif backbone_size == 2:
            self.backbone = MobileNetV4ConvMediumBackbone()
        elif backbone_size == 3:
            self.backbone = MobileNetV4ConvLargeBackbone()
        elif backbone_size in (4, 5):
            self.backbone = MobileNetV4ConvMediumBackbone()
        else:
            raise ValueError(f"backbone_size {backbone_size}는 1~3만 지원합니다.")

        # 2. Neck
        # 백본의 출력 채널을 파악하여 Neck에 전달 (예: 96, 192, 512)
        bb_out_channels: tuple[int, int, int] = self.backbone.out_channels
        width_channels: float = 1.00
        expansion: float = 4.00

        match backbone_size:
            case 1:
                # Small 백본: 연산량 최소화를 위한 수준의 폭(0.25) 적용
                self.backbone = MobileNetV4ConvSmallBackbone()
                width_channels = 0.25
            case 2:
                # Medium 백본: 균형 잡힌 수준의 폭(0.50) 적용
                self.backbone = MobileNetV4ConvMediumBackbone()
                width_channels = 0.50
            case 3:
                # Large 백본: 최대 표현력을 위한한 폭(1.00) 적용
                self.backbone = MobileNetV4ConvLargeBackbone()
                width_channels = 0.75
                expansion = 2.0
            case 4:
                # Custom 설정: 백본은 Medium, Neck 연산량은 높임
                self.backbone = MobileNetV4ConvMediumBackbone()
                width_channels = 0.75
                expansion = 2.0
            case 5:
                # Custom 설정: 백본은 Medium, Neck 연산량 최대 (파라미터 병목 테스트용)
                self.backbone = MobileNetV4ConvMediumBackbone()
                width_channels = 1.00
                expansion = 2.0
            case _:
                raise ValueError(f"지원하지 않는 backbone_size 입니다: {backbone_size}")

        self.neck: LightweightPANNeck = LightweightPANNeck(
            in_channels=bb_out_channels,
            width_mult=width_channels,
            expansion=expansion,
        )

        # 3. Head
        # self.head: StandardDecoupledHead = StandardDecoupledHead(
        #     num_classes=self.nc,
        #     in_channels=(self.neck.p3_out, self.neck.p4_out, self.neck.p5_out),
        # )
        self.head: DWConvDecoupledHead = DWConvDecoupledHead(
            num_classes=self.nc,
            in_channels=(self.neck.p3_out, self.neck.p4_out, self.neck.p5_out),
        )
        self._anchor_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _make_anchors_vectorized(
        self, feats: list[torch.Tensor], strides: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """텐서 크기가 바뀔 때만 앵커를 생성하고 나머지는 캐시를 사용합니다."""
        h_w_str: str = f"{feats[0].shape[2]}_{feats[0].shape[3]}"
        if h_w_str in self._anchor_cache:
            return self._anchor_cache[h_w_str]

        anchor_points, stride_tensor = [], []
        device, dtype = feats[0].device, feats[0].dtype

        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            shift_y, shift_x = torch.meshgrid(
                torch.arange(end=h, device=device, dtype=dtype),
                torch.arange(end=w, device=device, dtype=dtype),
                indexing="ij",
            )
            # 앵커 중심점 좌표 생성 (C++ 텐서 연산으로 일괄 처리)
            anchors = torch.stack((shift_x + 0.5, shift_y + 0.5), dim=-1) * stride
            anchor_points.append(anchors.reshape(-1, 2))
            stride_tensor.append(
                torch.full((h * w, 1), stride, dtype=dtype, device=device)
            )

        result = (torch.cat(anchor_points), torch.cat(stride_tensor))
        self._anchor_cache[h_w_str] = result
        return result

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        p3, p4, p5 = self.backbone(x)
        features = self.neck(p3, p4, p5)
        return self.head(features)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        orig_size: tuple[int, int] | None = None,
        conf_thres: float = 0.001,
        iou_thres: float = 0.6,
    ) -> list[dict[str, torch.Tensor]]:

        self.eval()
        preds: list[torch.Tensor] = self.forward(x)
        bs, _, img_h, img_w = x.shape
        device = x.device

        import torchvision

        flatten_preds: list[torch.Tensor] = []
        anchors_list: list[torch.Tensor] = []
        strides_list: list[torch.Tensor] = []

        # 1. 앵커 및 텐서 플래튼 (단 3번의 루프)
        for i, p in enumerate(preds):
            stride = self.stride[i]
            _, _, h, w = p.shape

            # [B, 4 + nc, H, W] -> [B, 4 + nc, H*W]
            flatten_preds.append(p.view(bs, self.nc + 4, -1))

            # 앵커 및 스케일 캐싱 (배치 전체 공유)
            gy, gx = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            anchor = torch.stack((gx + 0.5, gy + 0.5), dim=-1).view(-1, 2) * stride
            anchors_list.append(anchor)
            strides_list.append(torch.full((h * w, 1), stride, device=device))

        # 2. C++ 백엔드를 통한 일괄 병합 (Vectorization)
        pred_concat = torch.cat(flatten_preds, dim=2)  # [B, 84, 8400]
        anchors = torch.cat(anchors_list, dim=0).unsqueeze(0)  # [1, 8400, 2]
        stride_tensor = torch.cat(strides_list, dim=0).unsqueeze(0)  # [1, 8400, 1]

        # 3. Box와 Cls 일괄 분리 및 차원 재배열
        pred_box = pred_concat[:, :4, :].permute(0, 2, 1)  # [B, 8400, 4]
        pred_cls = pred_concat[:, 4:, :].permute(0, 2, 1)  # [B, 8400, 80]

        # 4. 전체 배치에 대한 클래스 확률 일괄 계산
        scores = pred_cls.sigmoid()  # [B, 8400, 80]

        # 5. 전체 배치에 대한 박스 일괄 디코딩 (Relu + 스케일 복원)
        dist = pred_box.relu() * stride_tensor  # [B, 8400, 4]
        x1y1 = anchors - dist[..., :2]
        x2y2 = anchors + dist[..., 2:]
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # [B, 8400, 4]

        # 6. 배치별 NMS 후처리 (결과 갯수가 다르므로 여기만 for문 사용)
        results: list[dict[str, torch.Tensor]] = []
        for b in range(bs):
            score_b = scores[b]
            box_b = boxes[b]

            conf, labels = score_b.max(dim=1)
            mask = conf > conf_thres

            if not mask.any():
                results.append(
                    {
                        "boxes": torch.zeros((0, 4), device=device),
                        "scores": torch.zeros((0,), device=device),
                        "labels": torch.zeros((0,), device=device, dtype=torch.long),
                    }
                )
                continue

            valid_boxes = box_b[mask]
            valid_scores = conf[mask]
            valid_labels = labels[mask]

            keep = torchvision.ops.nms(valid_boxes, valid_scores, iou_thres)

            final_boxes = valid_boxes[keep]
            final_scores = valid_scores[keep]
            final_labels = valid_labels[keep]

            if orig_size is not None:
                H0, W0 = orig_size
                final_boxes[:, [0, 2]] *= W0 / img_w
                final_boxes[:, [1, 3]] *= H0 / img_h

            results.append(
                {"boxes": final_boxes, "scores": final_scores, "labels": final_labels}
            )

        return results

    # @torch.no_grad()
    # def predict(
    #     self,
    #     x: torch.Tensor,
    #     orig_size: tuple[int, int] | None = None,
    #     conf_thres: float = 0.001,
    #     iou_thres: float = 0.6,
    # ) -> list[dict[str, torch.Tensor]]:
    #     training = self.training
    #     self.eval()
    #     with torch.no_grad():
    #         preds = self.forward(x)
    #     self.train(training)

    #     preds: list[torch.Tensor] = self.forward(x)
    #     bs: int = x.shape[0]
    #     device: torch.device = x.device
    #     results: list[dict[str, torch.Tensor]] = []

    #     for b in range(bs):
    #         boxes_all: list[torch.Tensor] = []
    #         scores_all: list[torch.Tensor] = []
    #         classes_all: list[torch.Tensor] = []

    #         for i, p in enumerate(preds):
    #             stride: int = self.stride[i]
    #             _, _, h, w = p.shape
    #             p_b: torch.Tensor = p[b : b + 1]  # [1, C, H, W]

    #             # Box와 Cls 분리 (앞 4채널이 Box)
    #             pred_box, pred_cls = torch.split(p_b, [4, self.nc], dim=1)

    #             # 그리드 앵커 생성 (중심점: i+0.5, j+0.5)
    #             gy, gx = torch.meshgrid(
    #                 torch.arange(h, device=device),
    #                 torch.arange(w, device=device),
    #                 indexing="ij",
    #             )
    #             anchor_points: torch.Tensor = torch.stack(
    #                 ((gx + 0.5) * stride, (gy + 0.5) * stride), dim=-1
    #             ).reshape(-1, 2)

    #             # DFL 없는 직접 디코딩 로직 (거리 렐루 적용 후 앵커에 연산)
    #             pred_dist: torch.Tensor = (
    #                 pred_box.permute(0, 2, 3, 1).reshape(-1, 4).relu()
    #             )
    #             lt: torch.Tensor = pred_dist[:, :2] * stride
    #             rb: torch.Tensor = pred_dist[:, 2:] * stride

    #             boxes: torch.Tensor = torch.cat(
    #                 [anchor_points - lt, anchor_points + rb], dim=-1
    #             )  # [x1, y1, x2, y2]

    #             # Cls 디코딩
    #             scores: torch.Tensor = (
    #                 pred_cls.sigmoid().permute(0, 2, 3, 1).reshape(-1, self.nc)
    #             )
    #             conf, cls = scores.max(1)

    #             # Threshold 필터링
    #             mask: torch.Tensor = conf > conf_thres
    #             if mask.any():
    #                 boxes_all.append(boxes[mask])
    #                 scores_all.append(conf[mask])
    #                 classes_all.append(cls[mask])

    #         # NMS 처리 및 결과 저장
    #         if len(boxes_all):
    #             boxes_tensor: torch.Tensor = torch.cat(boxes_all, 0)
    #             scores_tensor: torch.Tensor = torch.cat(scores_all, 0)
    #             classes_tensor: torch.Tensor = torch.cat(classes_all, 0)

    #             keep: torch.Tensor = nms(boxes_tensor, scores_tensor, iou_thres)
    #             boxes_tensor = boxes_tensor[keep]

    #             # 원본 크기 복원 (옵션)
    #             if orig_size is not None:
    #                 H0, W0 = orig_size
    #                 boxes_tensor[:, [0, 2]] *= W0 / x.shape[-1]
    #                 boxes_tensor[:, [1, 3]] *= H0 / x.shape[-2]

    #             results.append(
    #                 {
    #                     "boxes": boxes_tensor,
    #                     "scores": scores_tensor[keep],
    #                     "labels": classes_tensor[keep],
    #                 }
    #             )
    #         else:
    #             results.append(
    #                 {
    #                     "boxes": torch.zeros((0, 4), device=device),
    #                     "scores": torch.zeros((0,), device=device),
    #                     "labels": torch.zeros((0,), device=device),
    #                 }
    #             )

    #     return results
