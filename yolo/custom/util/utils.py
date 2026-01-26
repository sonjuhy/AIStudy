from torchvision.ops import nms as tv_nms

import torch
import numpy as np


def xywh_norm_to_xyxy_abs(gt_boxes, img_h, img_w):
    # gt_boxes: [N, 4] (cx, cy, w, h) in [0,1]
    cx = gt_boxes[:, 0] * img_w
    cy = gt_boxes[:, 1] * img_h
    w = gt_boxes[:, 2] * img_w
    h = gt_boxes[:, 3] * img_h

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


# ------------------------------
# DFL(Distribution Focal Loss) 기반 회귀 → bbox 디코딩
# ------------------------------
def dist2bbox(distance, anchor_points, xywh=True):
    """
    distance: [B, N, 4], 좌/상/우/하 거리
    anchor_points: [N, 2], 각 feature grid 위치
    """
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        cxy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((cxy, wh), -1)
    return torch.cat((x1y1, x2y2), -1)


# ------------------------------
# NMS (Non-Maximum Suppression)
# ------------------------------
def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold=0.6):
    """
    boxes: [N, 4], scores: [N]
    return: kept indices
    """
    # keep = []
    # idxs = scores.argsort(descending=True)
    # while idxs.numel() > 0:
    #     i = idxs[0]
    #     keep.append(i)
    #     if idxs.numel() == 1:
    #         break
    #     ious = bbox_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])
    #     idxs = idxs[1:][ious < iou_threshold]
    # return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    return tv_nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold)


def bbox_iou(box1, box2, eps=1e-9):
    """
    box1: [N,4], box2: [M,4], xyxy 형식
    """
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    area1 = ((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]))[:, None]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1 + area2 - inter + eps)


def box_iou(box1, box2):
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


# Precision, Recall 계산
def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_map(
    pred_boxes,
    pred_scores,
    pred_labels,
    true_boxes,
    true_labels,
    iou_thresholds=np.arange(0.5, 1.0, 0.05),
):
    """Compute mAP@50:95 for one batch."""
    aps = []

    if len(true_boxes) == 0:
        return 0.0, 0.0

    for iou_thres in iou_thresholds:
        tp, fp, total_gt = [], [], 0
        for c in torch.unique(torch.cat([pred_labels, true_labels])):
            pred_mask = pred_labels == c
            gt_mask = true_labels == c

            preds_c = pred_boxes[pred_mask]
            scores_c = pred_scores[pred_mask]
            gts_c = true_boxes[gt_mask]

            if len(gts_c) == 0:
                continue

            total_gt += len(gts_c)
            if len(preds_c) == 0:
                continue

            ious = box_iou(preds_c, gts_c)
            assigned_gt = torch.zeros(len(gts_c), dtype=torch.bool)
            order = scores_c.argsort(descending=True)

            for i in order:
                iou, j = ious[i].max(0)
                if iou >= iou_thres and not assigned_gt[j]:
                    tp.append(1)
                    fp.append(0)
                    assigned_gt[j] = True
                else:
                    tp.append(0)
                    fp.append(1)

        if total_gt == 0:
            continue

        tp, fp = np.array(tp), np.array(fp)
        if tp.sum() + fp.sum() == 0:
            continue
        recall = np.cumsum(tp) / total_gt
        precision = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
        ap = compute_ap(recall, precision)
        aps.append(ap)

    return aps[0] if aps else 0.0, np.mean(aps) if aps else 0.0  # (mAP@50, mAP@50-95)


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
        # patience: 몇 epoch 동안 개선 없으면 stop
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, current):
        if current < self.best - self.min_delta:
            self.best = current
            self.counter = 0
            return False  # 계속 학습
        else:
            self.counter += 1
            return self.counter >= self.patience  # True면 stop
