import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from typing import List, Tuple, Dict

# --- (이전 정의된 Conv, Bottleneck, SPPF, C2PSA, C3k2, Concat, Detect, YoloV11, YoloV11Trainer) ---


class CocoDataset(Dataset):
    """
    COCODetection을 래핑하여 YOLOTrainer가 기대하는 targets 포맷 (N, 6) 으로 반환합니다.
    targets[:,0] = class_id, targets[:,1:5] = normalized [cx, cy, w, h]
    """

    def __init__(
        self, img_dir: str, ann_file: str, img_size: int = 640, transforms_=None
    ):
        super().__init__()
        self.coco = CocoDetection(img_dir, ann_file)
        self.img_size = img_size
        self.transforms = transforms_ or transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )
        # COCO에서 클래스 id 목록 (원본 id → 연속 id) 매핑
        self.cat_ids = self.coco.coco.getCatIds()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, anns = self.coco[idx]
        # image resize & tensor 변환
        img = self.transforms(img)

        h0, w0 = img.shape[1:]  # after resize, h0 = w0 = img_size
        targets = []
        for ann in anns:
            # crowd samples 무시
            if ann.get("iscrowd", 0):
                continue
            # bbox: [x_min, y_min, w, h]
            x, y, w, h = ann["bbox"]
            # normalize to [0,1]
            cx = (x + w / 2) / w0
            cy = (y + h / 2) / h0
            nw = w / w0
            nh = h / h0
            cls_id = self.cat2label[ann["category_id"]]
            targets.append([cls_id, cx, cy, nw, nh])
        if len(targets) == 0:
            # 아무 객체도 없으면 dummy zero
            targets = [[0, 0, 0, 0, 0]]
        targets = torch.tensor(targets, dtype=torch.float32)
        return img, targets
