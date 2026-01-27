from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from tqdm import tqdm

import os
import cv2
import json
import torch
import random
import numpy as np
import torchvision.transforms.functional as TF


class COCODetectionBasicYOLOv8(Dataset):
    """
    COCO annotations를 읽어서
    - 이미지: [3, H, W] float32 [0,1]
    - targets: dict(cls[T,N,1], bboxes[T,N,4] xywh_norm)
    를 반환합니다.
    """

    def __init__(self, img_dir, ann_file, img_size=640, cache=False):
        super().__init__()
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.img_ids = list(self.coco.imgs.keys())
        self.img_size = img_size

        # COCO category_id -> 0..(nc-1) 매핑
        cat_ids = sorted(self.coco.cats.keys())
        self.catid2trainid = {cat_id: i for i, cat_id in enumerate(cat_ids)}

        self.cache = cache
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.img_ids)

    def _load_image_anns(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs([img_id])[0]
        path = os.path.join(self.img_dir, img_info["file_name"])
        img = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        return img, img_info, anns

    def __getitem__(self, index):
        if self.cache and index in self._cache:
            return self._cache[index]

        img, img_info, anns = self._load_image_anns(index)

        # 원본 크기
        ow, oh = img.size

        # Resize to square (640x640) with letterbox 없이 단순 resize (간단 버전)
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = TF.to_tensor(img).float()  # [3, H, W], 0..1

        # bbox: COCO는 [x, y, w, h] (absolute, 원본)
        # Resize했으니 절대좌표도 같은 비율로 스케일 → 정규화(0~1)
        boxes = []
        labels = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            cx = (x + w / 2.0) * (self.img_size / ow)
            cy = (y + h / 2.0) * (self.img_size / oh)
            nw = w * (self.img_size / ow)
            nh = h * (self.img_size / oh)

            # 정규화
            cx /= self.img_size
            cy /= self.img_size
            nw /= self.img_size
            nh /= self.img_size

            boxes.append([cx, cy, nw, nh])  # xywh_norm
            labels.append(self.catid2trainid[a["category_id"]])

        if len(boxes) == 0:
            # GT 없음인 경우 길이 0 텐서로 반환
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            cls = torch.zeros((0, 1), dtype=torch.float32)
        else:
            bboxes = torch.tensor(boxes, dtype=torch.float32)  # [N, 4] xywh_norm
            cls = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # [N,1]

        sample = {
            "img": img,  # [3, 640, 640]
            "bboxes": bboxes,  # [N, 4] (xywh_norm)
            "cls": cls,  # [N, 1]
        }

        if self.cache:
            self._cache[index] = sample
        return sample


class COCODetectionYOLOv8(Dataset):
    """
    COCO annotations를 읽어서
    - 이미지: [3, H, W] float32 [0,1]
    - bboxes: [N,4] (cx,cy,w,h) / img_size 로 정규화된 값 (xywh_norm)
    - cls: [N,1] (class index)
    YOLO 스타일 증강 (mosaic/mixup/hsv/flip/letterbox) 적용 가능.
    """

    def __init__(
        self,
        img_dir,
        ann_file,
        img_size=640,
        cache=False,
        is_train=True,
        mosaic=True,
        mixup=True,
        mosaic_prob=0.8,
        mixup_prob=0.15,
        hsv_prob=1.0,
        flip_prob=0.5,
    ):
        super().__init__()
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.img_ids = list(self.coco.imgs.keys())
        self.img_size = img_size

        # COCO category_id -> 0..(nc-1) 매핑
        cat_ids = sorted(self.coco.cats.keys())
        self.catid2trainid = {cat_id: i for i, cat_id in enumerate(cat_ids)}

        self.cache = cache
        self._cache = {} if cache else None

        # augment 관련 설정
        self.is_train = is_train
        self.mosaic = mosaic and is_train
        self.mixup = mixup and is_train
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.hsv_prob = hsv_prob
        self.flip_prob = flip_prob

    # ---------------- 기본 COCO 로딩 ----------------
    def __len__(self):
        return len(self.img_ids)

    def _load_image_anns(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs([img_id])[0]
        path = os.path.join(self.img_dir, img_info["file_name"])
        img = cv2.imread(path)
        assert img is not None, f"Image not found: {path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        return img, img_info, anns

    def load_image_and_labels(self, index):
        """
        img: HWC, uint8, RGB
        labels: [N,5] (cls, x1, y1, x2, y2) absolute (원본 이미지 좌표)
        """
        img, img_info, anns = self._load_image_anns(index)
        h0, w0 = img.shape[:2]

        boxes = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            cls_id = self.catid2trainid[a["category_id"]]
            boxes.append([cls_id, x1, y1, x2, y2])

        if len(boxes):
            labels = np.array(boxes, dtype=np.float32)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        return img, labels

    # ---------------- Letterbox (비율 유지 + 패딩) ----------------
    @staticmethod
    def letterbox(img, new_shape=640, color=(114, 114, 114)):
        shape = img.shape[:2]  # [h, w]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = (r, r)

        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return img, ratio, (dw, dh)

    @staticmethod
    def clip_boxes(labels, w, h):
        """labels: [N,5] (cls,x1,y1,x2,y2)"""
        if labels.shape[0] == 0:
            return labels
        labels[:, 1] = np.clip(labels[:, 1], 0, w - 1)
        labels[:, 3] = np.clip(labels[:, 3], 0, w - 1)
        labels[:, 2] = np.clip(labels[:, 2], 0, h - 1)
        labels[:, 4] = np.clip(labels[:, 4], 0, h - 1)
        # 유효하지 않은 박스 제거
        w_box = labels[:, 3] - labels[:, 1]
        h_box = labels[:, 4] - labels[:, 2]
        keep = (w_box > 1) & (h_box > 1)
        return labels[keep]

    # ---------------- Mosaic ----------------
    def load_mosaic(self, index):
        """
        4장의 이미지를 하나의 큰 canvas(2*img_size)에 붙이고
        마지막에 letterbox로 img_size로 줄임.
        """
        img_size = self.img_size
        yc, xc = [int(random.uniform(img_size * 0.5, img_size * 1.5)) for _ in range(2)]
        mosaic_img = np.full((img_size * 2, img_size * 2, 3), 114, dtype=np.uint8)
        mosaic_labels = []

        indices = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]

        for i, idx in enumerate(indices):
            img, labels = self.load_image_and_labels(idx)
            h, w = img.shape[:2]

            # 이 이미지가 들어갈 위치 계산
            if i == 0:  # top left
                x1a = max(xc - w, 0)
                y1a = max(yc - h, 0)
                x2a = xc
                y2a = yc
                x1b = w - (x2a - x1a)
                y1b = h - (y2a - y1a)
                x2b = w
                y2b = h
            elif i == 1:  # top right
                x1a = xc
                y1a = max(yc - h, 0)
                x2a = min(xc + w, img_size * 2)
                y2a = yc
                x1b = 0
                y1b = h - (y2a - y1a)
                x2b = x2a - x1a
                y2b = h
            elif i == 2:  # bottom left
                x1a = max(xc - w, 0)
                y1a = yc
                x2a = xc
                y2a = min(img_size * 2, yc + h)
                x1b = w - (x2a - x1a)
                y1b = 0
                x2b = w
                y2b = y2a - y1a
            else:  # bottom right
                x1a = xc
                y1a = yc
                x2a = min(xc + w, img_size * 2)
                y2a = min(img_size * 2, yc + h)
                x1b = 0
                y1b = 0
                x2b = x2a - x1a
                y2b = y2a - y1a

            # 이미지 붙이기
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            if labels.shape[0]:
                labels_ = labels.copy()
                # 좌표 이동
                labels_[:, 1] = labels_[:, 1] + x1a - x1b
                labels_[:, 2] = labels_[:, 2] + y1a - y1b
                labels_[:, 3] = labels_[:, 3] + x1a - x1b
                labels_[:, 4] = labels_[:, 4] + y1a - y1b
                mosaic_labels.append(labels_)

        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels, axis=0)
            mosaic_labels = self.clip_boxes(
                mosaic_labels, mosaic_img.shape[1], mosaic_img.shape[0]
            )
        else:
            mosaic_labels = np.zeros((0, 5), dtype=np.float32)

        # 최종 letterbox로 img_size로 줄이기
        mosaic_img, ratio, (dw, dh) = self.letterbox(
            mosaic_img, new_shape=self.img_size
        )
        h_final, w_final = mosaic_img.shape[:2]

        if mosaic_labels.shape[0]:
            # ratio, pad 반영
            mosaic_labels[:, [1, 3]] = mosaic_labels[:, [1, 3]] * ratio[0] + dw * 2  # x
            mosaic_labels[:, [2, 4]] = mosaic_labels[:, [2, 4]] * ratio[1] + dh * 2  # y
            mosaic_labels = self.clip_boxes(mosaic_labels, w_final, h_final)

        return mosaic_img, mosaic_labels

    # ---------------- MixUp ----------------
    @staticmethod
    def mixup_augment(img1, labels1, img2, labels2):
        r = np.random.beta(32.0, 32.0)
        img = (img1.astype(np.float32) * r + img2.astype(np.float32) * (1 - r)).astype(
            np.uint8
        )
        if labels1.shape[0] and labels2.shape[0]:
            labels = np.concatenate((labels1, labels2), 0)
        elif labels1.shape[0]:
            labels = labels1
        else:
            labels = labels2
        return img, labels

    # ---------------- HSV ----------------
    @staticmethod
    def random_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

        dtype = img.dtype
        x = np.arange(0, 256, dtype=np.int16)
        lut_h = ((x * r[0]) % 180).astype(dtype)
        lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_v = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge(
            (cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v))
        )
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return img

    # ---------------- Flip ----------------
    @staticmethod
    def random_flip(img, labels, p=0.5):
        if random.random() < p:
            img = np.fliplr(img).copy()
            if labels.shape[0]:
                h, w = img.shape[:2]
                x1 = labels[:, 1].copy()
                x2 = labels[:, 3].copy()
                labels[:, 1] = w - x2
                labels[:, 3] = w - x1
        return img, labels

    # ---------------- __getitem__ ----------------
    def __getitem__(self, index):
        if self.cache and index in self._cache:
            return self._cache[index]

        # ---- Train: mosaic/mixup + aug ----
        if self.is_train and self.mosaic and random.random() < self.mosaic_prob:
            img, labels = self.load_mosaic(index)

            if self.mixup and random.random() < self.mixup_prob:
                idx2 = random.randint(0, len(self.img_ids) - 1)
                img2, labels2 = self.load_mosaic(idx2)
                img, labels = self.mixup_augment(img, labels, img2, labels2)

        else:
            # 단일 이미지 + letterbox
            img, labels = self.load_image_and_labels(index)
            img, ratio, (dw, dh) = self.letterbox(img, new_shape=self.img_size)
            if labels.shape[0]:
                labels[:, [1, 3]] = labels[:, [1, 3]] * ratio[0] + dw * 2
                labels[:, [2, 4]] = labels[:, [2, 4]] * ratio[1] + dh * 2
                labels = self.clip_boxes(labels, img.shape[1], img.shape[0])

        # 추가 색/flip 증강
        if self.is_train:
            if random.random() < self.hsv_prob:
                img = self.random_hsv(img)
            img, labels = self.random_flip(img, labels, p=self.flip_prob)

        # ---- 최종: Tensor + xywh_norm 변환 ----
        h, w = img.shape[:2]
        if labels.shape[0]:
            # xyxy -> xywh
            x1, y1, x2, y2 = (
                labels[:, 1],
                labels[:, 2],
                labels[:, 3],
                labels[:, 4],
            )
            bw = x2 - x1
            bh = y2 - y1
            cx = x1 + bw / 2.0
            cy = y1 + bh / 2.0

            # 정규화
            cx /= w
            cy /= h
            bw /= w
            bh /= h

            bboxes = np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)
            cls = labels[:, 0:1].astype(np.float32)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.zeros((0, 1), dtype=np.float32)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        bboxes_t = torch.from_numpy(bboxes)
        cls_t = torch.from_numpy(cls)

        sample = {
            "img": img_t,  # [3, 640, 640]
            "bboxes": bboxes_t,  # [N, 4] (xywh_norm)
            "cls": cls_t,  # [N, 1]
        }

        if self.cache:
            self._cache[index] = sample

        return sample


def yolo_v8_collate_fn(batch):
    """
    v8DetectionLoss가 기대하는 형태로 batch dict를 구성:
    - imgs: [B, 3, H, W]
    - batch: {"bboxes": [M,4], "cls":[M,1], "batch_idx":[M]}
        여기서 M = 배치 내 모든 GT의 총합
    """
    imgs = torch.stack([b["img"] for b in batch], dim=0)

    # 이미지별 인덱스를 GT 개수만큼 반복해서 이어붙이기
    batch_idx_list = []
    cls_list = []
    bbox_list = []
    for i, b in enumerate(batch):
        n = b["bboxes"].shape[0]
        if n == 0:
            continue
        batch_idx_list.append(torch.full((n,), i, dtype=torch.float32))
        cls_list.append(b["cls"])  # [n,1]
        bbox_list.append(b["bboxes"])  # [n,4]

    if len(bbox_list) == 0:
        # 배치 전체에 GT가 없는 경우
        batch_idx = torch.zeros((0,), dtype=torch.float32)
        cls = torch.zeros((0, 1), dtype=torch.float32)
        bboxes = torch.zeros((0, 4), dtype=torch.float32)
    else:
        batch_idx = torch.cat(batch_idx_list, dim=0)  # [M]
        cls = torch.cat(cls_list, dim=0)  # [M,1]
        bboxes = torch.cat(bbox_list, dim=0)  # [M,4]

    target = {
        "batch_idx": batch_idx,  # [M]
        "cls": cls,  # [M,1] float (v8Loss 내부에서 to(dtype) 처리)
        "bboxes": bboxes,  # [M,4] xywh_norm
    }
    return imgs, target


def get_coco_debug_train_loader(num_samples: int = 50):
    """
    기존 get_coco_dataset_train_loader()가 반환하는 DataLoader에서
    앞에서 num_samples개만 사용하도록 Subset Loader를 만든다.
    """
    full_loader = get_coco_dataset_train_loader()
    dataset = full_loader.dataset

    n = min(num_samples, len(dataset))
    indices = list(range(n))
    subset = Subset(dataset, indices)

    debug_loader = DataLoader(
        subset,
        batch_size=full_loader.batch_size,
        shuffle=True,
        num_workers=full_loader.num_workers,
        pin_memory=getattr(full_loader, "pin_memory", False),
        collate_fn=full_loader.collate_fn,
        drop_last=False,
    )
    return debug_loader


def get_coco_dataset_train_loader(img_size: int = 640):
    # ROOT_PATH = os.path.join(
    #     os.sep,
    #     "media",
    #     "edint",
    #     "64d115f7-57cc-417b-acf0-7738ac091615",
    #     "Ivern",
    #     "DataSets",
    #     "cocoset",
    # )
    ROOT_PATH = os.path.join(
        os.sep,
        "home",
        "edint",
        "Ivern_home",
        "WorkSpace",
        "Python",
        "Yolo",
        "coco_dataset",
    )

    train_dataset = COCODetectionYOLOv8(
        img_dir=os.path.join(ROOT_PATH, "train2017"),
        ann_file=os.path.join(ROOT_PATH, "annotations", "instances_train2017.json"),
        img_size=img_size,
        cache=False,
        is_train=True,  # 🔥 train 모드
        mosaic=False,
        mixup=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=yolo_v8_collate_fn,
    )

    return train_loader


def get_coco_dataset_val_loader(img_size: int = 640):
    # ROOT_PATH = os.path.join(
    #     os.sep,
    #     "media",
    #     "edint",
    #     "64d115f7-57cc-417b-acf0-7738ac091615",
    #     "Ivern",
    #     "DataSets",
    #     "cocoset",
    # )
    ROOT_PATH = os.path.join(
        os.sep,
        "home",
        "edint",
        "Ivern_home",
        "WorkSpace",
        "Python",
        "Yolo",
        "coco_dataset",
    )
    val_dataset = COCODetectionYOLOv8(
        img_dir=os.path.join(ROOT_PATH, "val2017"),
        ann_file=os.path.join(ROOT_PATH, "annotations", "instances_val2017.json"),
        img_size=img_size,
        cache=False,
        is_train=False,  # 🔥 val에서는 증강 off
        mosaic=False,
        mixup=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=yolo_v8_collate_fn,
    )
    return val_loader


def evaluate_map(
    model,
    val_loader,
    coco_gt,
    device="cuda",
    iou_type="bbox",
    save_json="predictions.json",
):
    """
    COCO ground truth와 모델 예측값으로 mAP 계산
    """
    model.eval()
    results = []

    print("\n📊 Running mAP evaluation...")
    with torch.no_grad():
        for imgs, batch in tqdm(val_loader, desc="Evaluating mAP"):
            imgs = imgs.to(device, non_blocking=True)
            preds = model(imgs)

            # preds: (p3, p4, p5)
            # → 각 scale별 detection 결과를 concat 후 post-processing 필요
            # YOLOv8-style 구조라면 detect() head에서 NMS 이후 결과 반환하도록 수정해야 함
            # 여기선 간단히 placeholder 형태로 예시

            # (예시) 각 이미지당 랜덤 예측 결과 (실제는 NMS/Decode 필요)
            for i in range(imgs.size(0)):
                img_id = (
                    int(batch["batch_idx"][i].item()) if "batch_idx" in batch else i
                )
                # 예시용 임의 결과
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": 1,  # 실제 클래스 매핑 필요
                        "bbox": [100, 100, 50, 80],  # [x, y, w, h]
                        "score": 0.8,
                    }
                )

    # ----------------------------
    #  COCOEval 실행
    # ----------------------------
    if len(results) == 0:
        print("⚠️ No detections to evaluate.")
        return {"mAP50": 0.0, "mAP5095": 0.0}

    with open(save_json, "w") as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(save_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 결과: mAP@50, mAP@50-95
    mAP_50_95 = coco_eval.stats[0]
    mAP_50 = coco_eval.stats[1]

    return {"mAP50": mAP_50, "mAP5095": mAP_50_95}


if __name__ == "__main__":
    # ROOT_PATH = os.path.join(
    #     os.sep,
    #     "media",
    #     "edint",
    #     "64d115f7-57cc-417b-acf0-7738ac091615",
    #     "Ivern",
    #     "DataSets",
    #     "cocoset",
    # )
    # /home/edint/Ivern_home/WorkSpace/Python/Yolo/coco_dataset/
    ROOT_PATH = os.path.join(
        os.sep,
        "home",
        "edint",
        "Ivern_home",
        "WorkSpace",
        "Python",
        "Yolo",
        "coco_dataset",
    )
    print(ROOT_PATH)
    print(os.path.exists(ROOT_PATH))
