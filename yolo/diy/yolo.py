from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

from modules.coco_dataset import CocoDataset
from modules.block import Conv, C3k2, SPPF, C2PSA
from modules.utils import Concat

import os
import torch
import torch.nn as nn
import torch.optim as optim


class Detect(nn.Module):
    """
    YOLO Detection Head.
    anchors: List of tuples
    nc: number of classes
    """

    def __init__(self, ch: List[int], nc: int, anchors: List[List[int]]):
        super().__init__()
        self.nc = nc
        self.na = len(anchors)  # number of anchors per scale
        self.no = nc + 5  # outputs per anchor
        self.m = nn.ModuleList(
            [  # 3 detection conv layers
                Conv(ch[i], self.na * self.no, k=1, s=1, act=False)
                for i in range(len(ch))
            ]
        )
        self.anchors = anchors

    def forward(self, xs: List[torch.Tensor]):
        # xs: list of feature maps from Neck ([P3, P4, P5])
        out = []
        for i, x in enumerate(xs):
            bs, _, h, w = x.shape
            y = self.m[i](x)
            # reshape to (bs, na, no, h, w) -> (bs, na*h*w, no)
            y = y.view(bs, self.na, self.no, h, w).permute(0, 1, 3, 4, 2)
            out.append(y)
        return out


class YoloV11(nn.Module):
    def __init__(self, nc: int):
        super().__init__()
        # --- 하이퍼파라미터 예시 ---
        # anchors 예시 (각 scale 당 3개)
        anchors = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]],
        ]

        # 채널 정의
        # 2의 n배수에서 n 값에 따라 모델 사이즈 달라짐 (지금은 2의 7배수 부터 시작 : 33M)
        # ~2M : n, ~7M : s, ~20M : m, ~45M : l, ~85M : x
        ch1, ch2, ch3, ch4, ch5 = 64, 128, 256, 512, 1024

        # Backbone 레이어 리스트
        self.backbone = nn.Sequential(
            Conv(3, ch1, 3, 1),
            Conv(ch1, ch1, 3, 2),
            C3k2(ch1, ch2, n=3, shortcut=False),  # → P3, ch2=128
            Conv(ch2, ch2, 3, 2),
            C3k2(ch2, ch3, n=6, shortcut=True),  # → P4, ch3=256
            Conv(ch3, ch3, 3, 2),
            C3k2(ch3, ch4, n=6, shortcut=True),  # → P5, ch4=512
            Conv(ch4, ch5, 3, 2),
            C3k2(ch5, ch5, n=3, shortcut=True),  # (더 깊은 레이어지만 사용 안함)
        )

        # 2) Neck
        #   - P5 → SPPF → C2PSA  → up → concat P4 → C3k2 → up → concat P3 → C3k2
        self.sppf = SPPF(ch5, ch5, k=5)
        self.c2psa = C2PSA(ch5, ch5, n=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.concat = Concat()

        # C3 blocks after concat at different scales
        self.neck_p4 = C3k2(ch4 + ch5, ch4, n=3, shortcut=False)
        self.neck_p3 = C3k2(ch3 + ch4, ch3, n=3, shortcut=False)

        # 3) Head
        # detection on [P3_out, P4_out, P5_out]
        self.detect = Detect(ch=[ch3, ch4, ch5], nc=nc, anchors=anchors)

    def forward(self, x):
        # 백본을 순차적으로 거치면서 P3, P4, P5를 정확히 뽑아 줍니다.
        x = self.backbone[0](x)
        x = self.backbone[1](x)
        x = self.backbone[2](x)

        x = self.backbone[3](x)
        x = self.backbone[4](x)
        p3 = x

        x = self.backbone[5](x)
        x = self.backbone[6](x)
        p4 = x

        x = self.backbone[7](x)
        p5 = x

        # P5 → up → concat with P4 (256 채널)
        # p5에 SPPF, C2PSA 등 적용…
        p5 = self.sppf(p5)
        p5 = self.c2psa(p5)

        # P5 → up → concat with P4 (256 채널)
        p5_up = self.upsample(p5)  # 512 → 512

        p4_in = self.concat([p5_up, p4])  # 512 + 256 = 768 !!!
        p4 = self.neck_p4(p4_in)  # 이제 neck_p4는 C3k2(768, 256)이어야 함

        # P4 → up → concat with P3 (128 채널)
        p4_up = self.upsample(p4)  # 256 → 256
        p3_in = self.concat([p4_up, p3])  # 256 + 128 = 384
        p3 = self.neck_p3(p3_in)  # 이제 neck_p3는 C3k2(384, 128)이어야 함

        # Detect head에 넘기기 (channels=[128,256,512])
        return self.detect([p3, p4, p5])


class YoloV11Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_ds: Dataset,
        val_ds: Dataset,
        nc: int,
        img_size: int = 640,
        batch_size: int = 16,
        lr: float = 1e-3,
        epochs: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "runs/train/exp",
    ):
        self.device = device
        self.model = model.to(self.device)
        self.epochs = epochs
        self.nc = nc

        self.save_dir = save_dir

        # Save 디렉토리 생성
        os.makedirs(self.save_dir, exist_ok=True)

        # DataLoader
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn
        )

        # Optimizer & Scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )

        # 손실 함수 (간단히 BCE+MSE 조합 예시)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs, 0)
        # targets 는 list of Tensor[N_i,6]
        return imgs, targets

    def compute_loss(self, preds: List[torch.Tensor], targets: List[torch.Tensor]):
        """
        preds: list of feature map outputs, 각 (bs, na, h, w, no)
        targets: list of (n_i, 6) tensor
        간단한 매칭 없이 전체 예측에 대해 동일 타깃을 적용하는 very-simplified 예시입니다.
        """
        loss = torch.tensor(0.0, device=self.device)
        for p in preds:
            # p[..., 0:self.nc] : 클래스 로짓
            # p[..., self.nc    ] : objectness 로짓
            # p[..., self.nc+1:] : box offsets
            cls_pred = p[..., : self.nc]
            obj_pred = p[..., self.nc : self.nc + 1]
            box_pred = p[..., self.nc + 1 :]

            # 여기서는 모든 그리드셀에 타깃 0을 부여
            target_cls = torch.zeros_like(cls_pred)
            target_obj = torch.zeros_like(obj_pred)
            target_box = torch.zeros_like(box_pred)

            loss += self.bce_loss(cls_pred, target_cls)
            loss += self.bce_loss(obj_pred, target_obj)
            loss += self.mse_loss(box_pred, target_box)

        return loss

    def train(self):
        print("start train")
        for epoch in range(self.epochs):
            # --- Training ---
            print(f"Training Epoch : {epoch+1}")
            self.model.train()
            running_loss = 0.0
            for imgs, targets in self.train_loader:
                imgs = imgs.to(self.device)
                # targets 리스트는 CPU에 둡니다
                self.optimizer.zero_grad()
                preds = self.model(imgs)  # List of feature maps
                loss = self.compute_loss(preds, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            self.scheduler.step()
            avg_train_loss = running_loss / len(self.train_loader)

            # --- Validation ---
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, targets in self.val_loader:
                    imgs = imgs.to(self.device)
                    preds = self.model(imgs)
                    loss = self.compute_loss(preds, targets)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(self.val_loader)

            print(
                f"Epoch {epoch+1}/{self.epochs} "
                f"- train_loss: {avg_train_loss:.4f} "
                f"- val_loss: {avg_val_loss:.4f}"
            )

        # 마지막 모델 저장
        last_path = os.path.join(self.save_dir, "last.pt")
        torch.save(self.model.state_dict(), last_path)
        print(f"Training complete. Saved last.pt")
        # print("Training complete.")


# -----------------------------
# 사용 예시
# -----------------------------
if __name__ == "__main__":

    # COCO 데이터셋 베이스 경로
    coco_base = os.path.join("datasets", "coco")

    # 학습용 이미지 및 어노테이션 경로
    train_img_dir = os.path.join(coco_base, "train2017")
    train_ann = os.path.join(coco_base, "annotations", "instances_train2017.json")

    # 검증용 이미지 및 어노테이션 경로
    val_img_dir = os.path.join(coco_base, "val2017")
    val_ann = os.path.join(coco_base, "annotations", "instances_val2017.json")

    # 데이터셋 & 트레이너 생성
    train_ds = CocoDataset(train_img_dir, train_ann, img_size=640)
    val_ds = CocoDataset(val_img_dir, val_ann, img_size=640)

    num_classes = len(train_ds.cat_ids)
    model = YoloV11(nc=num_classes)

    trainer = YoloV11Trainer(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        nc=num_classes,
        img_size=640,
        batch_size=1,
        lr=1e-4,
        epochs=30,
    )
    trainer.train()

    # model = YoloV11(nc=num_classes).to(
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )

    # # 전체 파라미터 수 집계
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total parameters: {total_params/1e6:.2f} M")
    # print(f"Trainable params: {trainable_params/1e6:.2f} M")
