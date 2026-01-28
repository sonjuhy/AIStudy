import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from models.ema import ModelEMA
from models.mobilenetv4_dfl import YOLO11_MobileNetV4_DFL
from models.mobilenetv4 import YOLO11_MobileNetV4

from util.loss import CustomDetectionLoss, DetectionLoss
from yolo.custom.util.dataset import (
    get_coco_dataset_train_loader,
    get_coco_dataset_val_loader,
    get_coco_debug_train_loader,
)
from yolo.custom.util.utils import (
    EarlyStopping,
    compute_ap,
    compute_map,
    xywh_norm_to_xyxy_abs,
)


def train(epochs: int = 300):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ 모델 초기화
    model = YOLO11_MobileNetV4(80)
    model = model.to(device)

    # ✅ DataParallel 로 여러 GPU 사용
    if torch.cuda.device_count() > 1:
        print(f"🔹 Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

    # ✅ 학습 설정
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = CustomDetectionLoss(num_classes=80)

    train_loader = get_coco_dataset_train_loader()
    val_loader = get_coco_dataset_val_loader()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training")

        for imgs, batch in pbar:
            imgs = imgs.to(device, non_blocking=True)
            batch = {k: v.to(device) for k, v in batch.items()}

            preds = model(imgs)
            loss, (lbox, lobj, lcls) = criterion(preds, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        # ---------------- Validation ----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, batch in tqdm(val_loader, desc=f"Epoch [{epoch+1}] Validation"):
                imgs = imgs.to(device, non_blocking=True)
                batch = {k: v.to(device) for k, v in batch.items()}
                preds = model(imgs)
                loss, _ = criterion(preds, batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"✅ Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print(f"💾 Model saved (best val loss = {best_val_loss:.4f})")


def train_with_dfl(epochs: int = 300):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------- 로그 파일 생성 -------------------------
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"training_log_{now}.txt"
    log_file = open(log_path, "w", encoding="utf-8")  # ### 수정됨

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    folder_path = f"train_{now}"
    os.makedirs(folder_path, exist_ok=True)
    # ---------------------------------------------------------------

    # 모델 초기화
    base_model = YOLO11_MobileNetV4_DFL(num_classes=80).to(device)

    # DataParallel
    if torch.cuda.device_count() > 1:
        log(f"🔹 Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    # ✅ EMA 모델 초기화 (base_model 기준)
    ema = ModelEMA(base_model, decay=0.9999)  # 🔸 EMA 추가

    # Optim / Loss
    base_lr = 1e-3
    # optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.937,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=epochs, eta_min=base_lr * 0.1
    )
    criterion = DetectionLoss(model=base_model)

    train_loader = get_coco_dataset_train_loader(img_size=640)
    val_loader = get_coco_dataset_val_loader(img_size=640)

    best_val_loss = float("inf")
    early_stopping = EarlyStopping(patience=20, min_delta=0.0)

    warm_up_epochs = 3

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training")

        for imgs, batch in pbar:
            imgs = imgs.to(device, non_blocking=True)
            batch = {k: v.to(device) for k, v in batch.items()}

            preds = model(imgs)
            loss, (lbox, lobj, lcls) = criterion(preds, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 🔹 step마다 EMA 업데이트
            if isinstance(model, nn.DataParallel):
                ema.update(model.module)
            else:
                ema.update(model)

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if epoch >= warm_up_epochs:
            scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        log(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        # ---------------- Validation ----------------
        model.eval()
        val_loss, total_map50, total_map5095, count = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for imgs, batch in tqdm(val_loader, desc=f"Epoch [{epoch+1}] Validation"):
                imgs = imgs.to(device, non_blocking=True)
                batch = {k: v.to(device) for k, v in batch.items()}
                B, _, H, W = imgs.shape  # 보통 640x640

                preds = model(imgs)
                loss, _ = criterion(preds, batch)
                val_loss += loss.item()

                # 🔸 inference + mAP 계산
                results = (
                    model.module.predict(imgs)
                    if isinstance(model, nn.DataParallel)
                    else model.predict(imgs)
                )

                for b in range(len(results)):
                    pred = results[b]
                    gt_mask = batch["batch_idx"] == b
                    if gt_mask.sum() == 0:
                        continue

                    gt_boxes = batch["bboxes"][gt_mask]
                    gt_classes = batch["cls"][gt_mask].squeeze(1).long()

                    # 정규화된 gt (0~1) → pixel 좌표로 변환
                    if gt_boxes.max() <= 1.5:  # 0~1 범위일 때만 변환
                        gt_boxes_xyxy = xywh_norm_to_xyxy_abs(gt_boxes, H, W)
                    else:
                        gt_boxes_xyxy = gt_boxes  # 이미 pixel 스케일인 경우

                    map50, map5095 = compute_map(
                        pred["boxes"],  # [Np, 4] xyxy pixel
                        pred["scores"],  # [Np]
                        pred["classes"].long(),  # [Np]
                        gt_boxes_xyxy,  # [Ng, 4] xyxy pixel
                        gt_classes,  # [Ng]
                    )

                    total_map50 += map50
                    total_map5095 += map5095
                    count += 1

        avg_val_loss = val_loss / len(val_loader)
        avg_map50 = total_map50 / max(1, count)
        avg_map5095 = total_map5095 / max(1, count)

        log(
            f"✅ Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | mAP@50: {avg_map50:.4f} | mAP@50-95: {avg_map5095:.4f}"
        )

        # ---------------- Save Best ----------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(folder_path, "mn4cl_yolo11_dfl_best_model.pt")
            torch.save(
                (
                    model.module.state_dict()
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict()
                ),
                save_path,
            )
            log(
                f"💾 Best Model Saved → {save_path} (best val loss = {best_val_loss:.4f})"
            )
            # EMA 모델(best)
            best_ema_path = os.path.join(
                folder_path, "mn4cl_yolo11_dfl_best_model_ema.pt"
            )
            torch.save(ema.ema.state_dict(), best_ema_path)
            log(f"💾 Best EMA Model Saved → {best_ema_path}")

        if epoch == 100:
            save_path = os.path.join(folder_path, "mn4cl_yolo11_dfl_100_epoch_model.pt")
            torch.save(
                (
                    model.module.state_dict()
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict()
                ),
                save_path,
            )
            log(
                f"💾 Best Model Saved → {save_path} (best val loss = {best_val_loss:.4f})"
            )
            # EMA 모델(best)
            epoch_100_ema_path = os.path.join(
                folder_path, "mn4cl_yolo11_dfl_100_epoch_model_ema.pt"
            )
            torch.save(ema.ema.state_dict(), epoch_100_ema_path)
            log(f"💾 Best EMA Model Saved → {epoch_100_ema_path}")

        # ---------------- Always Save Last ----------------
        save_last = os.path.join(folder_path, "mn4cl_yolo11_dfl_last_model.pt")
        torch.save(
            (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            ),
            save_last,
        )
        log(f"💾 Last Model Saved → {save_last}")

        last_ema_path = os.path.join(folder_path, "mn4cl_yolo11_dfl_last_model_ema.pt")
        torch.save(ema.ema.state_dict(), last_ema_path)
        log(f"💾 Last EMA Model Saved → {last_ema_path}")
        if early_stopping.step(avg_val_loss):
            log(
                f"⏹ Early stopping at epoch {epoch+1} (no improvement for {early_stopping.patience} epochs)"
            )
            break

    log_file.close()  # 로그 파일 닫기


def train_overfit_50(
    epochs: int = 100,
    num_samples: int = 50,
    num_classes: int = 80,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------ 저장 폴더 생성 ------------
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"overfit50_{now}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 Checkpoints will be saved in: {save_dir}")

    # ---------------- 모델 ----------------
    base_model = YOLO11_MobileNetV4_DFL(num_classes).to(device)

    if torch.cuda.device_count() > 1:
        print(f"🔹 Using {torch.cuda.device_count()} GPUs for training (overfit 50)")
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    criterion = DetectionLoss(model=base_model)

    # 🔑 오버피팅 테스트라 lr 조금 크게
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # ---------------- 데이터 (50장만) ----------------
    train_loader = get_coco_debug_train_loader(num_samples=num_samples)

    best_loss = float("inf")
    best_path = os.path.join(save_dir, "overfit50_best.pt")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[Overfit 50] Epoch {epoch+1}/{epochs}")
        for imgs, batch in pbar:
            imgs = imgs.to(device, non_blocking=True)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            preds = model(imgs)
            loss, (lbox, lobj, lcls) = criterion(preds, batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"[Overfit 50] Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")
        # ------------ best 모델 저장 (train loss 기준) ------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            state_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            torch.save(state_dict, best_path)
            print(f"💾 New BEST model saved (loss={best_loss:.4f}) → {best_path}")

        # ---------------- 같은 50장에 대해 mAP 찍어보기 (선택) ----------------
        # 실험할 때는 5epoch마다 정도만 찍어도 됨
        if (epoch + 1) % 5 == 0:
            model.eval()
            total_map50, total_map5095, count = 0.0, 0.0, 0
            with torch.no_grad():
                for imgs, batch in train_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    B, _, H, W = imgs.shape  # 보통 640x640

                    # predict()는 base_model 기준으로 호출
                    if isinstance(model, nn.DataParallel):
                        results = model.module.predict(imgs)
                    else:
                        results = model.predict(imgs)

                    for b in range(len(results)):
                        pred = results[b]
                        gt_mask = batch["batch_idx"] == b
                        if gt_mask.sum() == 0:
                            continue

                        gt_boxes = batch["bboxes"][gt_mask]
                        gt_classes = batch["cls"][gt_mask].squeeze(1).long()

                        # 정규화된 gt (0~1) → pixel 좌표로 변환
                        if gt_boxes.max() <= 1.5:  # 0~1 범위일 때만 변환
                            gt_boxes_xyxy = xywh_norm_to_xyxy_abs(gt_boxes, H, W)
                        else:
                            gt_boxes_xyxy = gt_boxes  # 이미 pixel 스케일인 경우

                        map50, map5095 = compute_map(
                            pred["boxes"],  # [Np, 4] xyxy pixel
                            pred["scores"],  # [Np]
                            pred["classes"].long(),  # [Np]
                            gt_boxes_xyxy,  # [Ng, 4] xyxy pixel
                            gt_classes,  # [Ng]
                        )

                        total_map50 += map50
                        total_map5095 += map5095
                        count += 1

            avg_map50 = total_map50 / max(1, count)
            avg_map5095 = total_map5095 / max(1, count)
            print(
                f"    🔍 (train 50장 기준) mAP@50: {avg_map50:.4f}, "
                f"mAP@50-95: {avg_map5095:.4f}"
            )

    # ------------ 마지막 모델도 별도로 저장 ------------
    last_path = os.path.join(save_dir, "overfit50_last.pt")
    state_dict = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    torch.save(state_dict, last_path)
    print(f"💾 Last model saved → {last_path}")
    print(f"✅ Best model path: {best_path}")


if __name__ == "__main__":
    train_with_dfl(epochs=300)
    # train_overfit_50(epochs=100, num_samples=50)
