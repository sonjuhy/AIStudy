import datetime
import os
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm

from model import ModelEMA, YOLO11_MobileNetV4_DFL
from loss import DetectionLoss
from dataset import COCODetectionYOLOv8, get_coco_dataset_val_loader, yolo_v8_collate_fn
from utils import compute_map, xywh_norm_to_xyxy_abs
from torch.utils.data import DataLoader


from model import StandardEMA, MobileVisionNet
from loss import StandardDetectionLoss

import torch.optim as optim
from accelerate import Accelerator, DistributedDataParallelKwargs


def get_finetune_train_loader(img_size: int = 640) -> DataLoader:
    """
    🔥 파인튜닝 전용 데이터 로더: Mosaic과 Mixup을 강제로 비활성화합니다.
    """
    ROOT_PATH: str = os.path.join(
        os.sep,
        "home",
        "edint",
        "Ivern_home",
        "WorkSpace",
        "Python",
        "Yolo",
        "coco_dataset",
    )

    finetune_dataset = COCODetectionYOLOv8(
        img_dir=os.path.join(ROOT_PATH, "train2017"),
        ann_file=os.path.join(ROOT_PATH, "annotations", "instances_train2017.json"),
        img_size=img_size,
        cache=False,
        is_train=True,
        mosaic=False,  # 🔥 [핵심 1] 원본 이미지 스케일 보존을 위해 Mosaic 끄기
        mixup=False,  # 🔥 [핵심 2] 객체 혼합 끄기
        hsv_prob=0.3,  # 색상 변형 확률은 낮춰서 유지
        flip_prob=0.5,  # 좌우 반전은 유지
    )

    finetune_loader: DataLoader = DataLoader(
        finetune_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=yolo_v8_collate_fn,
    )
    return finetune_loader


def finetune_close_mosaic(
    weights_path: str,
    ema_weights_path: str,
    epochs: int = 15,
) -> None:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------- 로그 파일 생성 -------------------------
    now: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path: str = f"finetune_{now}"
    os.makedirs(folder_path, exist_ok=True)

    log_path: str = os.path.join(folder_path, "finetune_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log("🚀 Starting Fine-tuning (Close Mosaic) Phase...")

    # ------------------------- 모델 초기화 및 가중치 로드 -------------------------
    # backbone_size=1 (Small 모델 기준)
    base_model: nn.Module = YOLO11_MobileNetV4_DFL(num_classes=80, backbone_size=1).to(
        device
    )

    log(f"🔹 Loading 500-epoch trained weights from {weights_path}...")
    base_model.load_state_dict(torch.load(weights_path, map_location=device))

    # DataParallel
    model: nn.Module
    if torch.cuda.device_count() > 1:
        log(f"🔹 Using {torch.cuda.device_count()} GPUs for Fine-tuning")
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    # EMA 모델 초기화 및 기존 EMA 가중치 로드
    ema = ModelEMA(base_model, decay=0.9999)
    if os.path.exists(ema_weights_path):
        log(f"🔹 Loading EMA weights from {ema_weights_path}...")
        ema.ema.load_state_dict(torch.load(ema_weights_path, map_location=device))

    # ------------------------- 파인튜닝 전용 옵티마이저 세팅 -------------------------
    g0: List[nn.Parameter] = []
    g1: List[nn.Parameter] = []
    g2: List[nn.Parameter] = []

    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            g2.append(v.bias)
        if isinstance(v, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            g0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            g1.append(v.weight)

    # 🔥 [핵심 3] 스케줄러 삭제 및 학습률 1e-5로 극단적 고정
    finetune_lr: float = 1e-5
    optimizer: torch.optim.SGD = torch.optim.SGD(
        g0, lr=finetune_lr, momentum=0.937, nesterov=True
    )
    optimizer.add_param_group({"params": g1, "weight_decay": 5e-4})
    optimizer.add_param_group({"params": g2})

    criterion = DetectionLoss(model=base_model)

    train_loader: DataLoader = get_finetune_train_loader(img_size=640)
    val_loader: DataLoader = get_coco_dataset_val_loader(img_size=640)

    best_fitness: float = 0.0

    # ------------------------- 파인튜닝 루프 시작 -------------------------
    for epoch in range(epochs):
        model.train()
        train_loss: float = 0.0
        pbar = tqdm(train_loader, desc=f"Finetune Epoch [{epoch+1}/{epochs}]")

        for imgs, batch in pbar:
            imgs = imgs.to(device, non_blocking=True)
            batch = {k: v.to(device) for k, v in batch.items()}

            preds = model(imgs)
            loss, _ = criterion(preds, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA 업데이트
            if isinstance(model, nn.DataParallel):
                ema.update(model.module)
            else:
                ema.update(model)

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 주의: scheduler.step() 없음!
        avg_train_loss: float = train_loss / len(train_loader)
        log(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | LR: {finetune_lr:.6f} (Fixed)"
        )

        # ------------------------- Validation -------------------------
        model.eval()
        val_loss: float = 0.0
        total_map50: float = 0.0
        total_map5095: float = 0.0
        count: int = 0

        with torch.no_grad():
            for imgs, batch in tqdm(val_loader, desc=f"Epoch [{epoch+1}] Validation"):
                imgs = imgs.to(device, non_blocking=True)
                batch = {k: v.to(device) for k, v in batch.items()}
                B, _, H, W = imgs.shape

                preds = model(imgs)
                loss, _ = criterion(preds, batch)
                val_loss += loss.item()

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

                    if gt_boxes.max() <= 1.5:
                        gt_boxes_xyxy = xywh_norm_to_xyxy_abs(gt_boxes, H, W)
                    else:
                        gt_boxes_xyxy = gt_boxes

                    map50, map5095 = compute_map(
                        pred["boxes"],
                        pred["scores"],
                        pred["classes"].long(),
                        gt_boxes_xyxy,
                        gt_classes,
                    )

                    total_map50 += map50
                    total_map5095 += map5095
                    count += 1

        avg_val_loss: float = val_loss / len(val_loader)
        avg_map50: float = total_map50 / max(1, count)
        avg_map5095: float = total_map5095 / max(1, count)

        log(
            f"✅ Finetune {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | mAP@50: {avg_map50:.4f} | mAP@50-95: {avg_map5095:.4f}"
        )

        # ------------------------- Save Checkpoints -------------------------
        fitness: float = (avg_map50 * 0.1) + (avg_map5095 * 0.9)

        if fitness > best_fitness:
            best_fitness = fitness
            save_path: str = os.path.join(folder_path, "finetune_best_model.pt")
            torch.save(
                (
                    model.module.state_dict()
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict()
                ),
                save_path,
            )
            log(
                f"💾 Best Finetuned Model Saved → {save_path} (best fitness = {best_fitness:.4f})"
            )

            best_ema_path: str = os.path.join(folder_path, "finetune_best_model_ema.pt")
            torch.save(ema.ema.state_dict(), best_ema_path)

        save_last: str = os.path.join(folder_path, "finetune_last_model.pt")
        torch.save(
            (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            ),
            save_last,
        )

        last_ema_path: str = os.path.join(folder_path, "finetune_last_model_ema.pt")
        torch.save(ema.ema.state_dict(), last_ema_path)

    log_file.close()


def finetune_close_mosaic_mobilenet_vision(
    weights_path: str, ema_weights_path: str, epochs: int = 15, multi_gpu: bool = True
):
    # finetune_lr: float = 1e-5
    # 1. Accelerator 초기화 (기본적으로 환경을 감지하여 Multi-GPU 여부 파악)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator: Accelerator = Accelerator(
        mixed_precision="fp16", kwargs_handlers=[ddp_kwargs]
    )
    device: torch.device = (
        accelerator.device
    )  # 수동 device 대신 accelerator가 지정해준 디바이스 사용

    # ------------------------- 로그 파일 생성 (메인 프로세스만) -------------------------
    def log(msg: str) -> None:
        if accelerator.is_main_process:
            print(msg)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    if accelerator.is_main_process:
        now: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_path: str = f"finetune_{now}"
        os.makedirs(folder_path, exist_ok=True)
        log_path = os.path.join(folder_path, "finetune_log.txt")
        log(f"Using {accelerator.num_processes} GPUs for DDP training")

    # ---------------------------------------------------------------
    # 2. 모델 초기화 (SyncBatchNorm 적용)
    base_model: nn.Module = MobileVisionNet(num_classes=80).to(device)
    if accelerator.num_processes > 1:
        base_model = nn.SyncBatchNorm.convert_sync_batchnorm(base_model)

    log(f"🔹 Loading 500-epoch trained weights from {weights_path}...")
    base_model.load_state_dict(torch.load(weights_path, map_location=device))

    # 3. 클린룸 EMA 초기화 (메인 프로세스에서만)
    if multi_gpu:
        # (이전 코드에서 multi_gpu일 때 전체에 EMA를 할당하던 로직을 지우고 통일하는 것이 좋습니다)
        ema = (
            StandardEMA(base_model, decay=0.9999)
            if accelerator.is_main_process
            else None
        )
    else:
        ema = (
            StandardEMA(base_model, decay=0.9999)
            if accelerator.is_main_process
            else None
        )

    # 🔥 [수정] ema가 None이 아닐 때(즉, 메인 프로세스일 때)만 가중치를 로드하도록 안전장치 추가
    if ema is not None and os.path.exists(ema_weights_path):
        log(f"🔹 Loading EMA weights from {ema_weights_path}...")
        ema.ema.load_state_dict(torch.load(ema_weights_path, map_location=device))

    # 4. Optimizer 및 스케줄러 설정

    # ------------------------- 파인튜닝 전용 옵티마이저 세팅 -------------------------
    g0: List[nn.Parameter] = []
    g1: List[nn.Parameter] = []
    g2: List[nn.Parameter] = []
    for v in base_model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            g2.append(v.bias)
        if isinstance(v, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            g0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            g1.append(v.weight)

    finetune_lr: float = 1e-5
    optimizer: optim.SGD = optim.SGD(g0, lr=finetune_lr, momentum=0.937, nesterov=True)
    optimizer.add_param_group({"params": g1, "weight_decay": 5e-4})
    optimizer.add_param_group({"params": g2})

    # warm_up_epochs: int = 3

    # def lr_lambda(epoch: int) -> float:
    #     if epoch < warm_up_epochs:
    #         return (epoch + 1) / warm_up_epochs
    #     progress: float = (epoch - warm_up_epochs) / (epochs - warm_up_epochs)
    #     return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    criterion: StandardDetectionLoss = StandardDetectionLoss(
        nc=80, stride=[8, 16, 32], device=device
    )

    train_loader: DataLoader = get_finetune_train_loader(img_size=640)
    val_loader: DataLoader = get_coco_dataset_val_loader(img_size=640)

    best_fitness: float = 0.0

    # 여기서 모델이 DDP로 래핑되고, 데이터 로더가 GPU별로 쪼개집니다.
    model, optimizer, train_loader = accelerator.prepare(
        base_model, optimizer, train_loader
    )

    # ------------------------- 학습 루프 -------------------------
    for epoch in range(epochs):
        model.train()
        train_loss: float = 0.0

        # tqdm은 메인 프로세스에서만 표시
        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{epochs}]",
            disable=not accelerator.is_main_process,
        )

        for imgs, batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            preds: list[torch.Tensor] = model(imgs)
            loss, loss_items = criterion(preds, batch)

            # Accelerate가 스케일링 및 역전파 자동 관리
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

            if accelerator.is_main_process and ema is not None:
                ema.update(accelerator.unwrap_model(model))

            train_loss += loss.item()

            if accelerator.is_main_process:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "box": f"{loss_items[0].item():.4f}",
                        "cls": f"{loss_items[1].item():.4f}",
                    }
                )

        # scheduler.step()
        avg_train_loss: float = train_loss / len(train_loader)
        accelerator.wait_for_everyone()

        # ------------------------- 평가 및 저장 루프 (메인 프로세스에서만) -------------------------
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.eval()
            val_loss, total_map50, total_map5095, count = 0.0, 0.0, 0.0, 0

            with torch.no_grad():
                for imgs, batch in tqdm(
                    val_loader, desc=f"Epoch [{epoch+1}] Validation"
                ):
                    imgs = imgs.to(device, non_blocking=True)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    B, _, H, W = imgs.shape

                    preds: list[torch.Tensor] = unwrapped_model(imgs)
                    loss, _ = criterion(preds, batch)
                    val_loss += loss.item()

                    results: list[dict[str, torch.Tensor]] = unwrapped_model.predict(
                        imgs
                    )

                    for b in range(len(results)):
                        pred: dict[str, torch.Tensor] = results[b]
                        gt_mask: torch.Tensor = batch["batch_idx"] == b
                        if gt_mask.sum() == 0:
                            continue

                        gt_boxes: torch.Tensor = batch["bboxes"][gt_mask]
                        gt_classes: torch.Tensor = (
                            batch["cls"][gt_mask].squeeze(1).long()
                        )

                        if gt_boxes.max() <= 1.5:
                            gt_boxes_xyxy: torch.Tensor = xywh_norm_to_xyxy_abs(
                                gt_boxes, H, W
                            )
                        else:
                            gt_boxes_xyxy = gt_boxes

                        map50, map5095 = compute_map(
                            pred["boxes"],
                            pred["scores"],
                            pred["labels"],
                            gt_boxes_xyxy,
                            gt_classes,
                        )

                        total_map50 += map50
                        total_map5095 += map5095
                        count += 1

            avg_val_loss: float = val_loss / len(val_loader)
            avg_map50: float = total_map50 / max(1, count)
            avg_map5095: float = total_map5095 / max(1, count)

            log(
                f"✅ Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | mAP@50: {avg_map50:.4f} | mAP@50-95: {avg_map5095:.4f}"
            )

            fitness: float = (avg_map50 * 0.1) + (avg_map5095 * 0.9)
            model_state = unwrapped_model.state_dict()

            if fitness > best_fitness:
                best_fitness = fitness
                save_path: str = os.path.join(
                    folder_path, "mobile_vision_net_finetuning_best.pt"
                )
                torch.save(model_state, save_path)
                log(f"💾 Best Model Saved → {save_path} (fitness = {best_fitness:.4f})")

                if ema is not None:
                    torch.save(
                        ema.ema.state_dict(),
                        os.path.join(
                            folder_path, "mobile_vision_net_finetuning_ema_best.pt"
                        ),
                    )

            torch.save(
                model_state,
                os.path.join(folder_path, "mobile_vision_net_finetuning_last.pt"),
            )
            if ema is not None:
                torch.save(
                    ema.ema.state_dict(),
                    os.path.join(
                        folder_path, "mobile_vision_net_finetuning_ema_last.pt"
                    ),
                )

    # 환경이 종료될 때 Accelerator 해제 대기
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"  # 통신 로그 출력 (문제가 생기면 로그가 찍힘)
    # 500 에포크 달성 시 기록되었던 최고 가중치(EMA 포함)의 경로를 입력해 주세요.
    # BEST_WEIGHTS_PATH: str = os.path.join(
    #     "train_20260303_101651", "mn4cs_yolo11_dfl_best_model.pt"
    # )
    # BEST_EMA_WEIGHTS_PATH: str = os.path.join(
    #     "train_20260303_101651", "mn4cs_yolo11_dfl_best_model_ema.pt"
    # )
    BEST_WEIGHTS_PATH: str = os.path.join(
        "train_20260401_154129", "mobile_vision_net_best.pt"
    )
    BEST_EMA_WEIGHTS_PATH: str = os.path.join(
        "train_20260401_154129", "mobile_vision_net_ema_best.pt"
    )

    # finetune_close_mosaic(
    #     weights_path=BEST_WEIGHTS_PATH,
    #     ema_weights_path=BEST_EMA_WEIGHTS_PATH,
    #     epochs=15,
    # )
    finetune_close_mosaic_mobilenet_vision(
        weights_path=BEST_WEIGHTS_PATH,
        ema_weights_path=BEST_EMA_WEIGHTS_PATH,
        epochs=30,
    )
