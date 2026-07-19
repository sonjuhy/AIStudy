import datetime
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs

from model import StandardEMA, MobileVisionNet
from loss import StandardDetectionLoss
from dataset import (
    get_coco_dataset_train_loader,
    get_coco_dataset_val_loader,
    get_coco_debug_train_loader,
)
from torch.utils.data import DataLoader
from utils import compute_map, xywh_norm_to_xyxy_abs


def train_clean_room_with_ddp(epochs: int = 500) -> None:
    # 1. Accelerator 초기화
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator: Accelerator = Accelerator(
        mixed_precision="fp16", kwargs_handlers=[ddp_kwargs]
    )
    device: torch.device = accelerator.device

    # ------------------------- 로그 파일 생성 (메인 프로세스만) -------------------------
    def log(msg: str) -> None:
        if accelerator.is_main_process:
            print(msg)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    if accelerator.is_main_process:
        now: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_path: str = f"train_{now}"
        os.makedirs(folder_path, exist_ok=True)
        log_path: str = os.path.join(folder_path, "training_log.txt")
        log(f"Using {accelerator.num_processes} GPUs for DDP training")

    # ---------------------------------------------------------------
    # 2. 모델 초기화 (SyncBatchNorm 적용)
    base_model: nn.Module = MobileVisionNet(num_classes=80).to(device)

    # 🔥 [수술 1] Focal Loss / BCE Loss 폭발 방지를 위한 초기 편향(Bias) 설정
    prior_prob: float = 0.01
    bias_value: float = -math.log((1.0 - prior_prob) / prior_prob)

    for m in base_model.modules():
        if isinstance(m, nn.Conv2d):
            # 출력 채널이 80(클래스 수)과 연관된 최종 헤드 레이어의 편향을 초기화
            if m.bias is not None and m.out_channels >= 80:
                m.bias.data[-80:].fill_(bias_value)

    if accelerator.num_processes > 1:
        base_model = nn.SyncBatchNorm.convert_sync_batchnorm(base_model)

    # 3. 클린룸 EMA 초기화 (메인 프로세스에서만)
    ema = StandardEMA(base_model, decay=0.9999) if accelerator.is_main_process else None

    # 4. Optimizer 및 스케줄러 설정
    g0, g1, g2 = [], [], []
    for v in base_model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            g2.append(v.bias)
        if isinstance(v, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            g0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            g1.append(v.weight)

    weight_decay: float = 1e-5
    optimizer: optim.SGD = optim.SGD(g0, lr=1e-3, momentum=0.937, nesterov=True)
    optimizer.add_param_group({"params": g1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": g2})

    warm_up_epochs: int = 3

    def lr_lambda(epoch: int) -> float:
        if epoch < warm_up_epochs:
            return (epoch + 1) / warm_up_epochs
        progress: float = (epoch - warm_up_epochs) / (epochs - warm_up_epochs)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    criterion: StandardDetectionLoss = StandardDetectionLoss(
        nc=80, stride=[8, 16, 32], device=device
    )

    train_loader: DataLoader = get_coco_dataset_train_loader(img_size=640)
    val_loader: DataLoader = get_coco_dataset_val_loader(img_size=640)

    best_fitness: float = 0.0

    # model, optimizer, train_loader, scheduler = accelerator.prepare(
    #     base_model, optimizer, train_loader, scheduler
    # )
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        base_model, optimizer, train_loader, val_loader, scheduler
    )
    # ------------------------- 학습 루프 -------------------------
    for epoch in range(epochs):
        model.train()
        train_loss: float = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{epochs}]",
            disable=not accelerator.is_main_process,
        )

        for imgs, batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            preds: list[torch.Tensor] = model(imgs)
            loss, loss_items = criterion(preds, batch)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            if accelerator.is_main_process and ema is not None:
                ema.update(accelerator.unwrap_model(model))

            # 🔥 [수술 1] DDP 환경에서 모든 GPU의 Loss를 모아서 정확한 평균 계산
            gathered_loss = accelerator.gather(loss.detach()).mean().item()
            train_loss += gathered_loss

            if accelerator.is_main_process:
                pbar.set_postfix(
                    {
                        "loss": f"{gathered_loss:.4f}",
                        "box": f"{loss_items[0].item():.4f}",
                        "cls": f"{loss_items[1].item():.4f}",
                    }
                )

        scheduler.step()
        avg_train_loss: float = train_loss / len(train_loader)
        accelerator.wait_for_everyone()
        # ------------------------- 평가 및 저장 루프 (분산 평가 적용) -------------------------
        # 1. 모든 GPU가 모델을 평가 모드로 전환
        unwrapped_model = (
            ema.ema if ema is not None else accelerator.unwrap_model(model)
        )
        unwrapped_model.eval()

        local_val_loss: torch.Tensor = torch.tensor(0.0, device=device)
        local_preds_list: list[dict[str, torch.Tensor]] = []
        local_targets_list: list[dict[str, torch.Tensor]] = []

        with torch.no_grad():
            # 2. 모든 GPU가 각자 할당된 절반의 검증 데이터를 병렬로 처리
            pbar_val = tqdm(
                val_loader,
                desc=f"Epoch [{epoch+1}] Val",
                disable=not accelerator.is_main_process,
                leave=False,
            )

            for imgs, batch in pbar_val:
                imgs = imgs.to(device, non_blocking=True)
                batch = {k: v.to(device) for k, v in batch.items()}
                B, _, H, W = imgs.shape

                preds: list[torch.Tensor] = unwrapped_model(imgs)
                loss, _ = criterion(preds, batch)
                local_val_loss += loss.item()

                results: list[dict[str, torch.Tensor]] = unwrapped_model.predict(
                    imgs, conf_thres=0.01, iou_thres=0.6
                )

                # 3. 각 GPU의 로컬 리스트에 결과 누적 (OOM 방지를 위해 CPU로 이동)
                for b in range(B):
                    pred: dict[str, torch.Tensor] = results[b]
                    local_preds_list.append(
                        {
                            "boxes": pred["boxes"].detach().cpu(),
                            "scores": pred["scores"].detach().cpu(),
                            "labels": pred["labels"].detach().cpu().long(),
                        }
                    )

                    gt_mask: torch.Tensor = batch["batch_idx"] == b
                    gt_boxes: torch.Tensor = batch["bboxes"][gt_mask]
                    gt_classes: torch.Tensor = batch["cls"][gt_mask].squeeze(-1).long()

                    if gt_mask.sum() > 0:
                        if gt_boxes.max() <= 1.5:
                            gt_boxes_xyxy: torch.Tensor = xywh_norm_to_xyxy_abs(
                                gt_boxes, H, W
                            )
                        else:
                            gt_boxes_xyxy = gt_boxes

                        local_targets_list.append(
                            {
                                "boxes": gt_boxes_xyxy.detach().cpu(),
                                "labels": gt_classes.detach().cpu(),
                            }
                        )
                    else:
                        local_targets_list.append(
                            {
                                "boxes": torch.empty((0, 4), dtype=torch.float32),
                                "labels": torch.empty((0,), dtype=torch.long),
                            }
                        )

        gathered_preds: list[dict[str, torch.Tensor]] = []
        gathered_targets: list[dict[str, torch.Tensor]] = []

        if accelerator.num_processes > 1:
            preds_container: list[list[dict[str, torch.Tensor]]] = [
                None for _ in range(accelerator.num_processes)
            ]
            targets_container: list[list[dict[str, torch.Tensor]]] = [
                None for _ in range(accelerator.num_processes)
            ]

            dist.all_gather_object(preds_container, local_preds_list)
            dist.all_gather_object(targets_container, local_targets_list)

            # 리스트 병합
            for sublist in preds_container:
                gathered_preds.extend(sublist)
            for sublist in targets_container:
                gathered_targets.extend(sublist)
        else:
            gathered_preds = local_preds_list
            gathered_targets = local_targets_list

        # 전체 Val Loss 계산
        global_val_loss: float = accelerator.gather(local_val_loss).sum().item() / len(
            val_loader.dataset
        )

        # ------------------------- 최종 mAP 연산 및 저장 (메인 프로세스 전담) -------------------------
        if accelerator.is_main_process:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision

            # 🔥 [수술 3] dist_sync_on_step=False, sync_on_compute=False 옵션을 강제로 부여하여 내부 동기화 로직 차단
            # CPU 상에서 이미 모인 글로벌 데이터를 바탕으로 mAP 계산
            metric = MeanAveragePrecision(
                box_format="xyxy",
                iou_type="bbox",
                dist_sync_on_step=False,  # 자동 동기화 끄기
                sync_on_compute=False,  # 자동 동기화 끄기
            )

            metric.update(gathered_preds, gathered_targets)

            metric_result: dict[str, torch.Tensor] = metric.compute()
            avg_map50: float = metric_result["map_50"].item()
            avg_map5095: float = metric_result["map"].item()

            log(
                f"✅ Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {global_val_loss:.4f} | mAP@50: {avg_map50:.4f} | mAP@50-95: {avg_map5095:.4f}"
            )

            fitness: float = (avg_map50 * 0.1) + (avg_map5095 * 0.9)
            model_state = unwrapped_model.state_dict()

            if fitness > best_fitness:
                best_fitness = fitness
                save_path: str = os.path.join(folder_path, "mobile_vision_net_best.pt")
                torch.save(model_state, save_path)
                log(f"💾 Best Model Saved → {save_path} (fitness = {best_fitness:.4f})")

                if ema is not None:
                    torch.save(
                        ema.ema.state_dict(),
                        os.path.join(folder_path, "mobile_vision_net_ema_best.pt"),
                    )

            torch.save(
                model_state, os.path.join(folder_path, "mobile_vision_net_last.pt")
            )
            if ema is not None:
                torch.save(
                    ema.ema.state_dict(),
                    os.path.join(folder_path, "mobile_vision_net_ema_last.pt"),
                )

        # 모든 GPU가 메인 프로세스의 평가 및 저장이 끝날 때까지 대기
        accelerator.wait_for_everyone()
    #     # ------------------------- 평가 및 저장 루프 (메인 프로세스에서만) -------------------------
    #     if accelerator.is_main_process:
    #         unwrapped_model = (
    #             ema.ema if ema is not None else accelerator.unwrap_model(model)
    #         )
    #         unwrapped_model.eval()
    #         val_loss = 0.0

    #         from torchmetrics.detection.mean_ap import MeanAveragePrecision

    #         # CPU 메모리에 메트릭 객체를 생성하여 VRAM 누수 원천 차단
    #         metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    #         with torch.no_grad():
    #             for imgs, batch in tqdm(
    #                 val_loader, desc=f"Epoch [{epoch+1}] Validation", leave=False
    #             ):
    #                 imgs = imgs.to(device, non_blocking=True)
    #                 batch = {k: v.to(device) for k, v in batch.items()}
    #                 B, _, H, W = imgs.shape

    #                 preds: list[torch.Tensor] = unwrapped_model(imgs)
    #                 loss, _ = criterion(preds, batch)
    #                 val_loss += loss.item()

    #                 results: list[dict[str, torch.Tensor]] = unwrapped_model.predict(
    #                     imgs, conf_thres=0.01, iou_thres=0.6
    #                 )

    #                 preds_list = []
    #                 targets_list = []

    #                 for b in range(B):
    #                     pred: dict[str, torch.Tensor] = results[b]
    #                     # 🔥 [수술 2] VRAM 메모리 누수를 막기 위한 CPU Off-loading
    #                     preds_list.append(
    #                         {
    #                             "boxes": pred["boxes"].detach().cpu(),
    #                             "scores": pred["scores"].detach().cpu(),
    #                             "labels": pred["labels"].detach().cpu().long(),
    #                         }
    #                     )

    #                     gt_mask: torch.Tensor = batch["batch_idx"] == b
    #                     gt_boxes: torch.Tensor = batch["bboxes"][gt_mask]
    #                     gt_classes: torch.Tensor = (
    #                         batch["cls"][gt_mask].squeeze(-1).long()
    #                     )

    #                     if gt_mask.sum() > 0:
    #                         if gt_boxes.max() <= 1.5:
    #                             gt_boxes_xyxy: torch.Tensor = xywh_norm_to_xyxy_abs(
    #                                 gt_boxes, H, W
    #                             )
    #                         else:
    #                             gt_boxes_xyxy = gt_boxes

    #                         # 🔥 GT 데이터 역시 CPU로 이동
    #                         targets_list.append(
    #                             {
    #                                 "boxes": gt_boxes_xyxy.detach().cpu(),
    #                                 "labels": gt_classes.detach().cpu(),
    #                             }
    #                         )
    #                     else:
    #                         targets_list.append(
    #                             {
    #                                 "boxes": torch.empty((0, 4), dtype=torch.float32),
    #                                 "labels": torch.empty((0,), dtype=torch.long),
    #                             }
    #                         )

    #                 metric.update(preds_list, targets_list)

    #         # CPU에서 안전하고 정확하게 글로벌 mAP 계산
    #         metric_result = metric.compute()
    #         avg_map50 = metric_result["map_50"].item()
    #         avg_map5095 = metric_result["map"].item()
    #         avg_val_loss: float = val_loss / len(val_loader)

    #         log(
    #             f"✅ Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
    #             f"Val Loss: {avg_val_loss:.4f} | mAP@50: {avg_map50:.4f} | mAP@50-95: {avg_map5095:.4f}"
    #         )

    #         fitness: float = (avg_map50 * 0.1) + (avg_map5095 * 0.9)
    #         model_state = unwrapped_model.state_dict()

    #         if fitness > best_fitness:
    #             best_fitness = fitness
    #             save_path: str = os.path.join(folder_path, "mobile_vision_net_best.pt")
    #             torch.save(model_state, save_path)
    #             log(f"💾 Best Model Saved → {save_path} (fitness = {best_fitness:.4f})")

    #             if ema is not None:
    #                 torch.save(
    #                     ema.ema.state_dict(),
    #                     os.path.join(folder_path, "mobile_vision_net_ema_best.pt"),
    #                 )

    #         torch.save(
    #             model_state, os.path.join(folder_path, "mobile_vision_net_last.pt")
    #         )
    #         if ema is not None:
    #             torch.save(
    #                 ema.ema.state_dict(),
    #                 os.path.join(folder_path, "mobile_vision_net_ema_last.pt"),
    #             )

    #         metric.reset()

    # accelerator.wait_for_everyone()


if __name__ == "__main__":
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    train_clean_room_with_ddp(epochs=500)
