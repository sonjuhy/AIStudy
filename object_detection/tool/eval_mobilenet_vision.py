import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

# 사용자 모델 및 데이터 로더 import
from model import MobileVisionNet
from dataset import get_coco_dataset_val_loader

# COCO 지표 계산을 위한 공식 라이브러리
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def xywh_norm_to_xyxy_abs(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    정규화된 [0~1] 범위의 (cx, cy, width, height) 포맷을
    픽셀 단위의 (x1, y1, x2, y2) 포맷으로 변환합니다.
    """
    if boxes.numel() == 0:
        return torch.zeros((0, 4), device=boxes.device)

    cx, cy, bw, bh = boxes.unbind(1)
    x1: torch.Tensor = (cx - bw / 2.0) * w
    y1: torch.Tensor = (cy - bh / 2.0) * h
    x2: torch.Tensor = (cx + bw / 2.0) * w
    y2: torch.Tensor = (cy + bh / 2.0) * h

    return torch.stack([x1, y1, x2, y2], dim=1)


def evaluate_coco_format(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device | str = "cuda",
    save_path: str | None = None,
) -> None:
    model.eval()

    # torchmetrics 초기화 (COCO 표준: xyxy 포맷 기준)
    metric: MeanAveragePrecision = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox"
    )
    metric.to(device)

    with torch.no_grad():
        for imgs, batch in tqdm(val_loader, desc="Evaluating COCO Metrics"):
            imgs: torch.Tensor = imgs.to(device, non_blocking=True)
            B, _, H, W = imgs.shape

            # mAP 계산을 위해 conf_thres를 극단적으로 낮춤
            # MobileVisionNet 내부의 predict 함수가 NMS와 디코딩을 수행합니다.
            preds: list[dict[str, torch.Tensor]] = model.predict(
                imgs, conf_thres=0.001, iou_thres=0.6
            )

            preds_list: list[dict[str, torch.Tensor]] = []
            targets_list: list[dict[str, torch.Tensor]] = []

            for b in range(B):
                # ------------------- 1. Prediction 준비 -------------------
                pred = preds[b]
                preds_list.append(
                    {
                        "boxes": pred["boxes"].to(device),
                        "scores": pred["scores"].to(device),
                        "labels": pred["labels"].long().to(device),
                    }
                )

                # ------------------- 2. Ground Truth 준비 -------------------
                gt_mask: torch.Tensor = batch["batch_idx"] == b
                gt_boxes: torch.Tensor = batch["bboxes"][gt_mask].to(device)
                gt_classes: torch.Tensor = (
                    batch["cls"][gt_mask].squeeze(-1).long().to(device)
                )

                # 정규화된 xywh를 픽셀 기반 xyxy로 변환
                if gt_boxes.numel() > 0 and gt_boxes.max() <= 1.5:
                    gt_boxes_xyxy: torch.Tensor = xywh_norm_to_xyxy_abs(gt_boxes, H, W)
                else:
                    gt_boxes_xyxy = gt_boxes  # 이미 픽셀 단위라면 그대로 사용

                targets_list.append({"boxes": gt_boxes_xyxy, "labels": gt_classes})

            # 배치별로 평가 메트릭에 누적 업데이트
            metric.update(preds_list, targets_list)

    print("\n⏳ Computing final COCO metrics... This may take a moment.")
    result: dict[str, torch.Tensor] = metric.compute()

    # ------------------- 3. 결과 출력 (COCO Official Format) -------------------
    report_str: str = (
        f"\n{'=' * 60}\n"
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {result['map'].item():.3f}\n"
        f" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {result['map_50'].item():.3f}\n"
        f" Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {result['map_75'].item():.3f}\n"
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {result['map_small'].item():.3f}\n"
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {result['map_medium'].item():.3f}\n"
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {result['map_large'].item():.3f}\n"
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {result['mar_1'].item():.3f}\n"
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {result['mar_10'].item():.3f}\n"
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {result['mar_100'].item():.3f}\n"
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {result['mar_small'].item():.3f}\n"
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {result['mar_medium'].item():.3f}\n"
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {result['mar_large'].item():.3f}\n\n"
        f"➡️  mAP@50-95: {result['map'].item():.4f}\n"
        f"➡️  mAP@50:    {result['map_50'].item():.4f}\n"
        f"➡️  mAP@75:    {result['map_75'].item():.4f}\n"
        f"{'=' * 60}\n"
    )
    print(report_str)

    # 🔥 파일로 저장
    if save_path is not None:
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report_str)
            print(f"💾 Evaluation results successfully saved to: {save_path}")
        except Exception as e:
            print(f"⚠️ Failed to save results to {save_path}: {e}")


if __name__ == "__main__":
    # GPU 환경 최우선 사용 설정
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # 평가를 진행할 체크포인트 경로 입력
    model_root_path: str = os.path.join("train_20260406_151547")
    model_path: str = os.path.join(
        "train_20260406_151547", "mobile_vision_net_ema_best.pt"
    )
    # model_root_path: str = os.path.join("finetune_20260404_175243")
    # model_path: str = os.path.join(
    #     "finetune_20260404_175243", "mobile_vision_net_finetuning_ema_best.pt"
    # )

    # model_root_path_list: list[str] = [
    #     "finetune_20260331_232730",
    #     "finetune_20260401_100024",
    # ]
    # model_name: str = "mobile_vision_net_finetuning_ema_best.pt"
    # for model_root_path in model_root_path_list:
    #     model_path: str = os.path.join(model_root_path, model_name)
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    print("=" * 60 + "\n")

    # 🔥 모델 초기화 (추가 인자 제거)
    model: nn.Module = MobileVisionNet(num_classes=80)

    # 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = torch.compile(model, backend="inductor", mode="reduce-overhead")
    model.to(device)

    # 데이터 로더 생성
    val_loader: torch.utils.data.DataLoader = get_coco_dataset_val_loader(img_size=640)

    # evaluate_coco_format 실행 시간 측정
    start_time = time.time()

    # 평가 실행
    evaluate_coco_format(
        model,
        val_loader,
        device=device,
        save_path=os.path.join(model_root_path, "validation_result.txt"),
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total evaluation time: {elapsed_time:.2f} seconds")
