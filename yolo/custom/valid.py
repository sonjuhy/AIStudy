from pathlib import Path
from tqdm import tqdm
from yolo.custom.util.dataset import get_coco_dataset_val_loader, get_coco_debug_train_loader
from models.mobilenetv4_dfl import YOLO11_MobileNetV4_DFL
from yolo.custom.util.utils import compute_map, xywh_norm_to_xyxy_abs
from util.loss import DetectionLoss

import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def plt_settings(rcparams=None, backend="Agg"):
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.

    Example:
        decorator: @plt_settings({"font.size": 12})
        context manager: with plt_settings({"font.size": 12}):

    Args:
        rcparams (dict): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use. Defaults to 'Agg'.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend. This decorator can be
            applied to any function that needs to have specific matplotlib rc parameters and backend for its execution.
    """
    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Sets rc parameters and backend, calls the original function, and restores the settings."""
            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close(
                    "all"
                )  # auto-close()ing of figures upon backend switching is deprecated since 3.8
                plt.switch_backend(backend)

            # Plot with backend and always revert to original backend
            try:
                with plt.rc_context(rcparams):
                    result = func(*args, **kwargs)
            finally:
                if switch:
                    plt.close("all")
                    plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


@plt_settings()
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names={}, on_plot=None):
    """Plots a precision-recall curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(
                px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}"
            )  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(
        px,
        py.mean(1),
        linewidth=3,
        color="blue",
        label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


@plt_settings()
def plot_mc_curve(
    px,
    py,
    save_dir=Path("mc_curve.png"),
    names={},
    xlabel="Confidence",
    ylabel="Metric",
    on_plot=None,
):
    """Plots a metric-confidence curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(
        px,
        y,
        linewidth=3,
        color="blue",
        label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(
    tp,
    conf,
    pred_cls,
    target_cls,
    plot=False,
    on_plot=None,
    save_dir=Path(),
    names={},
    eps=1e-16,
    prefix="",
):

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = (
        np.zeros((nc, tp.shape[1])),
        np.zeros((nc, 1000)),
        np.zeros((nc, 1000)),
    )
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = np.interp(
            -x, -conf[i], recall[:, 0], left=0
        )  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values)  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = [
        v for k, v in names.items() if k in unique_classes
    ]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(
            x,
            prec_values,
            ap,
            save_dir / f"{prefix}PR_curve.png",
            names,
            on_plot=on_plot,
        )
        plot_mc_curve(
            x,
            f1_curve,
            save_dir / f"{prefix}F1_curve.png",
            names,
            ylabel="F1",
            on_plot=on_plot,
        )
        plot_mc_curve(
            x,
            p_curve,
            save_dir / f"{prefix}P_curve.png",
            names,
            ylabel="Precision",
            on_plot=on_plot,
        )
        plot_mc_curve(
            x,
            r_curve,
            save_dir / f"{prefix}R_curve.png",
            names,
            ylabel="Recall",
            on_plot=on_plot,
        )

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = (
        p_curve[:, i],
        r_curve[:, i],
        f1_curve[:, i],
    )  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return (
        tp,
        fp,
        p,
        r,
        f1,
        ap,
        unique_classes.astype(int),
        p_curve,
        r_curve,
        f1_curve,
        x,
        prec_values,
    )


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(
        0
    ).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def process_batch(preds, targets, iou_thres=0.5):
    """
    preds: [N, 6] = [cls, conf, x1, y1, x2, y2]
    targets: [M, 5] = [cls, x1, y1, x2, y2]
    iou_thres: IoU 기준 (기본 0.5)
    """
    correct = torch.zeros(preds.shape[0], dtype=torch.bool)
    tcls = targets[:, 0]

    # IoU 계산
    iou = box_iou(preds[:, 2:], targets[:, 1:])  # [N_pred, N_gt]
    iou_max, iou_idx = iou.max(1)

    # 매칭 조건
    for i, (im, idx) in enumerate(zip(iou_max, iou_idx)):
        if im > iou_thres and preds[i, 0] == tcls[idx]:
            correct[i] = True
            targets[idx, 0] = -1  # 중복 매칭 방지

    # ap_per_class()로 넘길 형식 반환
    return correct, preds[:, 1], preds[:, 0], tcls


def evaluate_map50(model, val_loader, device="cuda"):
    model.eval()
    ious_all, confs_all, preds_all, targets_all = [], [], [], []

    with torch.no_grad():
        for imgs, batch in tqdm(val_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            preds = model.predict(imgs)

            for i, pred in enumerate(preds):
                gt_boxes = batch["bboxes"]
                gt_cls = batch["cls"].squeeze(1)
                pred_boxes = pred["boxes"].cpu()
                pred_cls = pred["classes"].cpu()
                pred_scores = pred["scores"].cpu()

                preds_all.append(
                    torch.cat(
                        (pred_cls.unsqueeze(1), pred_scores.unsqueeze(1), pred_boxes), 1
                    )
                )
                targets_all.append(torch.cat((gt_cls.unsqueeze(1), gt_boxes), 1))

    # mAP 계산
    p, r, ap, f1, ap_class = ap_per_class(*process_batch(preds_all, targets_all))
    print(f"mAP@50: {ap[:,0].mean():.4f}")
    print(f"mAP@50-95: {ap.mean():.4f}")
    return ap[:, 0].mean(), ap.mean()


@torch.no_grad()
def evaluate_model_on_loader(
    model: YOLO11_MobileNetV4_DFL, data_loader, device="cuda", desc="Eval"
):
    device = device if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    total_loss = 0.0
    total_map50, total_map5095, count = 0.0, 0.0, 0

    criterion = DetectionLoss(model=model)  # 학습 때 쓰던 그대로

    pbar = tqdm(data_loader, desc=desc)

    for imgs, batch in pbar:
        imgs = imgs.to(device, non_blocking=True)
        batch = {k: v.to(device) for k, v in batch.items()}

        B, _, H, W = imgs.shape  # 보통 640x640

        # ------- loss -------
        preds = model(imgs)
        loss, _ = criterion(preds, batch)
        total_loss += loss.item()

        # ------- mAP -------
        results = model.predict(imgs, conf_thres=0.001)  # pred boxes: xyxy in (0~640)

        for b in range(len(results)):
            pred = results[b]
            gt_mask = batch["batch_idx"] == b
            if gt_mask.sum() == 0:
                continue

            gt_boxes = batch["bboxes"][gt_mask]  # [Ng, 4] xywh_norm
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

    avg_loss = total_loss / max(1, len(data_loader))
    avg_map50 = total_map50 / max(1, count)
    avg_map5095 = total_map5095 / max(1, count)

    print("==============================================")
    print(f" Val Loss   : {avg_loss:.4f}")
    print(f" mAP@50     : {avg_map50:.4f}")
    print(f" mAP@50-95  : {avg_map5095:.4f}")
    print("==============================================")

    return avg_loss, avg_map50, avg_map5095


def eval_checkpoint_on_train50(
    checkpoint_path: str,
    num_classes: int = 80,
    num_samples: int = 50,
    device: str = "cuda",
):
    device = device if torch.cuda.is_available() else "cpu"

    model = YOLO11_MobileNetV4_DFL(num_classes=80)
    model.load_state_dict(
        torch.load(
            os.path.join("overfit50_20251119_175857", checkpoint_path),
            map_location="cuda",
        )
    )
    model.to("cuda").eval()

    train50_loader = get_coco_debug_train_loader(num_samples=num_samples)

    evaluate_model_on_loader(
        model,
        train50_loader,
        device=device,
        desc=f"Eval on TRAIN-50 ({checkpoint_path})",
    )


if __name__ == "__main__":
    model = YOLO11_MobileNetV4_DFL(num_classes=80)
    model.load_state_dict(
        torch.load("mn4cl_yolo11_dfl_best_model.pt", map_location="cuda")
    )
    model.to("cuda").eval()
    val_loader = get_coco_dataset_val_loader()
    evaluate_map50(model, val_loader, device="cuda")
    # eval_checkpoint_on_train50("overfit50_best.pt")
