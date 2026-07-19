import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def empty_like(x):
    """Creates empty torch.Tensor or np.ndarray with same shape as input and float32 dtype."""
    return (
        torch.empty_like(x, dtype=torch.float32)
        if isinstance(x, torch.Tensor)
        else np.empty_like(x, dtype=np.float32)
    )


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert (
        x.shape[-1] == 4
    ), f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(
            b2_x1
        )  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2)
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return (
            iou - (c_area - union) / c_area
        )  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(
        0, reg_max - 0.01
    )  # dist (lt, rb)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = (
            feats[i].shape[2:]
            if isinstance(feats, list)
            else (int(feats[i][0]), int(feats[i][1]))
        )
        sx = (
            torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        )  # shift x
        sy = (
            torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        )  # shift y
        sy, sx = torch.meshgrid(sy, sx)

        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(
                pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
            )
        except torch.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            print("WARNING: CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [
                t.cpu()
                for t in (
                    pd_scores,
                    pd_bboxes,
                    anc_points,
                    gt_labels,
                    gt_bboxes,
                    mask_gt,
                )
            ]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes
        )

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(
            dim=-1, keepdim=True
        )  # b, max_num_obj
        norm_align_metric = (
            (align_metric * pos_overlaps / (pos_align_metrics + self.eps))
            .amax(-2)
            .unsqueeze(-1)
        )
        target_scores = target_scores * norm_align_metric

        return (
            target_labels,
            target_bboxes,
            target_scores,
            fg_mask.bool(),
            target_gt_idx,
        )

    def get_pos_mask(
        self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
    ):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(
            align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool()
        )
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros(
            [self.bs, self.n_max_boxes, na],
            dtype=pd_bboxes.dtype,
            device=pd_bboxes.device,
        )
        bbox_scores = torch.zeros(
            [self.bs, self.n_max_boxes, na],
            dtype=pd_scores.dtype,
            device=pd_scores.device,
        )

        ind = torch.zeros(
            [2, self.bs, self.n_max_boxes], dtype=torch.long
        )  # 2, b, max_num_obj
        ind[0] = (
            torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        )  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][
            mask_gt
        ]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for horizontal bounding boxes."""
        return (
            bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
        )

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(
            metrics, self.topk, dim=-1, largest=largest
        )
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(
                topk_idxs
            )
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(
            metrics.shape, dtype=torch.int8, device=topk_idxs.device
        )
        ones = torch.ones_like(
            topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device
        )
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(
            end=self.bs, dtype=torch.int64, device=gt_labels.device
        )[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(
            1, 1, self.num_classes
        )  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability. Defaults to 1e-9.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat(
            (xy_centers[None] - lt, rb - xy_centers[None]), dim=2
        ).view(bs, n_boxes, n_anchors, -1)
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).

        Note:
            b: batch size, h: height, w: width.
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(
                -1, n_max_boxes, -1
            )  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(
                mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device
            )
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(
                mask_multi_gts, is_max_overlaps, mask_pos
            ).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
            * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
            * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True
        )
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes, self.dfl_loss.reg_max - 1
            )
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model: nn.Module, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class StandardDetectionLoss:
    """
    DFL을 완전히 제거하고 FCOS 스타일의 4채널 직접 회귀 방식을 사용하는
    모바일 최적화 손실 함수 클래스.
    """

    def __init__(self, nc: int, stride: list[int], device: torch.device) -> None:
        self.nc: int = nc
        self.stride: list[int] = stride
        self.device: torch.device = device

        # DFL이 사라졌으므로 출력 채널은 (클래스 수 + 4개의 좌표)로 극도로 단순화됩니다.
        self.no: int = nc + 4

        # 1. Classification Loss (BCE)
        self.bce: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction="none")

        # 2. Label Assigner (학계 논문 Task-Aligned Assignment 범용 구현체 사용)
        self.assigner: AcademicTaskAlignedAssigner = AcademicTaskAlignedAssigner(
            topk=10,
            num_classes=self.nc,
            beta=6.0,
            # alpha=0.5,
        )

        # 하이퍼파라미터 (단순화)
        self.hyp: dict[str, float] = {"box": 7.5, "cls": 0.5}

    def bbox_decode_direct(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor
    ) -> torch.Tensor:
        """
        [원리 및 근거]
        DFL의 Softmax 행렬 곱 대신, 4채널(l, t, r, b) 텐서에 ReLU를 적용하여
        음수를 제거하고 앵커 중심점을 기준으로 즉시 절대 픽셀 좌표(x1, y1, x2, y2)로 복원합니다.
        """
        # 신경망 출력이 음수가 되는 것을 방지 (가장 가벼운 연산)
        pred_dist = pred_dist.relu()

        # (l, t) 및 (r, b)
        lt: torch.Tensor = pred_dist[..., :2]
        rb: torch.Tensor = pred_dist[..., 2:]

        # (x1, y1) = 중심점 - (l, t) / (x2, y2) = 중심점 + (r, b)
        x1y1: torch.Tensor = anchor_points - lt
        x2y2: torch.Tensor = anchor_points + rb

        return torch.cat([x1y1, x2y2], dim=-1)  # [B, N, 4]

    def __call__(
        self, preds: list[torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computational Graph 파괴를 방지하고 완벽한 역전파를 보장하는 클린룸 손실 계산 로직.
        """
        batch_size: int = preds[0].shape[0]
        dtype: torch.dtype = preds[0].dtype

        # 1. 예측값 텐서 분리 및 정렬
        pred_concat = torch.cat(
            [xi.view(batch_size, self.no, -1) for xi in preds], dim=2
        )
        pred_concat = pred_concat.permute(0, 2, 1).contiguous()
        pred_distri, pred_scores = pred_concat.split((4, self.nc), dim=-1)

        # 2. 앵커 및 타겟 준비
        anchor_points, stride_tensor = make_anchors(preds, self.stride, 0.5)

        targets: torch.Tensor = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            dim=1,
        ).to(self.device)

        feat_h: int = preds[0].shape[2]
        feat_w: int = preds[0].shape[3]
        imgsz: torch.Tensor = (
            torch.tensor([feat_h, feat_w], device=self.device, dtype=dtype)
            * self.stride[0]
        )
        scale_tensor: torch.Tensor = imgsz[[1, 0, 1, 0]]

        targets_padded = self._preprocess_targets(
            targets, batch_size, scale_tensor=scale_tensor
        )
        gt_labels, gt_bboxes = targets_padded.split((1, 4), dim=-1)
        mask_gt: torch.Tensor = gt_bboxes.sum(dim=-1, keepdim=True).gt_(0.0)

        # 3. 예측 박스 디코딩
        pred_bboxes: torch.Tensor = self.bbox_decode_direct(
            anchor_points * stride_tensor, pred_distri * stride_tensor
        )

        # 4. 정답 할당 (Label Assignment)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes.detach(),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # Python 내장 max 대신 PyTorch의 clamp를 사용하여 안전하게 분모 설정
        target_scores_sum: torch.Tensor = torch.clamp(target_scores.sum(), min=1.0)

        # 5. 손실 계산 (그래프 단절을 막기 위해 독립 텐서 연산 후 합산)
        # 5-1. Classification Loss
        loss_cls: torch.Tensor = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )

        # 5-2. Box Loss 초기화 (연산 그래프 포함)
        loss_box: torch.Tensor = torch.tensor(0.0, device=self.device, dtype=dtype)

        if fg_mask.sum() > 0:
            pred_bboxes_pos = pred_bboxes[fg_mask]
            target_bboxes_pos = target_bboxes[fg_mask]

            iou_loss: torch.Tensor = 1.0 - compute_ciou(
                pred_bboxes_pos, target_bboxes_pos
            )

            # 🔥 [B, A, 80] 차원에서 fg_mask 적용 시 [Num_fg, 80]이 됨.
            # 할당된 정답 클래스의 확률만 스칼라로 뽑아오기 위해 sum(dim=-1) 수행
            weight: torch.Tensor = target_scores[fg_mask].sum(dim=-1)

            loss_box = (iou_loss * weight).sum() / target_scores_sum

        # 6. 하이퍼파라미터 가중치 적용 및 최종 합산
        loss_box = loss_box * self.hyp["box"]
        loss_cls = loss_cls * self.hyp["cls"]

        total_loss: torch.Tensor = loss_box + loss_cls

        # 모니터링을 위해 detach된 텐서 리스트 반환
        return total_loss * batch_size, torch.stack([loss_box, loss_cls]).detach()

    def _preprocess_targets(
        self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        배치 내 객체 수를 맞추기 위한 제로 패딩(Zero-padding) 및
        정규화된 xywh 텐서를 절대 픽셀의 xyxy 텐서로 변환하는 범용 함수입니다.

        Args:
            targets: [N, 6] 형태의 텐서 (batch_idx, class_id, cx, cy, w, h)
            batch_size: 현재 미니 배치의 크기
            scale_tensor: [w, h, w, h] 형태의 스케일 텐서 (원본 이미지 해상도 복원용)
        """

        num_targets, num_elements = targets.shape

        # 1. 타겟이 아예 없는 예외 케이스 처리
        if num_targets == 0:
            return torch.zeros((batch_size, 0, num_elements - 1), device=self.device)

        # 2. 배치 내 최대 객체 수(Max GT) 산출
        batch_indices: torch.Tensor = targets[:, 0].long()
        _, counts = torch.unique(batch_indices, return_counts=True)
        max_gt_per_image: int = counts.max().item()

        # 3. 빈 패딩 텐서 할당 [B, Max_GT, 5] -> (class_id, x, y, w, h)
        padded_targets: torch.Tensor = torch.zeros(
            (batch_size, max_gt_per_image, num_elements - 1), device=self.device
        )

        # 4. 각 배치 인덱스에 맞게 타겟 텐서 할당
        for b in range(batch_size):
            mask: torch.Tensor = batch_indices == b
            valid_gt_count: int = mask.sum().item()
            if valid_gt_count > 0:
                padded_targets[b, :valid_gt_count] = targets[mask, 1:]

        # 5. 좌표 변환 로직 (정규화된 xywh -> 절대 픽셀 xyxy)
        # padded_targets의 [..., 1:5] 영역이 좌표 데이터입니다.
        boxes_xywh: torch.Tensor = padded_targets[..., 1:5] * scale_tensor

        # unbind를 사용하여 텐서 복사 없이 메모리 상에서 바로 분리 연산
        cx, cy, w, h = boxes_xywh.unbind(dim=-1)

        x1: torch.Tensor = cx - w / 2.0
        y1: torch.Tensor = cy - h / 2.0
        x2: torch.Tensor = cx + w / 2.0
        y2: torch.Tensor = cy + h / 2.0

        # 변환된 xyxy를 원래 위치에 덮어쓰기
        padded_targets[..., 1:5] = torch.stack([x1, y1, x2, y2], dim=-1)

        return padded_targets


class KDDetectionLoss(nn.Module):
    """
    정답(GT) 기반의 기본 Detection Loss와 선생 모델 로짓 기반의 KD Loss를 결합한 손실 함수.
    """

    def __init__(
        self, base_loss_fn: nn.Module, temp: float = 4.0, alpha: float = 0.5
    ) -> None:
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.temp: float = temp
        self.alpha: float = alpha
        # KL Div는 Log-Softmax와 Softmax 간의 거리 측정
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        preds_student: list[torch.Tensor],
        preds_teacher: list[torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # 1. 정답(Ground Truth) 기반의 학생 모델 기본 Loss 계산
        loss_base, loss_items = self.base_loss_fn(preds_student, batch)

        # 2. 선생-학생 간 분류 로짓(Classification Logits) KD Loss 계산
        kd_loss: torch.Tensor = torch.tensor(0.0, device=preds_student[0].device)

        for p_s, p_t in zip(preds_student, preds_teacher):
            # 채널 기준 Box와 Class 분리 (학생: 4채널 회귀, 선생: 64채널 DFL)
            # 분류 클래스(80개)는 마지막 차원에 위치한다고 가정
            student_cls: torch.Tensor = p_s[:, -80:, :, :]
            teacher_cls: torch.Tensor = p_t[:, -80:, :, :]

            # 형태 변환: [B, 80, H, W] -> [B, 80, H*W] -> [B*H*W, 80]
            s_logits: torch.Tensor = (
                student_cls.flatten(2).transpose(1, 2).reshape(-1, 80)
            )
            t_logits: torch.Tensor = (
                teacher_cls.flatten(2).transpose(1, 2).reshape(-1, 80)
            )

            # Temperature 스케일링 적용 (소프트 라벨 생성)
            s_log_probs: torch.Tensor = F.log_softmax(
                s_logits.float() / self.temp, dim=-1
            )
            t_probs: torch.Tensor = F.softmax(t_logits.float() / self.temp, dim=-1)

            # KL Divergence 계산 (온도 스케일의 보정을 위해 temp 제곱 곱셈)
            kd_loss += self.kl_div(s_log_probs, t_probs) * (self.temp**2)

        # 3. 최종 Loss 결합
        total_loss: torch.Tensor = (1.0 - self.alpha) * loss_base + self.alpha * kd_loss

        return total_loss, loss_items


def compute_ciou(
    boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """
    [Zheng et al., 2020] 논문에 기반한 CIoU(Complete IoU) 계산 함수.
    boxes1, boxes2 format: [..., 4] (x1, y1, x2, y2)
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unbind(dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unbind(dim=-1)

    # 1. Intersection 영역
    inter_x1: torch.Tensor = torch.max(b1_x1, b2_x1)
    inter_y1: torch.Tensor = torch.max(b1_y1, b2_y1)
    inter_x2: torch.Tensor = torch.min(b1_x2, b2_x2)
    inter_y2: torch.Tensor = torch.min(b1_y2, b2_y2)

    inter_area: torch.Tensor = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )

    # 2. Union 영역
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union_area: torch.Tensor = w1 * h1 + w2 * h2 - inter_area + eps
    iou: torch.Tensor = inter_area / union_area

    # 3. Center Distance (중심점 거리)
    c_x1: torch.Tensor = torch.min(b1_x1, b2_x1)
    c_y1: torch.Tensor = torch.min(b1_y1, b2_y1)
    c_x2: torch.Tensor = torch.max(b1_x2, b2_x2)
    c_y2: torch.Tensor = torch.max(b1_y2, b2_y2)

    diagonal_sq: torch.Tensor = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps
    center_dist_sq: torch.Tensor = ((b1_x1 + b1_x2) - (b2_x1 + b2_x2)) ** 2 / 4 + (
        (b1_y1 + b1_y2) - (b2_y1 + b2_y2)
    ) ** 2 / 4

    # 4. Aspect Ratio (종횡비 페널티)
    v: torch.Tensor = (4 / (math.pi**2)) * torch.pow(
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
    )
    with torch.no_grad():
        alpha: torch.Tensor = v / (v - iou + 1 + eps)

    ciou: torch.Tensor = iou - (center_dist_sq / diagonal_sq + v * alpha)
    return ciou


class AcademicTaskAlignedAssigner(nn.Module):
    """
    [Feng et al., ICCV 2021] TOOD 논문의 Task-aligned Assigner 수식을
    바닥부터 재구현한 클린룸 코드.
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        self.topk: int = topk
        self.num_classes: int = num_classes
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = eps

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        bs, max_gts, _ = gt_bboxes.shape
        num_anchors: int = anc_points.shape[0]
        device: torch.device = gt_bboxes.device

        if max_gts == 0:
            return (
                torch.zeros((bs, num_anchors), dtype=torch.long, device=device),
                torch.zeros((bs, num_anchors, 4), device=device),
                torch.zeros((bs, num_anchors, self.num_classes), device=device),
                torch.zeros((bs, num_anchors), dtype=torch.bool, device=device),
                torch.zeros((bs, num_anchors), dtype=torch.long, device=device),
            )

        # 1. GT 박스 내부에 있는 앵커만 필터링 (Spatial Prior)
        anc_expanded: torch.Tensor = anc_points.unsqueeze(0).unsqueeze(
            0
        )  # [1, 1, A, 2]
        gt_lt: torch.Tensor = gt_bboxes[..., :2].unsqueeze(2)  # [B, G, 1, 2]
        gt_rb: torch.Tensor = gt_bboxes[..., 2:].unsqueeze(2)  # [B, G, 1, 2]

        # 앵커가 GT의 (left, top) 보다 크고 (right, bottom) 보다 작아야 함
        is_in_gts: torch.Tensor = ((anc_expanded > gt_lt) & (anc_expanded < gt_rb)).all(
            dim=-1
        )  # [B, G, A]
        is_in_gts = is_in_gts & mask_gt.bool().expand(-1, -1, num_anchors)

        # 2. TOOD 정렬 지표(Alignment Metric) 계산: t = s^alpha * u^beta
        # 예측 박스와 GT 박스 간의 IoU 계산
        pd_boxes_exp: torch.Tensor = pd_bboxes.unsqueeze(1).expand(
            -1, max_gts, -1, -1
        )  # [B, G, A, 4]
        gt_boxes_exp: torch.Tensor = gt_bboxes.unsqueeze(2).expand(
            -1, -1, num_anchors, -1
        )  # [B, G, A, 4]

        ious: torch.Tensor = compute_ciou(pd_boxes_exp, gt_boxes_exp).clamp(
            min=0
        )  # [B, G, A]

        # 예측 클래스 스코어 추출
        batch_idx: torch.Tensor = torch.arange(bs, device=device).view(-1, 1, 1)
        gt_labels_idx: torch.Tensor = gt_labels.long()

        # 벡터화
        ind0 = torch.arange(bs, device=device).view(-1, 1).expand(-1, max_gts)  # [B, G]
        ind1 = gt_labels.squeeze(-1).long()  # [B, G]
        # pd_scores: [B, A, C] -> 전치해서 [B, C, A]로 만든 뒤 인덱싱
        scores_for_gt = pd_scores.permute(0, 2, 1)[ind0, ind1, :]  # [B, G, A]

        # scores_for_gt: torch.Tensor = pd_scores[
        #     batch_idx, torch.arange(num_anchors), gt_labels_idx
        # ]  # [B, G, A]

        alignment_metric: torch.Tensor = (scores_for_gt**self.alpha) * (ious**self.beta)
        alignment_metric.masked_fill_(~is_in_gts, 0.0)

        # 3. Top-K 앵커 선택
        topk_mask: torch.Tensor = torch.zeros_like(alignment_metric, dtype=torch.bool)
        topk_vals, topk_idxs = torch.topk(
            alignment_metric, self.topk, dim=-1, largest=True
        )
        topk_mask.scatter_(-1, topk_idxs, topk_vals > self.eps)

        final_matching_mask: torch.Tensor = is_in_gts & topk_mask

        # 4. 다중 할당 충돌 해결 (하나의 앵커가 여러 GT에 할당된 경우 IoU가 가장 높은 GT 선택)
        matched_gt_counts: torch.Tensor = final_matching_mask.sum(dim=1)  # [B, A]
        if (matched_gt_counts > 1).any():
            max_iou_idx: torch.Tensor = ious.argmax(dim=1)  # [B, A]
            is_max_iou: torch.Tensor = torch.zeros_like(final_matching_mask)
            is_max_iou.scatter_(1, max_iou_idx.unsqueeze(1), True)
            final_matching_mask = final_matching_mask & (
                is_max_iou | (matched_gt_counts.unsqueeze(1) <= 1)
            )

        # 5. 최종 Target 생성
        fg_mask: torch.Tensor = final_matching_mask.any(dim=1)  # [B, A]
        target_gt_idx: torch.Tensor = final_matching_mask.float().argmax(
            dim=1
        )  # [B, A]

        # 타겟 인덱싱을 위한 1D 변환
        b_idx: torch.Tensor = (
            torch.arange(bs, device=device).unsqueeze(1).expand(-1, num_anchors)
        )

        target_labels: torch.Tensor = gt_labels[b_idx, target_gt_idx, 0].long()
        target_bboxes: torch.Tensor = gt_bboxes[b_idx, target_gt_idx, :]

        # TOOD 논문의 스코어 정규화
        align_masked: torch.Tensor = alignment_metric * final_matching_mask
        max_align_per_gt: torch.Tensor = align_masked.amax(dim=-1, keepdim=True)
        max_iou_per_gt: torch.Tensor = (ious * final_matching_mask).amax(
            dim=-1, keepdim=True
        )

        norm_align_metric: torch.Tensor = (
            align_masked * max_iou_per_gt / (max_align_per_gt + self.eps)
        ).amax(dim=1)

        target_scores: torch.Tensor = torch.zeros(
            (bs, num_anchors, self.num_classes), device=device
        )
        target_scores[b_idx, torch.arange(num_anchors), target_labels] = (
            norm_align_metric
        )
        target_scores.masked_fill_(~fg_mask.unsqueeze(-1), 0.0)
        target_labels.masked_fill_(~fg_mask, self.num_classes)

        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx


def bbox_ciou(box1, box2, eps=1e-7):
    """
    box1, box2: (..., 4) tensor [x, y, w, h] (center format)
    return: CIoU value in [0,1]
    """
    b1_x, b1_y, b1_w, b1_h = box1.unbind(-1)
    b2_x, b2_y, b2_w, b2_h = box2.unbind(-1)

    # xyxy 좌표로 변환
    b1_x1, b1_y1 = b1_x - b1_w / 2, b1_y - b1_h / 2
    b1_x2, b1_y2 = b1_x + b1_w / 2, b1_y + b1_h / 2
    b2_x1, b2_y1 = b2_x - b2_w / 2, b2_y - b2_h / 2
    b2_x2, b2_y2 = b2_x + b2_w / 2, b2_y + b2_h / 2

    # 교집합
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # 합집합
    area1 = b1_w * b1_h
    area2 = b2_w * b2_h
    union = area1 + area2 - inter_area + eps
    iou = inter_area / union

    # 중심 거리 및 외접박스 대각선
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw**2 + ch**2 + eps
    rho2 = (b2_x - b1_x) ** 2 + (b2_y - b1_y) ** 2

    # aspect ratio consistency
    v = (4 / (3.141592653589793**2)) * torch.pow(
        torch.atan(b2_w / b2_h) - torch.atan(b1_w / b1_h), 2
    )
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    ciou = iou - (rho2 / c2 + v * alpha)
    return ciou.clamp(0, 1)


# --------------------------
#  DetectionLoss 클래스
# --------------------------
class CustomDetectionLoss(nn.Module):
    """
    YOLO11_MobileNetV4 전용 Detection Loss
    - Box: CIoU Loss
    - Objectness: BCE
    - Class: BCE
    """

    def __init__(self, num_classes=80, lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, preds, targets):
        """
        preds: tuple(p3, p4, p5)
          - 각 p: [B, C, H, W], C = num_classes + 5
        targets: dict{"bboxes": [M,4], "cls":[M,1], "batch_idx":[M]}
        """
        device = preds[0].device
        B = preds[0].shape[0]

        # 1️⃣ multi-scale 예측 합치기
        out_list = []
        for p in preds:
            b, c, h, w = p.shape
            p = p.view(b, c, -1).permute(0, 2, 1)  # [B, HW, C]
            out_list.append(p)
        pred = torch.cat(out_list, dim=1)  # [B, N, C]

        pred_box = pred[..., 0:4]  # xywh
        pred_obj = pred[..., 4:5]
        pred_cls = pred[..., 5:]

        # 2️⃣ GT가 없으면 (empty batch)
        if targets["bboxes"].numel() == 0:
            loss_box = torch.tensor(0.0, device=device)
            loss_obj = F.binary_cross_entropy_with_logits(
                pred_obj, torch.zeros_like(pred_obj)
            )
            loss_cls = torch.tensor(0.0, device=device)
            total_loss = self.lambda_obj * loss_obj
            return total_loss, (loss_box, loss_obj, loss_cls)

        # 3️⃣ 임시 anchor-free 방식: GT 중심에 대해 가장 가까운 grid 선택 (간단 매칭)
        # 이 부분은 학습 안정화용 간단 matching이며, 실제 YOLO assigner와는 다름
        loss_box, loss_obj, loss_cls = 0.0, 0.0, 0.0
        for b in range(B):
            # 이미지별 GT 필터
            gt_mask = targets["batch_idx"] == b
            if gt_mask.sum() == 0:
                continue
            gt_boxes = targets["bboxes"][gt_mask]  # [G,4] (xywh norm)
            gt_cls = targets["cls"][gt_mask].long().squeeze(1)  # [G]

            pred_b = pred_box[b]  # [N,4]
            pred_o = pred_obj[b]  # [N,1]
            pred_c = pred_cls[b]  # [N,num_classes]

            # 간단하게 상위 obj confidence 몇 개만 GT에 매칭 (lightweight)
            topk = min(50, pred_o.numel())
            _, idx = pred_o.view(-1).topk(topk)
            selected_pred_box = pred_b[idx]
            selected_pred_obj = pred_o[idx]
            selected_pred_cls = pred_c[idx]

            # GT 반복 확장
            g = gt_boxes.shape[0]
            pred_expand = selected_pred_box.unsqueeze(1).repeat(1, g, 1)
            gt_expand = gt_boxes.unsqueeze(0).repeat(topk, 1, 1)

            ciou = bbox_ciou(pred_expand, gt_expand)
            ciou_max, ciou_idx = ciou.max(1)
            best_gt = gt_boxes[ciou_idx]

            # Box Loss = (1 - CIoU)
            box_loss = (1.0 - ciou_max).mean()

            # Objectness target: IoU 값 기반
            obj_target = ciou_max.detach().unsqueeze(1)
            obj_loss = F.binary_cross_entropy_with_logits(selected_pred_obj, obj_target)

            # Class Loss
            cls_target = torch.zeros_like(selected_pred_cls)
            cls_target[torch.arange(topk), gt_cls[ciou_idx]] = 1.0
            cls_loss = self.bce(selected_pred_cls, cls_target)

            loss_box += box_loss
            loss_obj += obj_loss
            loss_cls += cls_loss

        # 4️⃣ 총합
        loss_box /= B
        loss_obj /= B
        loss_cls /= B

        total_loss = (
            self.lambda_box * loss_box
            + self.lambda_obj * loss_obj
            + self.lambda_cls * loss_cls
        )

        return total_loss, (loss_box.detach(), loss_obj.detach(), loss_cls.detach())
