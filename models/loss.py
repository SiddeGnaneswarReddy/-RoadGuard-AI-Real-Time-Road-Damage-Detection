"""
Multi-Task Loss for GuardRailNet Detection.
Includes:
- CIoU Loss / Smooth-L1 for precise bounding box regression
- Focal Loss for foreground/background objectness handling class imbalance
- Cross-Entropy with Label Smoothing for barrier condition classification
"""

import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes Complete IoU (CIoU) between boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    inter_x1 = torch.maximum(b1_x1, b2_x1)
    inter_y1 = torch.maximum(b1_y1, b2_y1)
    inter_x2 = torch.minimum(b1_x2, b2_x2)
    inter_y2 = torch.minimum(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    b1_area = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    b2_area = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
    union_area = b1_area + b2_area - inter_area + eps
    iou = inter_area / union_area

    # Enclosing smallest bounding box
    c_x1 = torch.minimum(b1_x1, b2_x1)
    c_y1 = torch.minimum(b1_y1, b2_y1)
    c_x2 = torch.maximum(b1_x2, b2_x2)
    c_y2 = torch.maximum(b1_y2, b2_y2)
    c_diag_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    # Center distance
    b1_cx = (b1_x1 + b1_x2) / 2
    b1_cy = (b1_y1 + b1_y2) / 2
    b2_cx = (b2_x1 + b2_x2) / 2
    b2_cy = (b2_y1 + b2_y2) / 2
    center_dist_sq = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    # Aspect ratio consistency
    b1_w = (b1_x2 - b1_x1).clamp(min=eps)
    b1_h = (b1_y2 - b1_y1).clamp(min=eps)
    b2_w = (b2_x2 - b2_x1).clamp(min=eps)
    b2_h = (b2_y2 - b2_y1).clamp(min=eps)

    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(b2_w / b2_h) - torch.atan(b1_w / b1_h), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (center_dist_sq / c_diag_sq) - alpha * v
    return ciou.clamp(min=-1.0, max=1.0)


class FocalLoss(nn.Module):
    """Focal Loss for addressing severe foreground-background imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GuardRailLoss(nn.Module):
    """
    Combined multi-task loss for Road Guardrail Detection.
    Balancing box coordinate regression, presence objectness, and barrier condition classification.
    """

    def __init__(
        self,
        num_classes: int = 5,
        box_weight: float = 5.0,
        obj_weight: float = 2.0,
        cls_weight: float = 1.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight

        self.obj_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        self.cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        img_size: Tuple[int, int] = (512, 512)
    ) -> Dict[str, torch.Tensor]:
        """
        Computes total loss across all pyramid levels.
        targets: list of dicts with:
            - 'boxes': Tensor [N, 4] in [x1, y1, x2, y2] absolute coordinates
            - 'classes': Tensor [N] class indices
        """
        device = predictions[0]["box"].device
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)

        batch_size = len(targets)
        h_img, w_img = img_size

        for level_out in predictions:
            box_pred = level_out["box"]  # [B, 4, H, W]
            obj_pred = level_out["obj"]  # [B, 1, H, W]
            cls_pred = level_out["cls"]  # [B, C, H, W]
            stride = level_out["stride"]

            B, _, H, W = box_pred.shape

            target_obj = torch.zeros((B, 1, H, W), device=device)
            assigned_boxes = []
            pred_boxes = []
            assigned_cls_preds = []
            assigned_classes = []

            for b in range(batch_size):
                gt_boxes = targets[b]["boxes"].to(device)
                gt_classes = targets[b]["classes"].to(device)

                if gt_boxes.numel() == 0:
                    continue

                for k in range(len(gt_boxes)):
                    gx1, gy1, gx2, gy2 = gt_boxes[k]
                    g_cls = gt_classes[k]

                    # Grid cell coordinates for center
                    gcx = (gx1 + gx2) / 2.0
                    gcy = (gy1 + gy2) / 2.0

                    cell_x = int((gcx / stride).clamp(0, W - 1).item())
                    cell_y = int((gcy / stride).clamp(0, H - 1).item())

                    target_obj[b, 0, cell_y, cell_x] = 1.0

                    # Predicted box at cell
                    dx, dy, dw, dh = box_pred[b, :, cell_y, cell_x]
                    pcx = (cell_x + torch.sigmoid(dx)) * stride
                    pcy = (cell_y + torch.sigmoid(dy)) * stride
                    pw = torch.exp(torch.clamp(dw, max=4.0)) * stride * 2.5
                    ph = torch.exp(torch.clamp(dh, max=4.0)) * stride * 2.5

                    px1 = pcx - pw / 2
                    py1 = pcy - ph / 2
                    px2 = pcx + pw / 2
                    py2 = pcy + ph / 2

                    pred_boxes.append(torch.stack([px1, py1, px2, py2]))
                    assigned_boxes.append(gt_boxes[k])
                    assigned_cls_preds.append(cls_pred[b, :, cell_y, cell_x])
                    assigned_classes.append(g_cls)

            # Objectness loss across full feature map
            level_obj_loss = self.obj_loss_fn(obj_pred, target_obj)
            total_obj_loss += level_obj_loss

            # Regression and classification loss for positive cells
            if len(pred_boxes) > 0:
                p_boxes = torch.stack(pred_boxes)
                t_boxes = torch.stack(assigned_boxes)
                ciou = bbox_ciou(p_boxes, t_boxes)
                level_box_loss = (1.0 - ciou).mean()
                total_box_loss += level_box_loss

                p_cls = torch.stack(assigned_cls_preds)
                t_cls = torch.stack(assigned_classes)
                level_cls_loss = self.cls_loss_fn(p_cls, t_cls)
                total_cls_loss += level_cls_loss

        # Weighted combination
        loss = (
            self.box_weight * total_box_loss +
            self.obj_weight * total_obj_loss +
            self.cls_weight * total_cls_loss
        )

        return {
            "loss": loss,
            "box_loss": total_box_loss,
            "obj_loss": total_obj_loss,
            "cls_loss": total_cls_loss
        }
