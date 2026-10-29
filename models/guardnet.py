"""
GuardNet: End-to-End Deep Learning Architecture for Road Guardrail Detection
Features:
- Multi-scale FPN / PANet neck for capturing multi-range road assets
- Spatial & Channel Attention for elongated guardrail boundaries
- Multi-scale anchor-free detection heads with class & condition classification
"""

import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import GuardBackbone, ConvBlock


class PANetNeck(nn.Module):
    """
    Path Aggregation Network (PANet) Neck.
    Fuses low-level edge details with high-level semantic context through
    top-down and bottom-up feature pyramids.
    """

    def __init__(self, in_channels: List[int], out_channels: int = 128):
        super().__init__()
        c3, c4, c5 = in_channels
        self.out_channels = out_channels

        # Lateral 1x1 convolutions
        self.lat_p5 = nn.Conv2d(c5, out_channels, kernel_size=1)
        self.lat_p4 = nn.Conv2d(c4, out_channels, kernel_size=1)
        self.lat_p3 = nn.Conv2d(c3, out_channels, kernel_size=1)

        # Top-down fusion smoothers
        self.smooth_p4 = ConvBlock(out_channels, out_channels, kernel_size=3)
        self.smooth_p3 = ConvBlock(out_channels, out_channels, kernel_size=3)

        # Bottom-up path
        self.down_n3 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=2)
        self.smooth_n4 = ConvBlock(out_channels, out_channels, kernel_size=3)

        self.down_n4 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=2)
        self.smooth_n5 = ConvBlock(out_channels, out_channels, kernel_size=3)

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = features

        # Top-Down pathway
        m5 = self.lat_p5(p5)
        m4 = self.lat_p4(p4) + F.interpolate(m5, size=p4.shape[-2:], mode="nearest")
        m4 = self.smooth_p4(m4)

        m3 = self.lat_p3(p3) + F.interpolate(m4, size=p3.shape[-2:], mode="nearest")
        m3 = self.smooth_p3(m3)

        # Bottom-Up pathway
        n3 = m3
        n4 = self.smooth_n4(m4 + self.down_n3(n3))
        n5 = self.smooth_n5(m5 + self.down_n4(n4))

        return [n3, n4, n5]  # Strides 8, 16, 32


class DetectionHead(nn.Module):
    """
    Decoupled Detection Head.
    Separates objectness, bounding box regression, and class condition prediction.
    """

    def __init__(self, in_channels: int, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

        # Shared classification / condition branch
        self.cls_branch = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3),
            ConvBlock(in_channels, in_channels, kernel_size=3),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

        # Shared regression & objectness branch
        self.reg_branch = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3),
            ConvBlock(in_channels, in_channels, kernel_size=3),
        )
        self.box_head = nn.Conv2d(in_channels, 4, kernel_size=1)  # dx, dy, dw, dh
        self.obj_head = nn.Conv2d(in_channels, 1, kernel_size=1)  # objectness score

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_logits = self.cls_branch(x)
        reg_features = self.reg_branch(x)
        box_preds = self.box_head(reg_features)
        obj_logits = self.obj_head(reg_features)
        return box_preds, obj_logits, cls_logits


class GuardNet(nn.Module):
    """
    GuardRailNet Full Detection Architecture.
    Predicts multi-scale road barrier locations, dimensions, and condition classes.
    """

    def __init__(self, num_classes: int = 5, base_channels: int = 32, neck_channels: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = GuardBackbone(in_channels=3, base_channels=base_channels)
        self.neck = PANetNeck(in_channels=self.backbone.out_channels, out_channels=neck_channels)

        # Decoupled heads for strides 8, 16, 32
        self.heads = nn.ModuleList([
            DetectionHead(neck_channels, num_classes) for _ in range(3)
        ])
        self.strides = [8, 16, 32]

    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        features = self.backbone(x)
        pyramid_features = self.neck(features)

        outputs = []
        for i, feat in enumerate(pyramid_features):
            box_preds, obj_logits, cls_logits = self.heads[i](feat)
            outputs.append({
                "box": box_preds,        # [B, 4, H_i, W_i]
                "obj": obj_logits,       # [B, 1, H_i, W_i]
                "cls": cls_logits,       # [B, C, H_i, W_i]
                "stride": self.strides[i]
            })

        return outputs

    @torch.no_grad()
    def decode(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        image_shape: Tuple[int, int] = (512, 512)
    ) -> List[List[Dict[str, any]]]:
        """
        Decodes raw model output into postprocessed bounding boxes with NMS.
        Returns:
            List of detected objects per image in batch:
            [
              {"bbox": [x1, y1, x2, y2], "confidence": float, "class_id": int, "label": str},
              ...
            ]
        """
        batch_size = outputs[0]["box"].shape[0]
        img_h, img_w = image_shape
        batch_detections = [[] for _ in range(batch_size)]

        for b in range(batch_size):
            all_boxes = []
            all_scores = []
            all_classes = []

            for level_out in outputs:
                stride = level_out["stride"]
                box_pred = level_out["box"][b]      # [4, H, W]
                obj_pred = level_out["obj"][b]      # [1, H, W]
                cls_pred = level_out["cls"][b]      # [C, H, W]

                _, H, W = box_pred.shape

                obj_prob = torch.sigmoid(obj_pred[0])  # [H, W]
                cls_prob = torch.softmax(cls_pred, dim=0)  # [C, H, W]
                max_cls_prob, class_indices = torch.max(cls_prob, dim=0)  # [H, W]

                confidence = obj_prob * max_cls_prob  # [H, W]

                # Filter by confidence threshold
                mask = confidence > conf_threshold
                if not mask.any():
                    continue

                y_indices, x_indices = torch.where(mask)

                for y, x in zip(y_indices, x_indices):
                    conf = float(confidence[y, x].item())
                    cls_id = int(class_indices[y, x].item())

                    # Decode box relative to grid cell center
                    dx, dy, dw, dh = box_pred[:, y, x]

                    # Center in image coordinates
                    cx = (x.float() + torch.sigmoid(dx)) * stride
                    cy = (y.float() + torch.sigmoid(dy)) * stride

                    # Width and height in image coordinates
                    w = torch.exp(torch.clamp(dw, max=4.0)) * stride * 2.5
                    h = torch.exp(torch.clamp(dh, max=4.0)) * stride * 2.5

                    x1 = max(0.0, float(cx - w / 2))
                    y1 = max(0.0, float(cy - h / 2))
                    x2 = min(float(img_w), float(cx + w / 2))
                    y2 = min(float(img_h), float(cy + h / 2))

                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(conf)
                    all_classes.append(cls_id)

            if len(all_boxes) > 0:
                boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
                scores_t = torch.tensor(all_scores, dtype=torch.float32)
                classes_t = torch.tensor(all_classes, dtype=torch.long)

                keep = non_max_suppression_boxes(boxes_t, scores_t, iou_threshold)
                for idx in keep:
                    batch_detections[b].append({
                        "bbox": [round(v, 2) for v in boxes_t[idx].tolist()],
                        "confidence": round(float(scores_t[idx].item()), 4),
                        "class_id": int(classes_t[idx].item())
                    })

        return batch_detections


def non_max_suppression_boxes(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45) -> List[int]:
    """Pure PyTorch vectorized Non-Maximum Suppression (NMS)."""
    if boxes.numel() == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-7)

        remaining = torch.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep
