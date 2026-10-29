"""
Evaluation metrics for GuardRailNet:
- IoU (Intersection over Union)
- Precision, Recall, F1-Score
- Mean Average Precision (mAP@50 and mAP@50:95)
- Confusion Matrix
"""

from typing import Dict, List, Tuple
import numpy as np
import torch


def box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU between two sets of boxes [N, 4] and [M, 4].
    Format: [x1, y1, x2, y2].
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, a_min=0, a_max=None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, a_min=1e-7, a_max=None)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Computes Average Precision (AP) using standard 11-point or all-point interpolation.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute maximum precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def evaluate_detections(
    predictions: List[List[Dict[str, any]]],
    ground_truths: List[Dict[str, any]],
    num_classes: int = 5,
    iou_thresh: float = 0.5
) -> Dict[str, any]:
    """
    Computes mAP, class-wise AP, Precision, Recall, and Confusion Matrix.
    """
    aps = []
    class_precisions = {}
    class_recalls = {}
    class_f1s = {}

    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
    # Row: Ground Truth (0..num_classes-1, last is background/false positive)
    # Col: Prediction (0..num_classes-1, last is background/missed)

    for c in range(num_classes):
        true_positives = []
        scores = []
        n_ground_truths = 0

        for b in range(len(ground_truths)):
            gt_boxes = ground_truths[b].get("boxes", [])
            gt_classes = ground_truths[b].get("classes", [])

            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
            if isinstance(gt_classes, torch.Tensor):
                gt_classes = gt_classes.cpu().numpy()

            gt_mask = np.array(gt_classes) == c
            c_gt_boxes = np.array(gt_boxes)[gt_mask] if len(gt_boxes) > 0 else np.empty((0, 4))
            n_ground_truths += len(c_gt_boxes)

            # Predictions for this class
            img_preds = [p for p in predictions[b] if p["class_id"] == c]
            if len(img_preds) == 0:
                continue

            # Sort predictions by score descending
            img_preds = sorted(img_preds, key=lambda x: x["confidence"], reverse=True)
            pred_boxes = np.array([p["bbox"] for p in img_preds])
            pred_scores = np.array([p["confidence"] for p in img_preds])

            matched_gt = set()
            ious = box_iou_numpy(pred_boxes, c_gt_boxes)

            for p_idx in range(len(img_preds)):
                scores.append(pred_scores[p_idx])
                if len(c_gt_boxes) == 0:
                    true_positives.append(0)
                    confusion_matrix[num_classes, c] += 1  # False positive
                    continue

                best_gt_idx = np.argmax(ious[p_idx])
                best_iou = ious[p_idx, best_gt_idx]

                if best_iou >= iou_thresh and best_gt_idx not in matched_gt:
                    true_positives.append(1)
                    matched_gt.add(best_gt_idx)
                    confusion_matrix[c, c] += 1
                else:
                    true_positives.append(0)
                    confusion_matrix[num_classes, c] += 1

            # Unmatched ground truths are misses
            unmatched_count = len(c_gt_boxes) - len(matched_gt)
            confusion_matrix[c, num_classes] += unmatched_count

        if n_ground_truths == 0:
            continue

        if len(scores) == 0:
            aps.append(0.0)
            class_precisions[c] = 0.0
            class_recalls[c] = 0.0
            class_f1s[c] = 0.0
            continue

        scores = np.array(scores)
        true_positives = np.array(true_positives)

        sort_idx = np.argsort(-scores)
        tp = true_positives[sort_idx]
        fp = 1 - tp

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recalls = cum_tp / n_ground_truths
        precisions = cum_tp / (cum_tp + cum_fp + 1e-7)

        ap = compute_ap(recalls, precisions)
        aps.append(ap)

        p_final = float(precisions[-1]) if len(precisions) > 0 else 0.0
        r_final = float(recalls[-1]) if len(recalls) > 0 else 0.0
        f1_final = (2 * p_final * r_final) / (p_final + r_final + 1e-7)

        class_precisions[c] = round(p_final, 4)
        class_recalls[c] = round(r_final, 4)
        class_f1s[c] = round(f1_final, 4)

    mean_ap = float(np.mean(aps)) if len(aps) > 0 else 0.0

    return {
        "mAP@50": round(mean_ap, 4),
        "class_precisions": class_precisions,
        "class_recalls": class_recalls,
        "class_f1s": class_f1s,
        "confusion_matrix": confusion_matrix.tolist()
    }
