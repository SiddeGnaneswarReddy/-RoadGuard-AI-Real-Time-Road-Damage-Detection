"""
Visualization and Safety Telemetry HUD for GuardRailNet.
Draws high-contrast bounding boxes, barrier condition labels, safety distance zones,
and dashboard metrics onto images and video frames.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_NAMES = {
    0: "metal_w_beam",
    1: "concrete_jersey_barrier",
    2: "cable_barrier",
    3: "damaged_guardrail",
    4: "hazard_post"
}

CLASS_COLORS = {
    0: (255, 210, 0),    # Cyan/Sky (BGR)
    1: (0, 170, 255),    # Amber
    2: (255, 0, 170),    # Magenta/Purple
    3: (50, 50, 255),    # Red (Hazard)
    4: (50, 255, 120)    # Green
}


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, any]],
    conf_threshold: float = 0.35,
    show_hud: bool = True,
    show_risk_zones: bool = True
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Renders detection bounding boxes, condition tags, and safety telemetry HUD onto an image.
    Returns annotated image and telemetry summary.
    """
    canvas = image.copy()
    h, w = canvas.shape[:2]

    # Filter detections
    active_dets = [d for d in detections if d.get("confidence", 0.0) >= conf_threshold]

    stats = {
        "total_barriers": len(active_dets),
        "damaged_count": sum(1 for d in active_dets if d.get("class_id") == 3),
        "types": {},
        "collision_risk": "LOW"
    }

    # Count barrier types
    for d in active_dets:
        cid = d.get("class_id", 0)
        cname = CLASS_NAMES.get(cid, f"class_{cid}")
        stats["types"][cname] = stats["types"].get(cname, 0) + 1

    # Collision / Structural risk level
    if stats["damaged_count"] > 0:
        stats["collision_risk"] = "CRITICAL: BARRIER BREACH / DEFORMED"
    elif stats["total_barriers"] < 1:
        stats["collision_risk"] = "CAUTION: UNGUARDED ROAD EDGE"
    else:
        stats["collision_risk"] = "NORMAL: CONTINUOUS SAFETY BARRIER"

    # Draw Risk Zone Corridor if requested
    if show_risk_zones:
        overlay = canvas.copy()
        # Draw road boundary hazard guide lines
        left_bound = int(w * 0.18)
        right_bound = int(w * 0.82)
        cv2.line(overlay, (left_bound, int(h * 0.55)), (0, h), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(overlay, (right_bound, int(h * 0.55)), (w, h), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

    # Draw Detections
    for det in active_dets:
        bbox = det["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        conf = det.get("confidence", 0.0)
        cls_id = det.get("class_id", 0)

        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        label_text = f"{CLASS_NAMES.get(cls_id, 'Guardrail')} {conf:.1%}"

        # Draw box with rounded corner effect or thick border
        thickness = 3 if cls_id == 3 else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # Label background
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tag_y1 = max(0, y1 - text_h - 10)
        tag_y2 = y1
        cv2.rectangle(canvas, (x1, tag_y1), (x1 + text_w + 12, tag_y2), color, -1)

        # Text
        text_color = (0, 0, 0) if cls_id != 3 else (255, 255, 255)
        cv2.putText(
            canvas,
            label_text,
            (x1 + 6, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            1,
            cv2.LINE_AA
        )

        # Highlight damage with danger indicator
        if cls_id == 3:
            cv2.putText(
                canvas,
                "⚠️ CRITICAL DAMAGE",
                (x1, min(h - 10, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

    # Draw HUD Header if requested
    if show_hud:
        hud_h = 75
        hud_overlay = canvas.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (15, 23, 42), -1)
        cv2.addWeighted(hud_overlay, 0.85, canvas, 0.15, 0, canvas)

        # HUD Text
        title = "GUARDRAILNET AI ROAD SAFETY AUDITOR"
        cv2.putText(canvas, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

        risk_color = (0, 255, 0) if "NORMAL" in stats["collision_risk"] else (0, 0, 255) if "CRITICAL" in stats["collision_risk"] else (0, 200, 255)
        cv2.putText(
            canvas,
            f"STATUS: {stats['collision_risk']}",
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            risk_color,
            2,
            cv2.LINE_AA
        )

        hud_info = f"BARRIERS: {stats['total_barriers']} | DAMAGE DETECTIONS: {stats['damaged_count']}"
        cv2.putText(canvas, hud_info, (w - 480, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas, stats


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> plt.Figure:
    """Generates a clean confusion matrix heatmap figure for Streamlit / reporting."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    labels = class_names + ["background"]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", color="#e2e8f0")
    ax.set_yticklabels(labels, color="#e2e8f0")

    ax.set_title("Guardrail Condition Confusion Matrix", color="#38bdf8", fontsize=14, pad=15)
    ax.set_ylabel("True Condition", color="#e2e8f0", fontsize=12)
    ax.set_xlabel("Predicted Condition", color="#e2e8f0", fontsize=12)

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "#94a3b8",
                fontsize=11, fontweight="bold"
            )

    fig.tight_layout()
    return fig


def plot_training_history(history: Dict[str, List[float]]) -> plt.Figure:
    """Generates loss and mAP curves over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor('#0f172a')

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss plot
    ax1.set_facecolor('#1e293b')
    ax1.plot(epochs, history.get("train_loss", []), label="Train Loss", color="#38bdf8", linewidth=2.5)
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#f43f5e", linewidth=2.5, linestyle="--")
    ax1.set_title("Training & Validation Loss", color="#e2e8f0", fontsize=12)
    ax1.set_xlabel("Epoch", color="#94a3b8")
    ax1.set_ylabel("Loss", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")
    ax1.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0')
    ax1.grid(True, color="#334155", linestyle=":", alpha=0.6)

    # mAP plot
    ax2.set_facecolor('#1e293b')
    if "val_map" in history:
        ax2.plot(epochs, history["val_map"], label="mAP@50", color="#10b981", linewidth=2.5)
    ax2.set_title("Validation Detection mAP@50", color="#e2e8f0", fontsize=12)
    ax2.set_xlabel("Epoch", color="#94a3b8")
    ax2.set_ylabel("mAP@50", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    ax2.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0')
    ax2.grid(True, color="#334155", linestyle=":", alpha=0.6)

    fig.tight_layout()
    return fig
