"""
Evaluation and Benchmarking Script for GuardRailNet.
Calculates mAP, per-class Precision, Recall, F1 scores and outputs confusion matrix.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config
from models.guardnet import GuardNet
from data.dataset import build_dataloaders
from utils.metrics import evaluate_detections
from utils.visualizer import plot_confusion_matrix


def run_evaluation(
    weights_path: str = "checkpoints/guardnet_best.pt",
    config_path: str = "config.json",
    data_dir: str = "data",
    conf_thresh: float = 0.35,
    iou_thresh: float = 0.5
) -> dict:
    """Executes validation suite and produces benchmark metrics."""
    cfg = load_config(config_path)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["classes"]["num_classes"]
    class_names = [cfg["classes"]["names"][i] for i in range(num_classes)]
    img_size = tuple(cfg["model"]["input_size"])

    model = GuardNet(num_classes=num_classes).to(device)

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {weights_path}")
    else:
        print(f"Warning: Checkpoint not found at {weights_path}, running with initialized weights.")

    model.eval()
    _, val_loader = build_dataloaders(data_dir, batch_size=8, img_size=img_size)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            targets = batch["targets"]

            preds = model(images)
            decoded = model.decode(preds, conf_threshold=conf_thresh, iou_threshold=iou_thresh, image_shape=img_size)

            all_preds.extend(decoded)
            all_targets.extend(targets)

    results = evaluate_detections(all_preds, all_targets, num_classes=num_classes, iou_thresh=iou_thresh)

    print("\n" + "=" * 65)
    print("🚦 GUARDRAILNET BENCHMARK EVALUATION RESULTS")
    print("=" * 65)
    print(f"Overall mAP@50: {results['mAP@50']:.4f}\n")
    print(f"{'Class Name':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)
    for c in range(num_classes):
        cname = class_names[c]
        p = results["class_precisions"].get(c, 0.0)
        r = results["class_recalls"].get(c, 0.0)
        f1 = results["class_f1s"].get(c, 0.0)
        print(f"{cname:<28} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")
    print("=" * 65 + "\n")

    # Save report
    os.makedirs("checkpoints", exist_ok=True)
    report_path = "checkpoints/evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "mAP@50": results["mAP@50"],
            "class_metrics": {
                class_names[c]: {
                    "precision": results["class_precisions"].get(c, 0.0),
                    "recall": results["class_recalls"].get(c, 0.0),
                    "f1": results["class_f1s"].get(c, 0.0)
                } for c in range(num_classes)
            },
            "confusion_matrix": results["confusion_matrix"]
        }, f, indent=2)

    # Save confusion matrix plot
    cm_arr = np.array(results["confusion_matrix"])
    fig = plot_confusion_matrix(cm_arr, class_names)
    fig.savefig("checkpoints/confusion_matrix.png", dpi=200, bbox_inches="tight")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="checkpoints/guardnet_best.pt")
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    run_evaluation(weights_path=args.weights, config_path=args.config)
