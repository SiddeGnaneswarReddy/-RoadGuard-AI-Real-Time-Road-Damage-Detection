"""
Training pipeline for GuardRailNet.
Runs multi-scale detection training with validation, early stopping, and checkpointing.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config
from models.guardnet import GuardNet
from models.loss import GuardRailLoss
from data.dataset import build_dataloaders
from data.dataset_generator import generate_dataset
from utils.metrics import evaluate_detections


def get_device(requested: str = "auto") -> torch.device:
    """Selects best available acceleration hardware (MPS, CUDA, or CPU)."""
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def train_guardrail_net(
    config_path: str = "config.json",
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    progress_callback: Optional[callable] = None
) -> Dict[str, any]:
    """Runs the complete training and validation cycle."""
    # Load config
    cfg = load_config(config_path)

    epochs = epochs_override or cfg["training"]["epochs"]
    batch_size = batch_size_override or cfg["training"]["batch_size"]
    lr = float(cfg["training"]["learning_rate"])
    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    device = get_device(cfg["training"].get("device", "auto"))
    print(f"🚀 GuardRailNet Training initialized on device: {device}")

    # Verify or generate dataset
    data_dir = "data"
    train_img_dir = os.path.join(data_dir, "train", "images")
    if not os.path.exists(train_img_dir) or len(os.listdir(train_img_dir)) == 0:
        print("📦 Dataset not detected. Generating procedural highway training data...")
        generate_dataset(data_dir, num_train=120, num_val=25)

    img_size = tuple(cfg["model"]["input_size"])
    train_loader, val_loader = build_dataloaders(
        data_dir,
        batch_size=batch_size,
        img_size=img_size
    )

    # Instantiate Model & Loss
    num_classes = cfg["classes"]["num_classes"]
    model = GuardNet(num_classes=num_classes).to(device)
    criterion = GuardRailLoss(
        num_classes=num_classes,
        box_weight=cfg["loss_weights"]["box_loss"],
        obj_weight=cfg["loss_weights"]["obj_loss"],
        cls_weight=cfg["loss_weights"]["cls_loss"]
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_map = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_map": [],
        "epochs": []
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_box_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_cls_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device)
            targets = batch["targets"]

            optimizer.zero_grad()
            predictions = model(images)
            loss_dict = criterion(predictions, targets, img_size=img_size)

            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_box_loss += loss_dict["box_loss"].item()
            epoch_obj_loss += loss_dict["obj_loss"].item()
            epoch_cls_loss += loss_dict["cls_loss"].item()

        scheduler.step()
        num_batches = len(train_loader)
        avg_train_loss = epoch_loss / max(1, num_batches)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["epochs"].append(epoch)

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_ground_truths = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                targets = batch["targets"]

                predictions = model(images)
                loss_dict = criterion(predictions, targets, img_size=img_size)
                val_loss += loss_dict["loss"].item()

                decoded = model.decode(predictions, conf_threshold=0.35, image_shape=img_size)
                val_predictions.extend(decoded)
                val_ground_truths.extend(targets)

        avg_val_loss = val_loss / max(1, len(val_loader))
        history["val_loss"].append(round(avg_val_loss, 4))

        # Calculate metrics
        eval_metrics = evaluate_detections(val_predictions, val_ground_truths, num_classes=num_classes)
        val_map = eval_metrics["mAP@50"]
        history["val_map"].append(val_map)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val mAP@50: {val_map:.4f}"
        )

        if progress_callback is not None:
            progress_callback(epoch, epochs, avg_train_loss, avg_val_loss, val_map)

        # Checkpointing
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_map = val_map
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val_loss,
                "val_map": val_map
            }, os.path.join(save_dir, "guardnet_best.pt"))

        # Always save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": avg_val_loss,
            "val_map": val_map
        }, os.path.join(save_dir, "guardnet_last.pt"))

        # Save history JSON
        with open(os.path.join(save_dir, "train_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print(f"✨ Training complete in {total_time:.1f}s! Best Val mAP: {best_map:.4f}")

    return {
        "best_val_loss": best_val_loss,
        "best_map": best_map,
        "history": history,
        "total_time_seconds": total_time
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GuardRailNet Detection Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    args = parser.parse_args()

    train_guardrail_net(config_path=args.config, epochs_override=args.epochs, batch_size_override=args.batch_size)
