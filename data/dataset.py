"""
PyTorch Dataset and DataLoader for Road Guardrail Detection.
Handles image loading, bounding box parsing, online data augmentation,
and batch collation.
"""

import os
import glob
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class RoadGuardDataset(Dataset):
    """
    Dataset loader for road guardrail & barrier images with YOLO format annotations.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: Tuple[int, int] = (512, 512),
        augment: bool = True
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")

        self.images_dir = os.path.join(root_dir, split, "images")
        self.labels_dir = os.path.join(root_dir, split, "labels")

        self.image_paths = sorted(
            glob.glob(os.path.join(self.images_dir, "*.jpg")) +
            glob.glob(os.path.join(self.images_dir, "*.png"))
        )

        # ImageNet normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_labels(self, base_name: str, orig_w: int, orig_h: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label_file = os.path.join(self.labels_dir, f"{base_name}.txt")
        if not os.path.exists(label_file):
            return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

        boxes = []
        classes = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx = float(parts[1]) * orig_w
                    cy = float(parts[2]) * orig_h
                    w = float(parts[3]) * orig_w
                    h = float(parts[4]) * orig_h

                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0

                    boxes.append([x1, y1, x2, y2])
                    classes.append(cls_id)

        if len(boxes) == 0:
            return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(classes, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        img_path = self.image_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Read image in RGB
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Failed to load image at {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]

        boxes, classes = self._load_labels(base_name, orig_w, orig_h)

        # Resize image to target input size
        target_w, target_h = self.img_size
        if (orig_w, orig_h) != (target_w, target_h):
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if len(boxes) > 0:
                scale_x = target_w / float(orig_w)
                scale_y = target_h / float(orig_h)
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y

        # Convert to float tensor [0, 1]
        img_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0

        # Data augmentation for training
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])
                if len(boxes) > 0:
                    x1 = boxes[:, 0].clone()
                    x2 = boxes[:, 2].clone()
                    boxes[:, 0] = target_w - x2
                    boxes[:, 2] = target_w - x1

            # Random brightness & contrast
            if torch.rand(1).item() > 0.3:
                alpha = 0.8 + torch.rand(1).item() * 0.4
                beta = (torch.rand(1).item() - 0.5) * 0.2
                img_tensor = (img_tensor * alpha + beta).clamp(0.0, 1.0)

        # Normalize with ImageNet stats
        norm_tensor = (img_tensor - self.mean) / self.std

        return {
            "image": norm_tensor,
            "boxes": boxes,
            "classes": classes,
            "image_path": img_path,
            "orig_size": (orig_h, orig_w)
        }


def collate_fn(batch: List[Dict[str, any]]) -> Dict[str, any]:
    """Custom collator handling variable box counts per image."""
    images = torch.stack([item["image"] for item in batch])
    targets = [
        {"boxes": item["boxes"], "classes": item["classes"]}
        for item in batch
    ]
    paths = [item["image_path"] for item in batch]
    orig_sizes = [item["orig_size"] for item in batch]

    return {
        "images": images,
        "targets": targets,
        "image_paths": paths,
        "orig_sizes": orig_sizes
    }


def build_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    img_size: Tuple[int, int] = (512, 512),
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Constructs train and validation DataLoaders."""
    train_dataset = RoadGuardDataset(data_dir, split="train", img_size=img_size, augment=True)
    val_dataset = RoadGuardDataset(data_dir, split="val", img_size=img_size, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader
