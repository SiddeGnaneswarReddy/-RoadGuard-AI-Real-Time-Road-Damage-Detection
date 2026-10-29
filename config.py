"""
Configuration module for GuardRailNet.
Provides default settings and loaders supporting JSON and YAML.
"""

import os
import json

DEFAULT_CONFIG = {
    "project": {
        "name": "GuardRailNet",
        "version": "1.0.0",
        "description": "End-to-End Deep Learning System for Road Guardrail Detection, Classification & Safety Risk Telemetry",
        "author": "Deep Learning Engineering Team"
    },
    "classes": {
        "names": {
            0: "metal_w_beam",
            1: "concrete_jersey_barrier",
            2: "cable_barrier",
            3: "damaged_guardrail",
            4: "hazard_post"
        },
        "num_classes": 5,
        "colors": {
            0: [0, 210, 255],
            1: [255, 170, 0],
            2: [170, 0, 255],
            3: [255, 50, 50],
            4: [50, 255, 120]
        }
    },
    "model": {
        "architecture": "GuardNet",
        "input_size": [512, 512],
        "backbone": "guard_resnet18",
        "anchor_grid_sizes": [16, 32, 64],
        "pretrained": False,
        "weights_path": "checkpoints/guardnet_best.pt"
    },
    "training": {
        "batch_size": 8,
        "epochs": 15,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "lr_scheduler": "cosine",
        "device": "auto",
        "save_dir": "checkpoints",
        "val_interval": 1,
        "early_stopping_patience": 10
    },
    "loss_weights": {
        "box_loss": 5.0,
        "cls_loss": 1.0,
        "obj_loss": 2.0
    },
    "inference": {
        "conf_threshold": 0.40,
        "iou_threshold": 0.45,
        "max_detections": 50,
        "safety_distance_warning_px": 120
    }
}


def load_config(config_path: str = "config.json") -> dict:
    """Loads configuration from JSON or YAML file, falling back to defaults."""
    cfg = dict(DEFAULT_CONFIG)

    if config_path and os.path.exists(config_path):
        if config_path.endswith(".json"):
            with open(config_path, "r") as f:
                loaded = json.load(f)
                cfg.update(loaded)
        elif config_path.endswith((".yaml", ".yml")):
            try:
                import yaml
                with open(config_path, "r") as f:
                    loaded = yaml.safe_load(f)
                    if loaded:
                        cfg.update(loaded)
            except ImportError:
                # If yaml is not installed, parse json or use defaults
                json_path = config_path.rsplit(".", 1)[0] + ".json"
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        cfg.update(json.load(f))

    # Ensure class names keys are integers
    if "classes" in cfg and "names" in cfg["classes"]:
        cfg["classes"]["names"] = {int(k): v for k, v in cfg["classes"]["names"].items()}

    return cfg


if __name__ == "__main__":
    # Also save as config.json for zero-dependency portability
    with open("config.json", "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print("Saved default config.json successfully.")
