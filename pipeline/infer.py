"""
Inference engine for GuardRailNet.
Supports single image prediction, directory batching, and dashcam video processing.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config
from models.guardnet import GuardNet
from utils.visualizer import draw_detections, CLASS_NAMES


class GuardRailDetector:
    """End-to-end inference engine for road guardrail and crash barrier detection."""

    def __init__(
        self,
        weights_path: Optional[str] = "checkpoints/guardnet_best.pt",
        config_path: str = "config.json",
        device: Optional[str] = None
    ):
        self.cfg = load_config(config_path)

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.num_classes = self.cfg["classes"]["num_classes"]
        self.img_size = tuple(self.cfg["model"]["input_size"])

        self.model = GuardNet(num_classes=self.num_classes).to(self.device)
        self.model.eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # Load weights if available
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
            self.has_trained_weights = True
        else:
            self.has_trained_weights = False

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocesses raw BGR OpenCV image for model input."""
        orig_h, orig_w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.img_size, interpolation=cv2.INTER_LINEAR)

        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        norm_tensor = (tensor - self.mean) / self.std

        return norm_tensor, (orig_h, orig_w)

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, np.ndarray],
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ) -> Dict[str, any]:
        """Runs detection on an image input (file path or numpy array)."""
        if isinstance(image_input, str):
            bgr = cv2.imread(image_input)
            if bgr is None:
                raise FileNotFoundError(f"Could not load image: {image_input}")
        else:
            bgr = image_input

        orig_h, orig_w = bgr.shape[:2]
        tensor, (h, w) = self.preprocess(bgr)

        t0 = time.time()
        raw_outputs = self.model(tensor)
        decoded = self.model.decode(
            raw_outputs,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            image_shape=self.img_size
        )
        latency_ms = (time.time() - t0) * 1000.0

        # Rescale detections to original dimensions
        scale_x = orig_w / float(self.img_size[0])
        scale_y = orig_h / float(self.img_size[1])

        image_detections = []
        for det in decoded[0]:
            x1, y1, x2, y2 = det["bbox"]
            orig_bbox = [
                round(x1 * scale_x, 1),
                round(y1 * scale_y, 1),
                round(x2 * scale_x, 1),
                round(y2 * scale_y, 1)
            ]
            cid = det["class_id"]
            image_detections.append({
                "bbox": orig_bbox,
                "confidence": det["confidence"],
                "class_id": cid,
                "class_name": CLASS_NAMES.get(cid, f"class_{cid}")
            })

        annotated_img, stats = draw_detections(
            bgr,
            image_detections,
            conf_threshold=conf_threshold
        )

        return {
            "detections": image_detections,
            "annotated_image": annotated_img,
            "stats": stats,
            "latency_ms": round(latency_ms, 2),
            "orig_size": (orig_h, orig_w)
        }

    def process_video(
        self,
        video_path: str,
        output_path: str,
        conf_threshold: float = 0.35,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, any]:
        """Processes a video file and saves annotated video with detection overlay."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        total_damage_frames = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.predict(frame, conf_threshold=conf_threshold)
            out.write(result["annotated_image"])

            if result["stats"]["damaged_count"] > 0:
                total_damage_frames += 1

            frame_idx += 1
            if progress_callback:
                progress_callback(frame_idx, total_frames)

        cap.release()
        out.release()

        elapsed = time.time() - start_time
        return {
            "total_frames": frame_idx,
            "damage_frames": total_damage_frames,
            "fps_processed": round(frame_idx / max(1e-5, elapsed), 2),
            "output_path": output_path
        }
