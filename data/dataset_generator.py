"""
Procedural Synthetic Road & Guardrail Scene Generator for Deep Learning.
Generates diverse highway scenes with realistic perspective, barrier types,
lighting variations, weather conditions, and structural damage labels.
"""

import os
import json
import random
import numpy as np
import cv2
from typing import Dict, List, Tuple


CLASS_NAMES = [
    "metal_w_beam",
    "concrete_jersey_barrier",
    "cable_barrier",
    "damaged_guardrail",
    "hazard_post"
]


class RoadSceneGenerator:
    """Generates procedural road scenes with annotated guardrails and crash barriers."""

    def __init__(self, img_size: Tuple[int, int] = (512, 512)):
        self.w, self.h = img_size

    def _draw_sky_and_horizon(self, canvas: np.ndarray, horizon_y: int, weather: str):
        """Draws sky gradient, mountains/trees, and horizon."""
        if weather == "sunset":
            top_color = (40, 60, 180)     # Orange-purple sunset
            horizon_color = (80, 170, 255) # Golden yellow
        elif weather == "night":
            top_color = (15, 15, 30)      # Deep dark
            horizon_color = (30, 40, 60)
        elif weather == "foggy":
            top_color = (190, 195, 200)   # Overcast grey
            horizon_color = (170, 175, 180)
        else: # sunny
            top_color = (255, 190, 100)   # Blue sky (BGR)
            horizon_color = (240, 230, 210)

        # Gradient sky
        for y in range(horizon_y):
            alpha = y / float(horizon_y)
            color = [
                int(top_color[c] * (1 - alpha) + horizon_color[c] * alpha)
                for c in range(3)
            ]
            canvas[y, :] = color

        # Distant hills / mountains
        points = [(0, horizon_y)]
        num_peaks = random.randint(4, 7)
        for i in range(1, num_peaks):
            px = int(i * (self.w / num_peaks))
            py = horizon_y - random.randint(15, 55)
            points.append((px, py))
        points.append((self.w, horizon_y))
        hill_color = (50, 70, 50) if weather != "night" else (20, 25, 25)
        cv2.fillPoly(canvas, [np.array(points, dtype=np.int32)], hill_color)

    def _draw_road(self, canvas: np.ndarray, horizon_y: int, vanish_x: int):
        """Draws asphalt highway surface and lane markings."""
        # Road trapezoid
        road_left_top = vanish_x - random.randint(30, 60)
        road_right_top = vanish_x + random.randint(30, 60)
        road_left_bottom = -int(self.w * 0.1)
        road_right_bottom = int(self.w * 1.1)

        # Terrain (grass/soil shoulders)
        canvas[horizon_y:, :] = (45, 90, 45)  # Green grass

        # Asphalt
        road_pts = np.array([
            (road_left_top, horizon_y),
            (road_right_top, horizon_y),
            (road_right_bottom, self.h),
            (road_left_bottom, self.h)
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [road_pts], (60, 60, 65))

        # Lane markings (dashed center lines)
        num_dashes = 10
        for i in range(num_dashes):
            t1 = (i / float(num_dashes)) ** 2
            t2 = ((i + 0.5) / float(num_dashes)) ** 2
            y1 = int(horizon_y + t1 * (self.h - horizon_y))
            y2 = int(horizon_y + t2 * (self.h - horizon_y))
            x1 = int(vanish_x + (vanish_x - vanish_x) * t1)
            x2 = int(vanish_x + (vanish_x - vanish_x) * t2)
            cv2.line(canvas, (x1, y1), (x2, y2), (240, 240, 240), max(1, int(t2 * 6)))

        # Solid edge lines
        edge_pts_l1 = (road_left_top + 10, horizon_y)
        edge_pts_l2 = (road_left_bottom + int(self.w * 0.15), self.h)
        cv2.line(canvas, edge_pts_l1, edge_pts_l2, (255, 255, 255), 3)

        edge_pts_r1 = (road_right_top - 10, horizon_y)
        edge_pts_r2 = (road_right_bottom - int(self.w * 0.15), self.h)
        cv2.line(canvas, edge_pts_r1, edge_pts_r2, (0, 220, 240), 3)  # Yellow edge

        return road_pts

    def _draw_w_beam_guardrail(
        self,
        canvas: np.ndarray,
        side: str,
        horizon_y: int,
        is_damaged: bool = False
    ) -> List[Dict[str, any]]:
        """Draws metallic W-beam guardrail along the road shoulder."""
        boxes = []
        start_y = int(horizon_y + 15)
        end_y = int(self.h - 10)

        if side == "left":
            x_top = int(self.w * 0.28)
            x_bottom = int(self.w * 0.05)
        else:
            x_top = int(self.w * 0.72)
            x_bottom = int(self.w * 0.95)

        num_segments = 4
        y_steps = np.linspace(start_y, end_y, num_segments + 1).astype(int)
        x_steps = np.linspace(x_top, x_bottom, num_segments + 1).astype(int)

        for s in range(num_segments):
            y1, y2 = y_steps[s], y_steps[s + 1]
            x1, x2 = x_steps[s], x_steps[s + 1]

            rail_h = int(14 + (y2 / float(self.h)) * 32)
            damaged_this_seg = is_damaged and (s == 2 or s == 3)

            # Draw vertical steel I-beam posts
            num_posts = 3
            for p in range(num_posts):
                alpha = p / float(num_posts)
                post_x = int(x1 + alpha * (x2 - x1))
                post_y = int(y1 + alpha * (y2 - y1))
                post_h = rail_h + int(10 + alpha * 15)
                cv2.rectangle(
                    canvas,
                    (post_x - 3, post_y - rail_h // 2),
                    (post_x + 3, post_y + post_h - rail_h // 2),
                    (100, 100, 105),
                    -1
                )

            # Guardrail beam polygon
            if damaged_this_seg:
                # Deform polygon to simulate crash deformation
                offset = random.choice([-25, 25])
                pts = np.array([
                    (x1, y1 - rail_h // 2),
                    (x2 + offset, y2 - rail_h // 2 + 10),
                    (x2 + offset, y2 + rail_h // 2 + 15),
                    (x1, y1 + rail_h // 2)
                ], dtype=np.int32)
                rail_color = (90, 80, 130)  # Crushed / scratched steel
                cls_id = 3  # damaged_guardrail
            else:
                pts = np.array([
                    (x1, y1 - rail_h // 2),
                    (x2, y2 - rail_h // 2),
                    (x2, y2 + rail_h // 2),
                    (x1, y1 + rail_h // 2)
                ], dtype=np.int32)
                rail_color = (195, 200, 205)  # Galvanized steel
                cls_id = 0  # metal_w_beam

            cv2.fillPoly(canvas, [pts], rail_color)
            cv2.polylines(canvas, [pts], True, (120, 120, 125), 1)

            # Corrugation line (w-beam groove)
            mid_pts = np.array([
                (x1, y1),
                (x2 if not damaged_this_seg else x2 + offset, y2 + (12 if damaged_this_seg else 0))
            ], dtype=np.int32)
            cv2.polylines(canvas, [mid_pts], False, (140, 140, 145), 2)

            # Compute bounding box
            min_x = max(0, int(np.min(pts[:, 0]) - 5))
            min_y = max(0, int(np.min(pts[:, 1]) - 5))
            max_x = min(self.w, int(np.max(pts[:, 0]) + 5))
            max_y = min(self.h, int(np.max(pts[:, 1]) + 5))

            if max_x - min_x > 10 and max_y - min_y > 10:
                boxes.append({
                    "bbox": [min_x, min_y, max_x, max_y],
                    "class_id": cls_id
                })

        return boxes

    def _draw_concrete_barrier(
        self,
        canvas: np.ndarray,
        side: str,
        horizon_y: int,
        is_damaged: bool = False
    ) -> List[Dict[str, any]]:
        """Draws concrete Jersey barrier along highway median or shoulder."""
        boxes = []
        start_y = int(horizon_y + 10)
        end_y = int(self.h - 10)

        if side == "left":
            x_top = int(self.w * 0.32)
            x_bottom = int(self.w * 0.12)
        else:
            x_top = int(self.w * 0.68)
            x_bottom = int(self.w * 0.88)

        num_segments = 3
        y_steps = np.linspace(start_y, end_y, num_segments + 1).astype(int)
        x_steps = np.linspace(x_top, x_bottom, num_segments + 1).astype(int)

        for s in range(num_segments):
            y1, y2 = y_steps[s], y_steps[s + 1]
            x1, x2 = x_steps[s], x_steps[s + 1]
            h_segment = int(18 + (y2 / float(self.h)) * 40)
            damaged_this_seg = is_damaged and s == 1

            pts = np.array([
                (x1, y1 - h_segment // 2),
                (x2, y2 - h_segment // 2),
                (x2 + 8, y2 + h_segment // 2),
                (x1 + 4, y1 + h_segment // 2)
            ], dtype=np.int32)

            color = (130, 135, 140) if not damaged_this_seg else (90, 95, 110)
            cls_id = 1 if not damaged_this_seg else 3

            cv2.fillPoly(canvas, [pts], color)
            cv2.polylines(canvas, [pts], True, (90, 90, 95), 1)

            # Cracks / fracture lines if damaged
            if damaged_this_seg:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.line(canvas, (cx - 10, cy - 15), (cx + 10, cy + 15), (30, 30, 35), 2)
                cv2.line(canvas, (cx, cy), (cx + 8, cy - 8), (30, 30, 35), 2)

            min_x = max(0, int(np.min(pts[:, 0]) - 4))
            min_y = max(0, int(np.min(pts[:, 1]) - 4))
            max_x = min(self.w, int(np.max(pts[:, 0]) + 4))
            max_y = min(self.h, int(np.max(pts[:, 1]) + 4))

            if max_x - min_x > 10 and max_y - min_y > 10:
                boxes.append({
                    "bbox": [min_x, min_y, max_x, max_y],
                    "class_id": cls_id
                })

        return boxes

    def _draw_hazard_posts(
        self,
        canvas: np.ndarray,
        side: str,
        horizon_y: int
    ) -> List[Dict[str, any]]:
        """Draws delineator guide posts with retro-reflective bands."""
        boxes = []
        if side == "left":
            x_top = int(self.w * 0.25)
            x_bottom = int(self.w * 0.02)
        else:
            x_top = int(self.w * 0.75)
            x_bottom = int(self.w * 0.98)

        num_posts = random.randint(3, 5)
        for i in range(num_posts):
            alpha = (i + 1) / float(num_posts + 1)
            py = int(horizon_y + 30 + alpha * (self.h - horizon_y - 60))
            px = int(x_top + alpha * (x_bottom - x_top))

            post_h = int(12 + alpha * 35)
            post_w = max(4, int(2 + alpha * 6))

            x1 = px - post_w // 2
            y1 = py - post_h
            x2 = px + post_w // 2
            y2 = py

            # White post body
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (230, 230, 230), -1)

            # Red or Amber reflective band
            band_y1 = y1 + int(post_h * 0.2)
            band_y2 = y1 + int(post_h * 0.45)
            cv2.rectangle(canvas, (x1, band_y1), (x2, band_y2), (0, 0, 255), -1)

            boxes.append({
                "bbox": [max(0, x1 - 2), max(0, y1 - 2), min(self.w, x2 + 2), min(self.h, y2 + 2)],
                "class_id": 4  # hazard_post
            })

        return boxes

    def generate_scene(
        self,
        weather: Optional[str] = None,
        force_damaged: bool = False
    ) -> Tuple[np.ndarray, List[Dict[str, any]], Dict[str, any]]:
        """Generates a complete synthetic highway scene with annotations."""
        weathers = ["sunny", "sunset", "night", "foggy"]
        weather = weather or random.choice(weathers)

        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        horizon_y = int(self.h * 0.45 + random.randint(-15, 20))
        vanish_x = int(self.w * 0.5 + random.randint(-30, 30))

        # 1. Background sky and landscape
        self._draw_sky_and_horizon(canvas, horizon_y, weather)

        # 2. Highway road
        self._draw_road(canvas, horizon_y, vanish_x)

        # 3. Guardrails and barriers on left and right sides
        all_boxes = []

        # Left barrier selection
        left_type = random.choice(["metal", "concrete", "none"])
        left_damaged = force_damaged or (random.random() < 0.25)

        if left_type == "metal":
            all_boxes.extend(self._draw_w_beam_guardrail(canvas, "left", horizon_y, left_damaged))
        elif left_type == "concrete":
            all_boxes.extend(self._draw_concrete_barrier(canvas, "left", horizon_y, left_damaged))

        # Right barrier selection
        right_type = random.choice(["metal", "concrete", "posts"])
        right_damaged = (force_damaged and left_type != "metal") or (random.random() < 0.25)

        if right_type == "metal":
            all_boxes.extend(self._draw_w_beam_guardrail(canvas, "right", horizon_y, right_damaged))
        elif right_type == "concrete":
            all_boxes.extend(self._draw_concrete_barrier(canvas, "right", horizon_y, right_damaged))
        elif right_type == "posts":
            all_boxes.extend(self._draw_hazard_posts(canvas, "right", horizon_y))

        # Add camera sensor noise or blur
        noise = np.random.normal(0, random.uniform(2, 6), canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        metadata = {
            "weather": weather,
            "has_damage": any(b["class_id"] == 3 for b in all_boxes),
            "num_annotations": len(all_boxes)
        }

        return canvas, all_boxes, metadata


def generate_dataset(
    output_dir: str,
    num_train: int = 150,
    num_val: int = 30,
    img_size: Tuple[int, int] = (512, 512)
):
    """Generates train and val sets with images and YOLO formatted labels."""
    generator = RoadSceneGenerator(img_size)

    splits = {
        "train": num_train,
        "val": num_val
    }

    for split, count in splits.items():
        images_dir = os.path.join(output_dir, split, "images")
        labels_dir = os.path.join(output_dir, split, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        print(f"Generating {count} {split} scenes...")
        for i in range(count):
            damaged = (i % 3 == 0)  # Ensure healthy representation of damaged guardrails
            img, boxes, _ = generator.generate_scene(force_damaged=damaged)

            base_name = f"road_{split}_{i:04d}"
            img_path = os.path.join(images_dir, f"{base_name}.jpg")
            label_path = os.path.join(labels_dir, f"{base_name}.txt")

            # Save image
            cv2.imwrite(img_path, img)

            # Save YOLO format labels: class_id cx cy w h (normalized)
            with open(label_path, "w") as f:
                for b in boxes:
                    cid = b["class_id"]
                    x1, y1, x2, y2 = b["bbox"]
                    cx = ((x1 + x2) / 2.0) / img_size[0]
                    cy = ((y1 + y2) / 2.0) / img_size[1]
                    w = (x2 - x1) / float(img_size[0])
                    h = (y2 - y1) / float(img_size[1])
                    f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # Generate dedicated sample media for immediate UI testing
    sample_dir = os.path.join(output_dir, "sample_media")
    os.makedirs(sample_dir, exist_ok=True)

    sample_scenarios = [
        ("sunny_highway_intact.jpg", "sunny", False),
        ("damaged_guardrail_crash.jpg", "sunny", True),
        ("sunset_concrete_barrier.jpg", "sunset", False),
        ("night_highway_inspection.jpg", "night", False),
        ("foggy_mountain_pass.jpg", "foggy", True),
    ]

    for fname, weather, force_dam in sample_scenarios:
        s_img, s_boxes, _ = generator.generate_scene(weather=weather, force_damaged=force_dam)
        cv2.imwrite(os.path.join(sample_dir, fname), s_img)

    print("Dataset generation complete!")


if __name__ == "__main__":
    generate_dataset("data", num_train=120, num_val=25)
