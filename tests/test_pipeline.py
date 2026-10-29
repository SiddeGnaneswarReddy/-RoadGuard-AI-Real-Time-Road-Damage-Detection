"""
Unit tests for data pipeline, metrics, and visualizer.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset_generator import RoadSceneGenerator
from utils.metrics import evaluate_detections, box_iou_numpy
from utils.visualizer import draw_detections


class TestPipeline(unittest.TestCase):
    """Test suite for data generation, evaluation metrics, and visualizer."""

    def test_procedural_scene_generator(self):
        """Verifies synthetic scene generation and annotation shapes."""
        generator = RoadSceneGenerator(img_size=(256, 256))
        img, boxes, metadata = generator.generate_scene(weather="sunny", force_damaged=True)

        self.assertEqual(img.shape, (256, 256, 3))
        self.assertIsInstance(boxes, list)
        self.assertIn("weather", metadata)
        self.assertEqual(metadata["weather"], "sunny")

        for b in boxes:
            self.assertIn("bbox", b)
            self.assertIn("class_id", b)
            x1, y1, x2, y2 = b["bbox"]
            self.assertLessEqual(x1, x2)
            self.assertLessEqual(y1, y2)

    def test_box_iou_numpy(self):
        """Verifies IoU computation."""
        b1 = np.array([[10, 10, 50, 50]])
        b2 = np.array([[10, 10, 50, 50], [100, 100, 200, 200]])
        ious = box_iou_numpy(b1, b2)
        self.assertEqual(ious.shape, (1, 2))
        self.assertAlmostEqual(float(ious[0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(ious[0, 1]), 0.0, places=4)

    def test_metrics_evaluation(self):
        """Verifies evaluation metric calculation."""
        preds = [[
            {"bbox": [10, 10, 50, 50], "confidence": 0.9, "class_id": 0}
        ]]
        targets = [{
            "boxes": np.array([[10, 10, 50, 50]]),
            "classes": np.array([0])
        }]
        results = evaluate_detections(preds, targets, num_classes=5)
        self.assertIn("mAP@50", results)
        self.assertGreater(results["mAP@50"], 0.5)

    def test_visualizer(self):
        """Verifies visualizer HUD rendering."""
        dummy_img = np.zeros((300, 400, 3), dtype=np.uint8)
        detections = [
            {"bbox": [20, 50, 100, 200], "confidence": 0.88, "class_id": 0},
            {"bbox": [250, 60, 350, 220], "confidence": 0.92, "class_id": 3}
        ]
        annotated, stats = draw_detections(dummy_img, detections)
        self.assertEqual(annotated.shape, dummy_img.shape)
        self.assertEqual(stats["total_barriers"], 2)
        self.assertEqual(stats["damaged_count"], 1)
        self.assertIn("CRITICAL", stats["collision_risk"])


if __name__ == "__main__":
    unittest.main()
