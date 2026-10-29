"""
Unit tests for GuardRailNet model components and loss functions.
"""

import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.backbone import GuardBackbone
from models.guardnet import GuardNet
from models.loss import GuardRailLoss, bbox_ciou


class TestGuardRailModel(unittest.TestCase):
    """Test suite for GuardRailNet architecture."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.img_size = 512
        self.num_classes = 5

    def test_backbone_multi_scale_outputs(self):
        """Verifies multi-scale feature strides (8, 16, 32)."""
        backbone = GuardBackbone(in_channels=3, base_channels=16).to(self.device)
        dummy_input = torch.randn(self.batch_size, 3, self.img_size, self.img_size, device=self.device)
        p3, p4, p5 = backbone(dummy_input)

        self.assertEqual(p3.shape, (self.batch_size, 64, self.img_size // 8, self.img_size // 8))
        self.assertEqual(p4.shape, (self.batch_size, 128, self.img_size // 16, self.img_size // 16))
        self.assertEqual(p5.shape, (self.batch_size, 256, self.img_size // 32, self.img_size // 32))

    def test_guardnet_forward_and_decode(self):
        """Verifies full GuardNet forward pass and decoding."""
        model = GuardNet(num_classes=self.num_classes, base_channels=16, neck_channels=64).to(self.device)
        dummy_input = torch.randn(self.batch_size, 3, self.img_size, self.img_size, device=self.device)

        outputs = model(dummy_input)
        self.assertEqual(len(outputs), 3)

        # Check output structure
        for out in outputs:
            self.assertIn("box", out)
            self.assertIn("obj", out)
            self.assertIn("cls", out)
            self.assertIn("stride", out)

        # Test decode
        detections = model.decode(outputs, conf_threshold=0.1, image_shape=(self.img_size, self.img_size))
        self.assertEqual(len(detections), self.batch_size)
        self.assertIsInstance(detections[0], list)

    def test_ciou_loss_computation(self):
        """Verifies CIoU calculation on identical and separated boxes."""
        b1 = torch.tensor([[50.0, 50.0, 150.0, 150.0]])
        b2 = torch.tensor([[50.0, 50.0, 150.0, 150.0]])
        ciou_perfect = bbox_ciou(b1, b2)
        self.assertAlmostEqual(ciou_perfect.item(), 1.0, places=4)

        b3 = torch.tensor([[300.0, 300.0, 400.0, 400.0]])
        ciou_apart = bbox_ciou(b1, b3)
        self.assertLess(ciou_apart.item(), 0.0)

    def test_multitask_loss_backward(self):
        """Verifies multi-task loss and gradient computation."""
        model = GuardNet(num_classes=self.num_classes, base_channels=16, neck_channels=64).to(self.device)
        criterion = GuardRailLoss(num_classes=self.num_classes)

        dummy_input = torch.randn(self.batch_size, 3, self.img_size, self.img_size, device=self.device)
        targets = [
            {
                "boxes": torch.tensor([[40.0, 200.0, 120.0, 350.0]], dtype=torch.float32),
                "classes": torch.tensor([0], dtype=torch.long)
            },
            {
                "boxes": torch.tensor([[380.0, 180.0, 480.0, 400.0]], dtype=torch.float32),
                "classes": torch.tensor([3], dtype=torch.long)
            }
        ]

        predictions = model(dummy_input)
        loss_dict = criterion(predictions, targets, img_size=(self.img_size, self.img_size))

        loss = loss_dict["loss"]
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        # Verify gradient backprop
        loss.backward()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())
                break


if __name__ == "__main__":
    unittest.main()
