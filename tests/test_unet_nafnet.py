"""
tests/test_unet_nafnet.py
-------------------------
Tests for UNet++, YOLO detector, and NAFNet restoration branch.
"""

import unittest
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for UNet++ and NAFNet deep model tests")
from src.damage_analysis.unet_plus_plus import UNetPlusPlus, UNetPlusPlusDamageSegmentor
from src.damage_analysis.yolo_detector import YOLORegionDetector
from src.restoration.nafnet_restoration import NAFNet, NAFNetRestorationBranch


class TestDeepModelsAndRestoration(unittest.TestCase):

    def test_unet_plus_plus_forward_shape(self):
        """UNet++ forward pass produces 1-channel segmentation logits matching input spatial dims."""
        model = UNetPlusPlus(in_channels=1, num_classes=1, feature_channels=(16, 32, 64, 128, 256))
        model.eval()
        dummy_input = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (1, 1, 64, 64))

    def test_unet_segmentor_honest_unweighted_fallback(self):
        """UNetPlusPlusDamageSegmentor without weights reports is_ready=False and returns None."""
        segmentor = UNetPlusPlusDamageSegmentor(weights_path=None)
        self.assertFalse(segmentor.is_ready)
        img = np.full((100, 100), 200, dtype=np.uint8)
        mask = segmentor.predict_mask(img)
        self.assertIsNone(mask)

    def test_nafnet_forward_shape(self):
        """NAFNet forward pass preserves input 3-channel spatial dimensions."""
        model = NAFNet(in_channels=3, out_channels=3, width=16, num_blocks=2)
        model.eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (1, 3, 32, 32))

    def test_nafnet_restoration_branch_fallback(self):
        """NAFNet branch without weights falls back to classical bilateral filter and reports PSNR/SSIM."""
        branch = NAFNetRestorationBranch(weights_path=None)
        img = np.full((100, 100, 3), 150, dtype=np.uint8)
        enhanced, metrics = branch.enhance(img)
        self.assertEqual(enhanced.shape, (100, 100, 3))
        self.assertIn("psnr", metrics)
        self.assertIn("ssim", metrics)
        self.assertEqual(metrics["method"], "bilateral_filter_fallback")

    def test_yolo_detector_unweighted_returns_empty(self):
        """YOLO detector without weights returns empty list without error."""
        detector = YOLORegionDetector(weights_path=None)
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        regions = detector.detect(img)
        self.assertEqual(len(regions), 0)


if __name__ == "__main__":
    unittest.main()
