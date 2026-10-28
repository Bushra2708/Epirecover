"""
src/restoration/nafnet_restoration.py
-------------------------------------
NAFNet Image Enhancement and Restoration Branch for EpiRecover.

As specified in EpiRecover PPT Slide 12:
  "Restoration NAFNet: enhancement branch / ablation"
  and Slide 14:
  "Ablation of evidence sources... Enhancement branch with and without NAFNet"

Reference:
  L. Chen et al., "Simple Baselines for Image Restoration" (NAFNet), ECCV 2022.

Scientific Integrity Guarantee:
  - This module is an ablation branch for visual denoising and contrast enhancement.
  - Image restoration CANNOT restore physically absent characters with certainty.
  - If trained weights are unavailable, the module operates in passthrough/classical mode
    and reports this clearly without claiming deep neural enhancement.

EpiRecover, 2026.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr
    from skimage.metrics import structural_similarity as compute_ssim
except ImportError:
    def compute_psnr(a, b):
        return 0.0
    def compute_ssim(a, b, channel_axis=None):
        return 0.0

log = logging.getLogger(__name__)


if _HAS_TORCH:
    class SimpleGate(nn.Module):
        """Element-wise multiplication of feature halves (replaces non-linear activation)."""
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.chunk(2, dim=1)
            return x1 * x2

    class NAFBlock(nn.Module):
        """Nonlinear Activation Free Block for image restoration."""
        def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
            super().__init__()
            dw_channel = c * dw_expand
            self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, bias=True)
            self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel, bias=True)
            self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, bias=True)

            self.sg1 = SimpleGate()

            # Simplified channel attention
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, bias=True),
            )

            # Feed-forward network
            ffn_channel = ffn_expand * c
            self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, bias=True)
            self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, bias=True)
            self.sg2 = SimpleGate()

            self.norm1 = nn.GroupNorm(1, c)
            self.norm2 = nn.GroupNorm(1, c)

            self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        def forward(self, inp: torch.Tensor) -> torch.Tensor:
            x = inp
            x = self.norm1(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.sg1(x)
            x = x * self.sca(x)
            x = self.conv3(x)
            y = inp + x * self.beta

            x = self.norm2(y)
            x = self.conv4(x)
            x = self.sg2(x)
            x = self.conv5(x)
            return y + x * self.gamma

    class NAFNet(nn.Module):
        """Lightweight NAFNet architecture for lithic image restoration ablation."""
        def __init__(self, in_channels: int = 3, out_channels: int = 3, width: int = 32, num_blocks: int = 4) -> None:
            super().__init__()
            self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=True)
            self.body = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])
            self.ending = nn.Conv2d(width, out_channels, kernel_size=3, padding=1, bias=True)

        def forward(self, inp: torch.Tensor) -> torch.Tensor:
            feats = self.intro(inp)
            feats = self.body(feats)
            out = self.ending(feats) + inp
            return out
else:
    class NAFNet:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for NAFNet.")


class NAFNetRestorationBranch:
    """
    Manages the optional NAFNet restoration ablation branch.
    """

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self.weights_path = weights_path
        self.model = None
        self.is_ready = False

        if not _HAS_TORCH:
            log.info("PyTorch not installed. NAFNet restoration branch disabled.")
            return

        if weights_path and os.path.exists(weights_path):
            try:
                self.model = NAFNet(in_channels=3, out_channels=3, width=32, num_blocks=4)
                state = torch.load(weights_path, map_location="cpu")
                self.model.load_state_dict(state)
                self.model.eval()
                self.is_ready = True
                log.info("NAFNet restoration model loaded from %s", weights_path)
            except Exception as e:
                log.warning("Could not load NAFNet weights from %s: %s", weights_path, e)
        else:
            log.info("NAFNet architecture ready. Pretrained weights (%s) not provided (ablation unweighted).", weights_path)

    def enhance(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Enhance image using NAFNet if ready, otherwise return original.

        Returns (enhanced_image_bgr, quality_metrics).
        """
        if not self.is_ready or self.model is None or not _HAS_TORCH:
            # Passthrough with classical bilateral smoothing baseline
            enhanced = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)
            is_identical = np.array_equal(image_bgr, enhanced)
            psnr_val = 100.0 if is_identical else float(np.round(compute_psnr(image_bgr, enhanced), 2))
            ssim_val = 1.0 if is_identical else float(np.round(compute_ssim(image_bgr, enhanced, channel_axis=2), 4))
            metrics = {
                "psnr": psnr_val,
                "ssim": ssim_val,
                "method": "bilateral_filter_fallback",
            }
            return enhanced, metrics

        # PyTorch NAFNet inference
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            out_tensor = self.model(tensor)
            out_rgb = torch.clamp(out_tensor.squeeze(0).permute(1, 2, 0) * 255.0, 0, 255).byte().numpy()

        enhanced_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        metrics = {
            "psnr": float(np.round(compute_psnr(image_bgr, enhanced_bgr), 2)),
            "ssim": float(np.round(compute_ssim(image_bgr, enhanced_bgr, channel_axis=2), 4)),
            "method": "nafnet_deep_enhancement",
        }
        return enhanced_bgr, metrics
