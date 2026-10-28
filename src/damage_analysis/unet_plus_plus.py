"""
src/damage_analysis/unet_plus_plus.py
-------------------------------------
U-Net++ (Nested U-Net) architecture for stone inscription damage and line segmentation.

Reference:
  Z. Zhou, M. M. Rahman Siddiquee, N. Tajbakhsh, J. Liang,
  "UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation",
  IEEE Trans. Med. Imaging, vol. 39, no. 6, pp. 1856-1867, 2020.
  (Referenced in EpiRecover PPT Slide 5, Item 6, and Slide 12).

Architecture:
  - Deep supervision with nested, dense skip connections across multiscale feature levels.
  - Reduces the semantic gap between sub-networks of encoder and decoder.

EpiRecover, 2026.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

log = logging.getLogger(__name__)


if _HAS_TORCH:
    class ConvBlock(nn.Module):
        """Standard Conv-BN-ReLU block (two 3x3 convs)."""
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            return x

    class UNetPlusPlus(nn.Module):
        """
        UNet++ nested dense architecture for 2D stone inscription damage segmentation.
        """
        def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 1,
            feature_channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
            deep_supervision: bool = False,
        ) -> None:
            super().__init__()
            self.deep_supervision = deep_supervision
            nb_filter = feature_channels

            self.pool = nn.MaxPool2d(2, 2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

            self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
            self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
            self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
            self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
            self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

            self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
            self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
            self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
            self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])

            self.conv0_2 = ConvBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
            self.conv1_2 = ConvBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
            self.conv2_2 = ConvBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

            self.conv0_3 = ConvBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
            self.conv1_3 = ConvBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

            self.conv0_4 = ConvBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x0_0 = self.conv0_0(x)
            x1_0 = self.conv1_0(self.pool(x0_0))
            x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

            x2_0 = self.conv2_0(self.pool(x1_0))
            x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
            x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

            x3_0 = self.conv3_0(self.pool(x2_0))
            x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
            x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
            x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

            x4_0 = self.conv4_0(self.pool(x3_0))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

            output = self.final(x0_4)
            return output
else:
    class UNetPlusPlus:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for UNetPlusPlus.")


class UNetPlusPlusDamageSegmentor:
    """
    Interface for UNet++ stone inscription damage segmentation.
    Honest reporting: if model weights file does not exist, marks as unweighted.
    """

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self.weights_path = weights_path
        self.model = None
        self.is_ready = False

        if not _HAS_TORCH:
            log.warning("PyTorch not installed. UNetPlusPlusDamageSegmentor unavailable.")
            return

        if weights_path and os.path.exists(weights_path):
            try:
                self.model = UNetPlusPlus(in_channels=1, num_classes=1)
                state = torch.load(weights_path, map_location="cpu")
                self.model.load_state_dict(state)
                self.model.eval()
                self.is_ready = True
                log.info("UNet++ model loaded successfully from %s", weights_path)
            except Exception as e:
                log.warning("Failed to load UNet++ weights from %s: %s", weights_path, e)
        else:
            log.info("UNet++ architecture initialized. Model weights (%s) not provided.", weights_path)

    def predict_mask(self, grayscale_img: np.ndarray, threshold: float = 0.5) -> Optional[np.ndarray]:
        """
        Run inference if weights exist, otherwise returns None.
        """
        if not self.is_ready or self.model is None or not _HAS_TORCH:
            return None

        h, w = grayscale_img.shape[:2]
        # Resize to multiple of 32 for UNet pooling
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        padded = np.pad(grayscale_img, ((0, pad_h), (0, pad_w)), mode="reflect")

        tensor = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0) / 255.0
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze().numpy()

        unpadded_probs = probs[:h, :w]
        mask = (unpadded_probs >= threshold).astype(np.uint8) * 255
        return mask
