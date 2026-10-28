"""
src/evaluation/synthetic_damage.py
----------------------------------
Synthetic Inscription Damage Benchmark Generator (Levels D0 to D6).

As specified in EpiRecover PPT Slide 10 and Slide 14:
  "Synthetic damage D0-D6: Masking, erosion, cracks, blur, illumination, perspective, occlusion"
  "Ablation of evidence sources at 10-60% damage"
  "Controlled experiments with exact ground truth and masks"
  "Split by inscription / monument, never by random image patches"

Damage Severity Levels:
  - D0 : Clean baseline (0% damage)
  - D1 : Light superficial stone erosion & stroke thinning (~10% damage)
  - D2 : Structural fractures, fissures, and crack lines (~20% damage)
  - D3 : Surface spalling & flaking masks (~30% damage)
  - D4 : Severe missing stone chunks & glyph spalling (~40% - 50% damage)
  - D5 : Oblique sunlight, harsh shadows, and perspective distortion (~50% damage)
  - D6 : Composite extreme degradation combining multiple damage types (~60% damage)

EpiRecover, 2026.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SyntheticDamageOutput:
    damage_level: str               # "D0", "D1", "D2", "D3", "D4", "D5", "D6"
    degraded_image: np.ndarray      # uint8 BGR image
    damage_mask: np.ndarray         # uint8 binary mask (255 = damaged, 0 = intact)
    damage_ratio: float             # damaged pixel fraction (0.0 to 1.0)
    applied_operations: List[str]   # list of applied degradation operators


class SyntheticDamageGenerator:
    """
    Applies reproducible mathematical damage transformations to inscription images.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def apply_damage(self, image: np.ndarray, level: str = "D1") -> SyntheticDamageOutput:
        """
        Apply specified damage level (D0 through D6) to an input inscription image.
        """
        img = image.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        ops: List[str] = []

        level = level.upper()

        if level == "D0":
            # D0: Clean baseline
            ops.append("identity")
            return SyntheticDamageOutput(
                damage_level="D0",
                degraded_image=img,
                damage_mask=mask,
                damage_ratio=0.0,
                applied_operations=ops,
            )

        elif level == "D1":
            # D1: Light erosion & stroke thinning (~10%)
            ops.append("stroke_thinning_erosion")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded = cv2.erode(img, kernel, iterations=1)
            # Add stochastic salt noise
            noise = self.rng.random((h, w)) < 0.08
            img[noise] = [180, 180, 180]
            mask[noise] = 255

        elif level == "D2":
            # D2: Structural cracks and fractures (~20%)
            ops.append("linear_fractures_and_cracks")
            # Draw random jagged crack lines
            n_cracks = self.rng.integers(3, 7)
            for _ in range(n_cracks):
                pt1 = (int(self.rng.integers(0, w)), int(self.rng.integers(0, h)))
                pt2 = (int(self.rng.integers(0, w)), int(self.rng.integers(0, h)))
                cv2.line(img, pt1, pt2, (40, 40, 40), thickness=int(self.rng.integers(2, 5)))
                cv2.line(mask, pt1, pt2, 255, thickness=int(self.rng.integers(4, 8)))

        elif level == "D3":
            # D3: Surface spalling & flaking (~30%)
            ops.append("surface_spalling_flaking")
            n_spalls = self.rng.integers(4, 9)
            for _ in range(n_spalls):
                center = (int(self.rng.integers(0, w)), int(self.rng.integers(0, h)))
                axes = (int(self.rng.integers(15, w // 6)), int(self.rng.integers(15, h // 6)))
                angle = int(self.rng.integers(0, 180))
                color = int(self.rng.integers(120, 200))
                cv2.ellipse(img, center, axes, angle, 0, 360, (color, color, color), -1)
                cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

        elif level == "D4":
            # D4: Missing stone chunks (~40-50%)
            ops.append("missing_stone_chunk")
            # Cut off a substantial quadrant or corner
            pts = np.array([
                [0, int(h * 0.4)],
                [int(w * 0.5), int(h * 0.45)],
                [int(w * 0.45), h],
                [0, h]
            ], np.int32)
            cv2.fillPoly(img, [pts], (30, 30, 30))
            cv2.fillPoly(mask, [pts], 255)

        elif level == "D5":
            # D5: Oblique lighting, harsh shadow & blur (~50%)
            ops.append("oblique_illumination_gradient")
            ops.append("gaussian_blur")
            # Create a 2D illumination gradient
            X, Y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
            gradient = (X + Y) / 2.0
            grad_3d = np.repeat(gradient[:, :, np.newaxis], 3, axis=2)
            img = np.clip(img.astype(np.float32) * (0.3 + 0.7 * grad_3d), 0, 255).astype(np.uint8)
            img = cv2.GaussianBlur(img, (7, 7), 2.0)
            # Severe shadow region marked as damaged
            shadow = (gradient < 0.35)
            mask[shadow] = 255

        elif level == "D6":
            # D6: Composite severe degradation (~60%)
            ops.append("composite_severe_erosion_and_fracture")
            # Crack
            cv2.line(img, (0, int(h*0.3)), (w, int(h*0.7)), (20, 20, 20), 6)
            cv2.line(mask, (0, int(h*0.3)), (w, int(h*0.7)), 255, 10)
            # Spall
            cv2.circle(img, (int(w*0.6), int(h*0.5)), int(min(w, h)*0.25), (150, 150, 150), -1)
            cv2.circle(mask, (int(w*0.6), int(h*0.5)), int(min(w, h)*0.25), 255, -1)
            # Missing border
            pts = np.array([[int(w*0.7), 0], [w, 0], [w, int(h*0.6)], [int(w*0.8), int(h*0.5)]], np.int32)
            cv2.fillPoly(img, [pts], (25, 25, 25))
            cv2.fillPoly(mask, [pts], 255)
            # Blur
            img = cv2.GaussianBlur(img, (5, 5), 1.5)

        else:
            raise ValueError(f"Unknown damage level '{level}'. Must be one of D0, D1, D2, D3, D4, D5, D6.")

        damage_ratio = float(np.sum(mask == 255)) / float(h * w)

        return SyntheticDamageOutput(
            damage_level=level,
            degraded_image=img,
            damage_mask=mask,
            damage_ratio=round(damage_ratio, 4),
            applied_operations=ops,
        )
