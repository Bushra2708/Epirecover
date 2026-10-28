"""
src/damage_analysis/damage_detector.py
---------------------------------------
Damage localization module for EpiRecover.

Current implementation: HEURISTIC BASELINE using classical OpenCV operations.

IMPORTANT RESEARCH INTEGRITY NOTE:
------------------------------------
This module does NOT implement a trained deep-learning model.
The output is labelled "HEURISTIC_BASELINE" throughout.

A U-Net or U-Net++ segmentation model can be plugged in here once:
  1. A suitable labelled training dataset is available.
  2. The model has been trained and validated on held-out data.
  3. Model weights are saved to models/ and the path is set in config.

Until then, any damage mask produced by this module should be described
as a "classical image analysis baseline" in publications, not as
"deep-learning predicted damage".

Heuristic method:
  - Convert the preprocessed binary image to grayscale.
  - Locate regions where pixel intensity falls below a threshold
    (very dark areas typically correspond to deep erosion, gaps, or shadow).
  - Apply morphological dilation to grow the detected regions.
  - Return a binary mask and a list of bounding-box regions.

Reference for future U-Net integration:
  Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
  Segmentation", MICCAI 2015.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DamageRegion:
    """A single detected or annotated damaged region."""
    x: int
    y: int
    w: int
    h: int
    severity: str = "unknown"       # "light" | "medium" | "heavy" | "unknown"
    confidence: float = 0.0
    method: str = "heuristic"       # "heuristic" | "annotation" | "unet"

    def as_dict(self) -> Dict:
        return {
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "severity": self.severity, "confidence": self.confidence,
            "method": self.method,
        }


@dataclass
class DamageAnalysisResult:
    """Output of a single damage analysis run."""
    method: str = "heuristic"
    label: str = "HEURISTIC_BASELINE"
    mask: Optional[np.ndarray] = None       # binary uint8 mask, 255=damaged
    overlay: Optional[np.ndarray] = None    # original image with regions drawn
    regions: List[DamageRegion] = field(default_factory=list)
    mask_path: Optional[str] = None

    def as_dict(self) -> Dict:
        return {
            "method": self.method,
            "label": self.label,
            "regions": [r.as_dict() for r in self.regions],
            "mask_path": self.mask_path,
        }


# ---------------------------------------------------------------------------
# Heuristic baseline
# ---------------------------------------------------------------------------

class HeuristicDamageDetector:
    """
    Classical OpenCV-based damage localization.

    Detects candidate damaged regions by thresholding dark pixels on the
    preprocessed binary image and applying morphological dilation.

    Label: HEURISTIC_BASELINE — not a trained model prediction.
    """

    LABEL = "HEURISTIC_BASELINE"

    def __init__(
        self,
        dark_threshold: int = 40,
        min_area: int = 50,
        dilation_iter: int = 2,
    ) -> None:
        self.dark_threshold = dark_threshold
        self.min_area = min_area
        self.dilation_iter = dilation_iter

    def analyze(self, original: np.ndarray, preprocessed: np.ndarray) -> DamageAnalysisResult:
        """
        Produce a heuristic damage mask.

        Parameters
        ----------
        original : np.ndarray
            Original BGR image (used for the overlay visualization).
        preprocessed : np.ndarray
            Binary/cleaned image from preprocessing stage B.

        Returns
        -------
        DamageAnalysisResult with method='heuristic' and label='HEURISTIC_BASELINE'.
        """
        # Work on a grayscale version of the preprocessed image
        if len(preprocessed.shape) == 3:
            gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        else:
            gray = preprocessed.copy()

        # Threshold: very dark pixels (< threshold) are candidate damage
        _, dark_mask = cv2.threshold(gray, self.dark_threshold, 255, cv2.THRESH_BINARY_INV)

        # Dilate to grow detected regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(dark_mask, kernel, iterations=self.dilation_iter)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: List[DamageRegion] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Estimate severity heuristically by area fraction
            img_area = gray.shape[0] * gray.shape[1]
            frac = area / img_area if img_area > 0 else 0
            severity = "heavy" if frac > 0.05 else ("medium" if frac > 0.01 else "light")
            regions.append(DamageRegion(
                x=x, y=y, w=w, h=h,
                severity=severity,
                confidence=0.0,    # heuristic; no calibrated confidence
                method="heuristic",
            ))

        # Build overlay on the original BGR image
        overlay = original.copy()
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        for r in regions:
            cv2.rectangle(
                overlay, (r.x, r.y), (r.x + r.w, r.y + r.h),
                (0, 0, 200), 2,
            )
        # Label the overlay so it cannot be mistaken for a trained-model output
        cv2.putText(
            overlay, "HEURISTIC BASELINE",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )

        log.info(
            "HeuristicDamageDetector: found %d candidate damage regions (label=%s)",
            len(regions), self.LABEL,
        )
        return DamageAnalysisResult(
            method="heuristic",
            label=self.LABEL,
            mask=dilated,
            overlay=overlay,
            regions=regions,
        )


# ---------------------------------------------------------------------------
# U-Net plug-in point (not yet implemented)
# ---------------------------------------------------------------------------

class UNetDamageDetector:
    """
    Placeholder for a trained U-Net segmentation model.

    NOT IMPLEMENTED. This class will raise NotImplementedError until:
      1. A suitable labelled dataset is compiled.
      2. The model is trained and weights are saved to models/.
      3. This class is completed with actual inference code.

    Do not instantiate this class in production until these conditions are met.
    """

    LABEL = "UNET_PREDICTION"

    def __init__(self, weights_path: str) -> None:
        raise NotImplementedError(
            "U-Net damage segmentation model is not yet implemented. "
            "A training dataset and model weights are required. "
            "Use HeuristicDamageDetector as the current baseline."
        )

    def analyze(self, original: np.ndarray, preprocessed: np.ndarray) -> DamageAnalysisResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

class DamageDetector:
    """
    Selects and runs the appropriate damage detection method based on config.

    Currently always selects the heuristic baseline unless a U-Net weights
    path is configured AND the UNetDamageDetector is fully implemented.
    """

    def __init__(self, cfg: Optional[Dict] = None) -> None:
        cfg = cfg or {}
        damage_cfg = cfg.get("damage_analysis", {})
        method = damage_cfg.get("method", "heuristic")
        weights = damage_cfg.get("unet_weights")

        if method == "unet" and weights:
            log.warning(
                "U-Net method requested but not implemented — falling back to heuristic baseline."
            )
            method = "heuristic"

        if method == "heuristic":
            self._detector = HeuristicDamageDetector(
                dark_threshold=damage_cfg.get("dark_threshold", 40),
                min_area=damage_cfg.get("min_damage_area", 50),
                dilation_iter=damage_cfg.get("dilation_iterations", 2),
            )
        else:
            log.warning("Unknown damage method '%s' — using heuristic.", method)
            self._detector = HeuristicDamageDetector()

    def analyze(self, original: np.ndarray, preprocessed: np.ndarray) -> Dict:
        """
        Run damage analysis and return a plain dict compatible with
        EpiRecoverPipeline.stage_c_damage().
        """
        result = self._detector.analyze(original, preprocessed)
        return {
            "method": result.method,
            "label": result.label,
            "mask": result.mask,
            "overlay": result.overlay,
            "regions": [r.as_dict() for r in result.regions],
        }
