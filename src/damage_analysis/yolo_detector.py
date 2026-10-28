"""
src/damage_analysis/yolo_detector.py
------------------------------------
YOLO-based optional localization stage for monument panel and line detection.

As specified in EpiRecover PPT Slide 12:
  Damage + line map: UNet++ +(YOLO optional)

Uses ultralytics (if installed) or graceful stub when weights are unprovided.
Clearly labels outputs as trained deep learning only when verified weights are present.

EpiRecover, 2026.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

log = logging.getLogger(__name__)

_HAS_ULTRALYTICS = False
try:
    from ultralytics import YOLO as _YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    pass


@dataclass
class DetectedRegion:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h


class YOLORegionDetector:
    """
    Optional YOLO detector for monument inscription panels, lines, or erosion zones.
    """

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self.weights_path = weights_path
        self.model = None
        self.is_ready = False

        if not _HAS_ULTRALYTICS:
            log.info("ultralytics not installed. YOLORegionDetector operating in stub mode.")
            return

        if weights_path and os.path.exists(weights_path):
            try:
                self.model = _YOLO(weights_path)
                self.is_ready = True
                log.info("YOLO model loaded from %s", weights_path)
            except Exception as e:
                log.warning("Could not load YOLO weights from %s: %s", weights_path, e)
        else:
            log.info("YOLO detector ready. No custom weights supplied at %s.", weights_path)

    def detect(self, img: np.ndarray, conf_threshold: float = 0.25) -> List[DetectedRegion]:
        """
        Run detection on image. Returns list of DetectedRegion objects.
        """
        if not self.is_ready or self.model is None:
            return []

        results = self.model(img, conf=conf_threshold, verbose=False)
        regions: List[DetectedRegion] = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                cls_name = r.names.get(cls_idx, f"class_{cls_idx}")
                regions.append(DetectedRegion(
                    label=cls_name,
                    confidence=round(conf, 4),
                    bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                ))
        return regions
