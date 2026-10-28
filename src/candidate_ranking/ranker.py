"""
src/candidate_ranking/ranker.py
--------------------------------
Evidence-grounded candidate ranking for EpiRecover.

As specified in the EpiRecover Architecture (PPT Slides 11, 12):
  Multimodal evidence fusion: V + O + L + H + R -> candidate score S(c)
  followed by uncertainty calibration (temperature scaling) and abstention evaluation.

Components:
  1. V (Visual Evidence)
  2. O (OCR Confidence)
  3. L (Linguistic / Script Compatibility)
  4. H (Historical Metadata Alignment)
  5. R (Retrieval Evidence Score)

Research integrity:
  - If weights are uncalibrated heuristics, they are explicitly tagged as HEURISTIC_DEFAULTS.
  - Component breakdowns are attached to every candidate for auditability.
  - Generates abstention decisions when top candidate confidence is low or ambiguous.

EpiRecover, 2026.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from utils import ReconstructionCandidate, Provenance
from src.candidate_ranking.fusion import LearnedEvidenceFusion
from src.candidate_ranking.calibration import temperature_scale, TemperatureScaler
from src.candidate_ranking.abstention import AbstentionEngine, AbstentionDecision

log = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "V": 0.20,
    "O": 0.25,
    "L": 0.15,
    "H": 0.15,
    "R": 0.25,
}


class CandidateRanker:
    """
    Ranks reconstruction candidates using multimodal evidence fusion:
    V + O + L + H + R -> candidate score.
    """

    def __init__(self, cfg: Optional[Dict] = None) -> None:
        ranking_cfg = (cfg or {}).get("ranking", {})
        custom_weights = ranking_cfg.get("weights", {})
        
        # Support legacy weight keys if present in config
        mapped_weights = {}
        legacy_map = {
            "visual_evidence": "V",
            "ocr_confidence": "O",
            "script_compatibility": "L",
            "source_quality": "H",
            "corpus_similarity": "R",
        }
        for k, v in custom_weights.items():
            mapped_key = legacy_map.get(k, k)
            mapped_weights[mapped_key] = v

        self._weights = {**DEFAULT_WEIGHTS, **mapped_weights}
        self._validated = ranking_cfg.get("weights_validated", False)
        
        # Instantiate fusion, calibration, and abstention engines
        self.fusion = LearnedEvidenceFusion(weights=self._weights)
        self.fusion.is_learned = self._validated
        self.scaler = TemperatureScaler(default_temperature=ranking_cfg.get("temperature", 1.0))
        self.abstention_engine = AbstentionEngine(
            confidence_threshold=ranking_cfg.get("abstention_threshold", 0.35),
            margin_threshold=ranking_cfg.get("margin_threshold", 0.08),
        )

        if not self._validated:
            log.info(
                "CandidateRanker: using HEURISTIC fusion weights (V=%.2f, O=%.2f, L=%.2f, H=%.2f, R=%.2f). "
                "Scores represent provisional ranking, not calibrated probabilities.",
                self._weights["V"], self._weights["O"], self._weights["L"],
                self._weights["H"], self._weights["R"],
            )

    def rank(
        self,
        candidates: List[ReconstructionCandidate],
        ocr_result,
        retrieved_records: List[Dict],
        metadata=None,
        damage_regions: Optional[List[Dict]] = None,
    ) -> List[ReconstructionCandidate]:
        """
        Rank candidates by evidence-grounded score.

        Attaches rank_score and rank_details to each candidate.
        Returns list sorted descending by rank_score.
        """
        if not candidates:
            return []

        ocr_conf = ocr_result.mean_confidence if ocr_result else 0.0

        for candidate in candidates:
            # Score candidate using PPT V+O+L+H+R fusion
            score, features = self.fusion.score_candidate(
                candidate=candidate,
                ocr_confidence=ocr_conf,
                retrieved_records=retrieved_records,
                metadata=metadata,
                damage_regions=damage_regions,
            )

            # Apply temperature scaling if calibrated
            calibrated_score = float(temperature_scale([score], self.scaler.temperature)[0]) if self.scaler.is_fitted else score

            candidate.rank_score = round(calibrated_score, 4)
            # Legacy compatible details along with V, O, L, H, R
            candidate.rank_details = {
                "V": features["V"],
                "O": features["O"],
                "L": features["L"],
                "H": features["H"],
                "R": features["R"],
                "visual_evidence": features["V"],
                "ocr_confidence": features["O"],
                "script_compatibility": features["L"],
                "source_quality": features["H"],
                "corpus_similarity": features["R"],
            }

        candidates.sort(key=lambda c: (c.rank_score if c.rank_score is not None else -1.0), reverse=True)
        log.info(
            "CandidateRanker: ranked %d candidates (weights_validated=%s)",
            len(candidates), self._validated,
        )
        return candidates

    def decide_abstention(self, ranked_candidates: List[ReconstructionCandidate]) -> AbstentionDecision:
        """Evaluate abstention rule on the ranked candidates."""
        return self.abstention_engine.evaluate(ranked_candidates)
