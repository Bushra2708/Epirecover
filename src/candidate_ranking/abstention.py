"""
src/candidate_ranking/abstention.py
-----------------------------------
Abstention and Reject-Option Logic for EpiRecover.

Based on Chow's Optimal Error-Reject Tradeoff (1970) and modern learning-to-reject
literature (Zhang et al., 2023) referenced in PPT Slides 4, 7, 11, 12, 14:
  - If maximum candidate calibrated confidence is below threshold tau_abstain,
    or if evidence is contradictory (e.g. top-2 candidates have near-identical
    conflicting scores), the system abstains and outputs [UNKNOWN].
  - Evaluates risk-coverage curves:
      Coverage = fraction of gaps where system committed to a candidate reading.
      Risk (Error Rate) = error rate strictly on the covered (non-abstained) predictions.

EpiRecover, 2026.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from utils import ReconstructionCandidate, Provenance

log = logging.getLogger(__name__)

UNKNOWN_MARKER = "[UNKNOWN]"


@dataclass
class AbstentionDecision:
    """Represents a decision whether to commit to Top-1 reconstruction or abstain."""
    should_abstain: bool
    selected_candidate: Optional[ReconstructionCandidate]
    effective_text: str
    confidence: float
    reason: str
    uncertainty_level: str  # "LOW", "MEDIUM", "HIGH"


class AbstentionEngine:
    """
    Evaluates candidate rankings against confidence and margin thresholds
    to determine whether to abstain or accept a reconstruction reading.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.35,
        margin_threshold: float = 0.08,
    ) -> None:
        self.conf_thresh = confidence_threshold
        self.margin_thresh = margin_threshold

    def evaluate(
        self,
        ranked_candidates: List[ReconstructionCandidate],
    ) -> AbstentionDecision:
        """
        Decide whether to accept the top candidate or abstain.

        Parameters
        ----------
        ranked_candidates : List[ReconstructionCandidate]
            Candidates sorted descending by rank/calibrated score.

        Returns
        -------
        AbstentionDecision
        """
        if not ranked_candidates:
            return AbstentionDecision(
                should_abstain=True,
                selected_candidate=None,
                effective_text=UNKNOWN_MARKER,
                confidence=0.0,
                reason="No candidate readings generated.",
                uncertainty_level="HIGH",
            )

        top_cand = ranked_candidates[0]

        # 1. Check if top candidate is already explicit UNKNOWN
        if top_cand.provenance == Provenance.UNKNOWN or top_cand.candidate_text == UNKNOWN_MARKER:
            return AbstentionDecision(
                should_abstain=True,
                selected_candidate=top_cand,
                effective_text=UNKNOWN_MARKER,
                confidence=0.0,
                reason="Top ranked candidate is an explicit gap/unknown marker.",
                uncertainty_level="HIGH",
            )

        # 2. Confidence threshold check
        top_score = float(top_cand.rank_score if top_cand.rank_score is not None else top_cand.confidence)
        if top_score < self.conf_thresh:
            return AbstentionDecision(
                should_abstain=True,
                selected_candidate=top_cand,
                effective_text=UNKNOWN_MARKER,
                confidence=top_score,
                reason=f"Top candidate score ({top_score:.3f}) below abstention threshold ({self.conf_thresh:.3f}).",
                uncertainty_level="HIGH",
            )

        # 3. Margin ambiguity check (Chow's margin rule)
        if len(ranked_candidates) > 1:
            second_cand = ranked_candidates[1]
            if second_cand.provenance != Provenance.UNKNOWN and second_cand.candidate_text != UNKNOWN_MARKER:
                second_score = float(second_cand.rank_score if second_cand.rank_score is not None else second_cand.confidence)
                margin = top_score - second_score
                if margin < self.margin_thresh and top_cand.candidate_text != second_cand.candidate_text:
                    return AbstentionDecision(
                        should_abstain=True,
                        selected_candidate=top_cand,
                        effective_text=UNKNOWN_MARKER,
                        confidence=top_score,
                        reason=f"Contradictory evidence: top candidate margin ({margin:.3f}) below ambiguity threshold ({self.margin_thresh:.3f}).",
                        uncertainty_level="MEDIUM",
                    )

        # 4. Accept Top-1 candidate
        uncertainty = "LOW" if top_score >= 0.70 else "MEDIUM"
        return AbstentionDecision(
            should_abstain=False,
            selected_candidate=top_cand,
            effective_text=top_cand.candidate_text,
            confidence=top_score,
            reason=f"Accepted reading with sufficient evidence (score={top_score:.3f}).",
            uncertainty_level=uncertainty,
        )


def compute_risk_coverage_curve(
    confidences: np.ndarray | List[float],
    correctness: np.ndarray | List[int | bool],
    num_thresholds: int = 50,
) -> Dict[str, List[float]]:
    """
    Compute empirical Risk-Coverage curve across confidence thresholds.

    Coverage(tau) = N(conf >= tau) / N_total
    Risk(tau)     = 1 - Accuracy on covered subset = N(conf >= tau & error) / N(conf >= tau)

    Parameters
    ----------
    confidences : list/array of floats in [0, 1]
    correctness : list/array of 0/1 (1 if predicted candidate matches ground truth, 0 otherwise)

    Returns
    -------
    dict with keys: 'coverages', 'risks', 'thresholds', 'aurc' (Area Under Risk-Coverage curve)
    """
    confs = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness, dtype=np.float64)
    n_total = float(len(confs))

    if n_total == 0:
        return {"coverages": [1.0], "risks": [0.0], "thresholds": [0.0], "aurc": 0.0}

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    coverages: List[float] = []
    risks: List[float] = []

    for tau in thresholds:
        covered = confs >= tau
        n_cov = np.sum(covered)
        if n_cov == 0:
            coverages.append(0.0)
            risks.append(0.0)
        else:
            cov = float(n_cov) / n_total
            acc = float(np.mean(corr[covered]))
            risk = 1.0 - acc
            coverages.append(round(cov, 4))
            risks.append(round(risk, 4))

    # AURC trapezoidal approximation
    sorted_pairs = sorted(zip(coverages, risks), key=lambda x: x[0])
    cov_arr = np.array([p[0] for p in sorted_pairs])
    risk_arr = np.array([p[1] for p in sorted_pairs])
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aurc = float(np.round(trapz_fn(risk_arr, cov_arr), 4)) if (trapz_fn and len(cov_arr) > 1) else 0.0

    return {
        "coverages": coverages,
        "risks": risks,
        "thresholds": [round(float(t), 4) for t in thresholds],
        "aurc": aurc,
    }
