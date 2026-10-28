"""
src/candidate_ranking/fusion.py
-------------------------------
Multimodal Evidence Fusion for EpiRecover Candidate Ranking.

As formulated in the EpiRecover architecture (PPT Slides 11, 12):
  V + O + L + H + R -> learned candidate score S(c)

Components:
  - V (Visual): compatibility with observed stone region & absence of severe spalling
  - O (OCR): character & line OCR confidence
  - L (Linguistic): Tamil script inventory, sandhi, virama (pulli) consistency
  - H (Historical): match with period, dynasty, monument, epigraphic formula
  - R (Retrieval): similarity score from corpus retrieval (exact, fuzzy, or dense)

Provides:
  - Feature extraction: maps candidate and context to feature vector [V, O, L, H, R]
  - LearnedFusionModel: scikit-learn LogisticRegression or Ridge model fitted on labelled pairs
  - Transparent heuristic fallback when weights are not yet trained on large-scale datasets.

EpiRecover, 2026.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from utils import ReconstructionCandidate, Provenance

log = logging.getLogger(__name__)


class MultimodalFeatureExtractor:
    """
    Extracts the 5 canonical PPT evidence dimensions [V, O, L, H, R]
    for an inscription reconstruction candidate.
    """

    def extract_features(
        self,
        candidate: ReconstructionCandidate,
        ocr_confidence: float,
        retrieved_records: List[Dict],
        metadata: Optional[Any] = None,
        damage_regions: Optional[List[Dict]] = None,
    ) -> Dict[str, float]:
        """
        Returns a dict of feature scores in [0.0, 1.0]:
          V: visual evidence compatibility
          O: OCR score
          L: linguistic / script compatibility
          H: historical metadata alignment
          R: retrieval evidence score
        """
        # 1. Visual evidence (V)
        # Higher if directly observed, reduced by nearby damage overlap
        prov_weights = {
            Provenance.OBSERVED: 1.0,
            Provenance.OCR_PRED: 0.65,
            Provenance.CORPUS: 0.40,
            Provenance.INFERRED: 0.30,
            Provenance.HUMAN: 1.0,
            Provenance.UNKNOWN: 0.0,
        }
        v_score = prov_weights.get(candidate.provenance, 0.0)

        # Ground CORPUS candidates in visual observation:
        # If OCR observed zero characters or [UNKNOWN], corpus candidate has no visual confirmation on the stone.
        if candidate.provenance == Provenance.CORPUS:
            if candidate.ocr_evidence is not None:
                ocr_ev = candidate.ocr_evidence.strip()
                cand_txt = (candidate.candidate_text or "").strip()
                if not ocr_ev or ocr_ev == "[UNKNOWN]":
                    v_score = 0.0
                else:
                    cand_chars = set(c for c in cand_txt.replace(" ", "") if "\u0b80" <= c <= "\u0bff")
                    ocr_chars = set(c for c in ocr_ev.replace(" ", "") if "\u0b80" <= c <= "\u0bff")
                    if cand_chars and ocr_chars:
                        overlap = len(cand_chars & ocr_chars) / max(len(cand_chars), 1)
                        v_score = min(0.40, round(0.10 + 0.30 * overlap, 4))
                    else:
                        v_score = 0.10

        if damage_regions and candidate.provenance == Provenance.UNKNOWN:
            v_score = 0.0  # Missing stone has zero visual character evidence
        elif damage_regions and len(damage_regions) > 2:
            v_score = max(0.1, v_score * 0.8)  # Mild visual penalty under heavy damage

        # Additional visual penalty: if candidate text is non-Tamil and provenance
        # is not HUMAN or CORPUS, the visual claim is unsupported by script evidence.
        text_for_check = candidate.candidate_text or ""
        if text_for_check and text_for_check != "[UNKNOWN]":
            stripped = text_for_check.replace(" ", "")
            if stripped:
                tamil_cnt = sum(1 for c in stripped if "\u0b80" <= c <= "\u0bff")
                tamil_ratio = tamil_cnt / len(stripped)
                if (
                    tamil_ratio < 0.10
                    and candidate.provenance not in (Provenance.HUMAN, Provenance.CORPUS)
                ):
                    # Script mismatch: visual score capped at 30% of normal value
                    v_score = max(0.0, v_score * 0.3)
                    log.debug(
                        "Fusion: visual score penalised (script mismatch: Tamil ratio=%.2f) "
                        "for candidate '%s'",
                        tamil_ratio, candidate.candidate_id,
                    )

        # 2. OCR evidence (O)
        if candidate.provenance == Provenance.UNKNOWN:
            o_score = 0.0
        elif candidate.provenance == Provenance.CORPUS:
            cand_tokens = set((candidate.candidate_text or "").split())
            ocr_tokens = set((candidate.ocr_evidence or "").split()) if candidate.ocr_evidence else set()
            if cand_tokens and ocr_tokens:
                overlap = len(cand_tokens & ocr_tokens) / max(len(cand_tokens), 1)
                o_score = (max(0.0, min(100.0, ocr_confidence)) / 100.0) * overlap
            else:
                o_score = 0.0
        elif candidate.provenance == Provenance.INFERRED:
            cand_tokens = set((candidate.candidate_text or "").split())
            ocr_tokens = set((candidate.ocr_evidence or "").split()) if candidate.ocr_evidence else set()
            overlap = len(cand_tokens & ocr_tokens) / max(len(cand_tokens), 1) if (cand_tokens and ocr_tokens) else 0.85
            o_score = (max(0.0, min(100.0, ocr_confidence)) / 100.0) * overlap
        else:
            o_score = max(0.0, min(1.0, ocr_confidence / 100.0))

        # 3. Linguistic & script constraints (L)
        text = (candidate.candidate_text or "").strip()
        if not text or text == "[UNKNOWN]":
            l_score = 0.0
        else:
            tamil_chars = sum(1 for c in text if "\u0b80" <= c <= "\u0bff")
            total_chars = max(1, len(text.replace(" ", "")))
            ratio = tamil_chars / total_chars
            if tamil_chars > 0:
                l_score = min(1.0, 0.3 + 0.7 * ratio)
            else:
                # Historical non-Tamil text or Latin Brahmi fixture in tests
                l_score = 0.5 if candidate.provenance == Provenance.CORPUS else 0.2

        # 4. Historical metadata alignment (H)
        h_score = 0.4
        if candidate.provenance == Provenance.UNKNOWN:
            h_score = 0.0
        elif candidate.provenance == Provenance.CORPUS:
            target_record = None
            if candidate.corpus_evidence and retrieved_records:
                for rec in retrieved_records:
                    if rec.get("id") == candidate.corpus_evidence:
                        target_record = rec
                        break

            if target_record and metadata:
                target_period = (getattr(metadata, "historical_period", None) or "").strip().lower()
                target_region = (getattr(metadata, "region", None) or "").strip().lower()
                target_monument = (getattr(metadata, "monument", None) or "").strip().lower()

                rec_period = (target_record.get("historical_period") or "").strip().lower()
                rec_region = (target_record.get("region") or "").strip().lower()
                rec_monument = (target_record.get("monument") or "").strip().lower()

                comparisons = 0
                matches = 0
                if target_period and rec_period:
                    comparisons += 1
                    if any(w in rec_period for w in target_period.split() if len(w) > 3):
                        matches += 1
                if target_region and rec_region:
                    comparisons += 1
                    if any(w in rec_region for w in target_region.split() if len(w) > 3):
                        matches += 1
                if target_monument and rec_monument:
                    comparisons += 1
                    if any(w in rec_monument for w in target_monument.split() if len(w) > 3):
                        matches += 1

                if comparisons > 0:
                    h_score = round(0.2 + 0.8 * (matches / comparisons), 4)
                else:
                    h_score = 0.5
            elif not metadata:
                h_score = 0.5
            else:
                h_score = 0.2
        elif metadata:
            target_period = (getattr(metadata, "historical_period", None) or "").strip().lower()
            if target_period and retrieved_records:
                matching_recs = [
                    r for r in retrieved_records
                    if target_period in (r.get("historical_period") or "").lower()
                ]
                h_score = 0.5 if matching_recs else 0.35
            else:
                h_score = 0.4

        # 5. Retrieval similarity (R)
        r_score = 0.0
        if candidate.provenance == Provenance.CORPUS:
            target_record = None
            if candidate.corpus_evidence and retrieved_records:
                for rec in retrieved_records:
                    if rec.get("id") == candidate.corpus_evidence:
                        target_record = rec
                        break
            if target_record:
                method = target_record.get("retrieval_method", "text")
                if method == "metadata_fallback":
                    # Metadata fallback is background context, NOT candidate text retrieval
                    r_score = 0.0
                else:
                    r_score = float(target_record.get("score", 0.0))
        elif candidate.provenance != Provenance.UNKNOWN and retrieved_records:
            cand_tokens = set(text.split())
            best_overlap = 0.0
            for rec in retrieved_records:
                if rec.get("retrieval_method") == "metadata_fallback":
                    continue
                trans_tokens = set((rec.get("transcription") or "").split())
                if cand_tokens and trans_tokens:
                    overlap = len(cand_tokens & trans_tokens) / max(len(cand_tokens), 1)
                    best_overlap = max(best_overlap, overlap * float(rec.get("score", 0.0)))
            r_score = best_overlap

        return {
            "V": round(float(v_score), 4),
            "O": round(float(o_score), 4),
            "L": round(float(l_score), 4),
            "H": round(float(h_score), 4),
            "R": round(float(r_score), 4),
        }


class LearnedEvidenceFusion:
    """
    Learned candidate score fusion engine.

    Fits weights w_v, w_o, w_l, w_h, w_r using scikit-learn Ridge or LogisticRegression.
    Falls back to normalized transparent heuristic defaults when weights are uncalibrated.
    """

    HEURISTIC_WEIGHTS = {
        "V": 0.20,
        "O": 0.25,
        "L": 0.15,
        "H": 0.15,
        "R": 0.25,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or dict(self.HEURISTIC_WEIGHTS)
        self.is_learned = False
        self.extractor = MultimodalFeatureExtractor()
        self.model = None

    def fit(self, X_features: List[Dict[str, float]], y_labels: List[int | float]) -> None:
        """
        Fit evidence weights from training records.
        """
        from sklearn.linear_model import Ridge

        X = np.array([[f["V"], f["O"], f["L"], f["H"], f["R"]] for f in X_features])
        y = np.array(y_labels, dtype=np.float64)

        if len(X) < 5:
            log.warning("Insufficient samples (<5) to fit learned fusion model. Retaining heuristic weights.")
            return

        clf = Ridge(alpha=1.0, positive=True)
        clf.fit(X, y)
        coefs = clf.coef_
        total = np.sum(coefs)
        if total > 0:
            norm_coefs = coefs / total
            self.weights = {
                "V": float(round(norm_coefs[0], 4)),
                "O": float(round(norm_coefs[1], 4)),
                "L": float(round(norm_coefs[2], 4)),
                "H": float(round(norm_coefs[3], 4)),
                "R": float(round(norm_coefs[4], 4)),
            }
            self.is_learned = True
            log.info("Learned evidence fusion fitted weights: %s", self.weights)

    def score_candidate(
        self,
        candidate: ReconstructionCandidate,
        ocr_confidence: float,
        retrieved_records: List[Dict],
        metadata: Optional[Any] = None,
        damage_regions: Optional[List[Dict]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute fused score S(c) = w_V*V + w_O*O + w_L*L + w_H*H + w_R*R.
        """
        features = self.extractor.extract_features(
            candidate=candidate,
            ocr_confidence=ocr_confidence,
            retrieved_records=retrieved_records,
            metadata=metadata,
            damage_regions=damage_regions,
        )

        fused_score = sum(features[dim] * self.weights.get(dim, 0.0) for dim in ["V", "O", "L", "H", "R"])
        return round(float(fused_score), 4), features
