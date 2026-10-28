"""
src/evaluation/metrics.py
--------------------------
Reproducible evaluation metrics for EpiRecover.

Metrics implemented:
  - Character Error Rate (CER): edit distance / length of reference
  - Word Error Rate (WER): word-level edit distance / reference word count
  - Candidate Top-1 Accuracy: fraction of Top-1 candidates matching ground truth
  - Candidate Top-5 Accuracy: fraction where ground truth is present in Top-5 candidates
  - Expected Calibration Error (ECE) & Brier Score
  - Risk-Coverage curve computation for abstention
  - Aggregate reporting: per-script, per-damage-level summaries

Dependencies:
  - jiwer >= 3.0 (for CER/WER) — install via pip install jiwer
  - Falls back to a pure-Python CER implementation if jiwer is unavailable.

Research integrity:
  - Metrics are only computed when a verified ground-truth transcription
    is available in data/annotations/.
  - If no reference is available, the metric is reported as 'unavailable'
    with an explanation of what data is needed.
  - Results are never fabricated or estimated without actual measurement.

EpiRecover, 2026.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Optional jiwer import
_HAS_JIWER = False
try:
    from jiwer import cer as _jiwer_cer, wer as _jiwer_wer
    _HAS_JIWER = True
except ImportError:
    log.info("jiwer not installed; using pure-Python CER fallback.")


# ---------------------------------------------------------------------------
# Pure-Python CER fallback (no external dependency)
# ---------------------------------------------------------------------------

def _levenshtein_chars(a: str, b: str) -> int:
    """Character-level Levenshtein distance."""
    if a == b:
        return 0
    rows, cols = len(a) + 1, len(b) + 1
    prev = list(range(cols))
    for i in range(1, rows):
        curr = [i] + [0] * (cols - 1)
        for j in range(1, cols):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
        prev = curr
    return prev[-1]


def normalize_text(text: Optional[str]) -> str:
    """
    Apply Unicode NFC normalization to ensure consistent grapheme representations.
    Essential for Tamil and Indic scripts where combining vowel markers and pulli
    can differ in byte decomposition.
    """
    if not text:
        return ""
    # NFC = Canonical Decomposition followed by Canonical Composition
    return unicodedata.normalize("NFC", text.strip())


def compute_cer(hypothesis: str, reference: str) -> Optional[float]:
    """
    Compute Character Error Rate: edit_distance(hyp, ref) / len(ref).

    Applies Unicode NFC normalization to both hypothesis and reference.
    Returns None if reference is empty (metric is unavailable).
    Normalised to [0, ∞); values > 1 indicate more errors than reference characters.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm:
        return None
    if _HAS_JIWER:
        try:
            return float(_jiwer_cer(ref_norm, hyp_norm))
        except Exception as e:
            log.warning("jiwer CER failed (%s); falling back to pure-Python.", e)
    return _levenshtein_chars(hyp_norm, ref_norm) / len(ref_norm)


def compute_wer(hypothesis: str, reference: str) -> Optional[float]:
    """
    Compute Word Error Rate: word-level edit distance / reference word count.

    Applies Unicode NFC normalization before tokenization.
    Returns None if reference is empty.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm:
        return None
    if _HAS_JIWER:
        try:
            return float(_jiwer_wer(ref_norm, hyp_norm))
        except Exception as e:
            log.warning("jiwer WER failed (%s); falling back to None.", e)
    # Pure-Python word-level Levenshtein
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    if not ref_words:
        return None
    rows, cols = len(hyp_words)+1, len(ref_words)+1
    prev = list(range(cols))
    for i in range(1, rows):
        curr = [i] + [0]*(cols-1)
        for j in range(1, cols):
            cost = 0 if hyp_words[i-1] == ref_words[j-1] else 1
            curr[j] = min(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
        prev = curr
    return prev[-1] / len(ref_words)


def compute_top_k_accuracy(
    candidates_list: List[List[str]],
    references: List[str],
    k: int = 1,
) -> float:
    """
    Compute Top-K candidate accuracy: fraction of instances where
    normalized reference matches at least one candidate in candidates[:k].
    """
    if not candidates_list or not references or len(candidates_list) != len(references):
        return 0.0

    matches = 0
    total = 0
    for cands, ref in zip(candidates_list, references):
        ref_norm = normalize_text(ref)
        if not ref_norm:
            continue
        total += 1
        cand_slice = [normalize_text(c) for c in cands[:k]]
        if ref_norm in cand_slice:
            matches += 1

    return round(matches / max(1, total), 4)


# ---------------------------------------------------------------------------
# Per-inscription result container
# ---------------------------------------------------------------------------

@dataclass
class InscriptionResult:
    """Evaluation result for a single inscription."""
    inscription_id: str
    script: Optional[str] = None
    damage_level: Optional[str] = None
    hypothesis: str = ""
    reference: str = ""
    candidate_list: List[str] = field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False
    cer: Optional[float] = None
    wer: Optional[float] = None
    ocr_cer: Optional[float] = None    # CER of raw OCR (baseline)
    ocr_wer: Optional[float] = None
    candidate_rank: Optional[int] = None   # rank of correct candidate (if known)
    notes: str = ""
    metric_unavailable_reason: str = ""

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Aggregate evaluation results across multiple inscriptions."""
    total: int = 0
    with_reference: int = 0
    mean_cer: Optional[float] = None
    mean_wer: Optional[float] = None
    ocr_baseline_cer: Optional[float] = None
    top_1_accuracy: Optional[float] = None
    top_5_accuracy: Optional[float] = None
    coverage_rate: Optional[float] = None
    ece: Optional[float] = None
    brier_score: Optional[float] = None
    by_script: Dict[str, Dict] = field(default_factory=dict)
    by_damage_level: Dict[str, Dict] = field(default_factory=dict)
    per_inscription: List[InscriptionResult] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["per_inscription"] = [r.as_dict() for r in self.per_inscription]
        return d


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def evaluate_single(
    inscription_id: str,
    hypothesis: str,
    reference: Optional[str],
    ocr_raw: Optional[str] = None,
    candidate_list: Optional[List[str]] = None,
    confidence: float = 0.0,
    abstained: bool = False,
    script: Optional[str] = None,
    damage_level: Optional[str] = None,
    candidate_rank: Optional[int] = None,
) -> InscriptionResult:
    """
    Compute metrics for a single inscription.

    If `reference` is None or empty, all metrics are set to None and
    metric_unavailable_reason is filled in explaining what is needed.
    """
    c_list = candidate_list or []
    result = InscriptionResult(
        inscription_id=inscription_id,
        script=script,
        damage_level=damage_level,
        hypothesis=hypothesis,
        reference=reference or "",
        candidate_list=c_list,
        confidence=confidence,
        abstained=abstained,
        candidate_rank=candidate_rank,
    )

    if not reference:
        result.metric_unavailable_reason = (
            "No verified ground-truth transcription available for this inscription. "
            "To compute CER/WER, provide a reference transcription in "
            "data/annotations/<inscription_id>.txt"
        )
        return result

    result.cer = compute_cer(hypothesis, reference)
    result.wer = compute_wer(hypothesis, reference)

    if ocr_raw:
        result.ocr_cer = compute_cer(ocr_raw, reference)
        result.ocr_wer = compute_wer(ocr_raw, reference)

    return result


def aggregate_results(results: List[InscriptionResult]) -> EvaluationReport:
    """
    Aggregate per-inscription results into an EvaluationReport.
    Computes Mean CER, Mean WER, Top-1, Top-5, and Calibration metrics.
    """
    report = EvaluationReport(
        total=len(results),
        per_inscription=results,
    )

    scored = [r for r in results if r.cer is not None]
    report.with_reference = len(scored)

    if scored:
        report.mean_cer = round(sum(r.cer for r in scored) / len(scored), 4)
        wer_scored = [r for r in scored if r.wer is not None]
        if wer_scored:
            report.mean_wer = round(sum(r.wer for r in wer_scored) / len(wer_scored), 4)
        ocr_scored = [r for r in scored if r.ocr_cer is not None]
        if ocr_scored:
            report.ocr_baseline_cer = round(
                sum(r.ocr_cer for r in ocr_scored) / len(ocr_scored), 4
            )

        # Top-1 and Top-5 accuracy
        cands_list = [r.candidate_list if r.candidate_list else [r.hypothesis] for r in scored]
        refs = [r.reference for r in scored]
        report.top_1_accuracy = compute_top_k_accuracy(cands_list, refs, k=1)
        report.top_5_accuracy = compute_top_k_accuracy(cands_list, refs, k=5)

        # Coverage rate (non-abstained fraction)
        non_abstained = [r for r in scored if not r.abstained]
        report.coverage_rate = round(len(non_abstained) / len(scored), 4)

        # ECE and Brier score
        try:
            from src.candidate_ranking.calibration import compute_ece, compute_brier_score
            confs = [r.confidence for r in scored]
            accs = [1 if (r.cer is not None and r.cer == 0.0) else 0 for r in scored]
            report.ece = compute_ece(confs, accs)
            report.brier_score = compute_brier_score(confs, accs)
        except Exception as e:
            log.warning("Calibration metrics computation skipped: %s", e)

    # Group by script
    for r in scored:
        s = r.script or "unknown"
        if s not in report.by_script:
            report.by_script[s] = {"count": 0, "sum_cer": 0.0}
        report.by_script[s]["count"] += 1
        report.by_script[s]["sum_cer"] += r.cer or 0.0
    for s, d in report.by_script.items():
        d["mean_cer"] = round(d["sum_cer"] / d["count"], 4) if d["count"] else None

    # Group by damage level
    for r in scored:
        dl = r.damage_level or "unknown"
        if dl not in report.by_damage_level:
            report.by_damage_level[dl] = {"count": 0, "sum_cer": 0.0}
        report.by_damage_level[dl]["count"] += 1
        report.by_damage_level[dl]["sum_cer"] += r.cer or 0.0
    for dl, d in report.by_damage_level.items():
        d["mean_cer"] = round(d["sum_cer"] / d["count"], 4) if d["count"] else None

    if not scored:
        report.notes = (
            "No inscriptions had verified ground-truth transcriptions. "
            "Add transcriptions to data/annotations/ to compute metrics."
        )

    return report
