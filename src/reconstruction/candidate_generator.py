"""
src/reconstruction/candidate_generator.py
------------------------------------------
Generates multiple evidence-grounded candidate readings for damaged or uncertain
inscription text in EpiRecover (Tamil Focus).

Strict Non-Generative Methodology (PPT Slides 11, 12):
  - NO generative LLMs (no GPT, Gemini, Groq, or prompt-based text fabrication).
  - Candidates are strictly generated from:
      1. Visible OCR output (OBSERVED for high confidence, OCR_PRED for lower confidence)
      2. Grounded historical epigraphic corpus matches (CORPUS_RETRIEVED)
      3. Documented Tamil stone-erosion character confusion rules (MODEL_INFERRED)
      4. Explicit gap and damage markers (UNKNOWN)
  - Every candidate maintains full provenance, confidence notes, and affected spans.
  - Never fabricates unobserved characters or historical facts.

EpiRecover (Tamil Focus), 2026.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from utils import Provenance, ReconstructionCandidate

log = logging.getLogger(__name__)

# Marker for unresolvable damaged spans
UNKNOWN_MARKER = "[UNKNOWN]"
INFERRED_OPEN = "\u27e8"   # ⟨
INFERRED_CLOSE = "\u27e9"  # ⟩

# Common character confusion pairs in carved Tamil lithic inscriptions:
# Due to chisel weathering, lichen, or rock spalling:
# 1. Pulli (virama) erosion: Pure consonants lose their dot, appearing as base consonants.
# 2. Visual stroke similarity: Curves or ascenders eroding.
TAMIL_STONE_CONFUSIONS: List[Tuple[str, str, str]] = [
    # (Char A, Char B, Reason)
    ("\u0bb0", "\u0bbe", "Tamil letter 'ra' vs vowel sign 'aa' (single vertical stroke confusion)"),
    ("\u0bbe", "\u0bb0", "Tamil vowel sign 'aa' vs letter 'ra' (single vertical stroke confusion)"),
    ("\u0bb2", "\u0bb5", "Tamil 'la' vs 'va' (bottom stroke weathering)"),
    ("\u0bb5", "\u0bb2", "Tamil 'va' vs 'la' (bottom stroke weathering)"),
    ("\u0b95", "\u0b9a", "Tamil 'ka' vs 'ca' (top crossbar / curve erosion)"),
    ("\u0b9a", "\u0b95", "Tamil 'ca' vs 'ka' (top crossbar / curve erosion)"),
    ("\u0ba3", "\u0ba9", "Tamil retroflex '\u1e47a' vs alveolar 'na' (loop erosion)"),
    ("\u0ba9", "\u0ba3", "Tamil alveolar 'na' vs retroflex '\u1e47a' (loop erosion)"),
    ("\u0da3", "\u0baa", "Tamil '\u1e6da' vs 'pa' (vertical stroke weathering)"),
    ("\u0baa", "\u0da3", "Tamil 'pa' vs '\u1e6da' (vertical stroke weathering)"),
    ("\u0baf", "\u0baa", "Tamil 'ya' vs 'pa' (medial cleft erosion)"),
]

# Tamil Unicode block: U+0B80 to U+0BFF
_TAMIL_BLOCK_START = "\u0b80"
_TAMIL_BLOCK_END = "\u0bff"
TAMIL_SCRIPT_MIN_RATIO = 0.30  # At least 30% of non-space chars must be Tamil Unicode


def _is_tamil_text(text: str) -> bool:
    """
    Return True if at least TAMIL_SCRIPT_MIN_RATIO of non-space characters
    fall within the Tamil Unicode block (U+0B80-U+0BFF).

    This is used to detect Latin/ASCII noise produced by a broken Tamil OCR
    model and prevent it from propagating as a candidate reading.
    """
    if not text or text == UNKNOWN_MARKER:
        return False
    stripped = text.replace(" ", "")
    if not stripped:
        return False
    tamil_count = sum(1 for c in stripped if _TAMIL_BLOCK_START <= c <= _TAMIL_BLOCK_END)
    return (tamil_count / len(stripped)) >= TAMIL_SCRIPT_MIN_RATIO


def _apply_tamil_erosion_confusions(text: str) -> List[Tuple[str, str]]:
    """
    Generate alternative readings using documented Tamil lithic confusion rules.
    Returns list of (variant_text, description_of_change).
    """
    variants: List[Tuple[str, str]] = []
    if not text or text == UNKNOWN_MARKER:
        return variants

    # Rule 1: Glyph substitutions
    for orig_char, subst_char, reason in TAMIL_STONE_CONFUSIONS:
        if orig_char in text:
            # Replace first occurrence to generate a minimal alternative
            alt_text = text.replace(orig_char, subst_char, 1)
            if alt_text != text:
                variants.append((alt_text, f"Substituted '{orig_char}' -> '{subst_char}': {reason}"))

    # Rule 2: Virama (Pulli) loss / restoration (U+0BCD)
    pulli = "\u0bcd"
    if pulli in text:
        no_pulli = text.replace(pulli, "", 1)
        variants.append((no_pulli, "Simulated pulli (virama) erosion common in stone inscriptions"))
    else:
        # Check if ending with a consonant that typically takes pulli in Tamil
        for stop_char in ["க", "ச", "ட", "த", "ப", "ற", "ர", "ல", "ள", "ன", "ம", "ந"]:
            if stop_char in text:
                with_pulli = text.replace(stop_char, stop_char + pulli, 1)
                variants.append((with_pulli, f"Restored hypothetical weathered pulli on '{stop_char}'"))
                break

    return variants


def _extract_corpus_candidates(
    retrieved_records: List[Dict],
    ocr_text: str,
) -> List[ReconstructionCandidate]:
    """
    Build corpus-backed candidates from retrieved inscription records.
    """
    candidates: List[ReconstructionCandidate] = []
    for i, rec in enumerate(retrieved_records):
        transcription = rec.get("transcription") or rec.get("text") or ""
        record_id = rec.get("id") or rec.get("record_id") or "unknown"
        score = float(rec.get("score", 0.0))
        if not transcription:
            continue
        candidates.append(ReconstructionCandidate(
            candidate_id=f"CORPUS-{record_id}",
            candidate_text=transcription,
            provenance=Provenance.CORPUS,
            corpus_evidence=record_id,
            ocr_evidence=ocr_text,
            confidence=min(score, 0.95),
            confidence_note=f"Corpus retrieval match score={score:.3f}; not a calibrated probability.",
            uncertainty_reasons=[] if score > 0.7 else ["Weak corpus similarity score"],
        ))
    return candidates


def _make_unknown_candidate(
    cid: str = "UNK-01",
    span: Optional[str] = None,
    reason: str = "",
) -> ReconstructionCandidate:
    """Return a clearly-marked 'insufficient evidence' candidate."""
    return ReconstructionCandidate(
        candidate_id=cid,
        candidate_text=UNKNOWN_MARKER,
        provenance=Provenance.UNKNOWN,
        affected_span=span,
        confidence=0.0,
        confidence_note="Insufficient visual, OCR, or corpus evidence to resolve reading.",
        uncertainty_reasons=[reason or "No sufficient evidence available; gap preserved."],
    )


class CandidateGenerator:
    """
    Generates structured, traceable multiple candidate readings for an inscription.

    Guarantees:
      - Strictly non-generative: no generative LLMs.
      - Always returns multiple candidates when text or damage is present.
      - Never presents model-inferred text as observed text.
      - Maintains provenance tags and affected spans.
    """

    MIN_CORPUS_SCORE = 0.35
    MIN_OCR_CONF = 15.0

    def __init__(
        self,
        translator=None,
        cfg: Optional[Dict] = None,
        max_candidates: int = 5,
    ) -> None:
        self._max = max_candidates
        self._cfg = cfg or {}

    def generate(
        self,
        ocr_result,
        mapped_text: str = "",
        metadata=None,
        retrieved_records: Optional[List[Dict]] = None,
        damage_regions: Optional[List[Dict]] = None,
    ) -> List[ReconstructionCandidate]:
        """
        Generate candidate readings using multimodal evidence.
        """
        candidates: List[ReconstructionCandidate] = []
        raw_text = (ocr_result.raw_text or "").strip() if ocr_result else ""
        ocr_conf = ocr_result.mean_confidence if ocr_result else 0.0

        # ---------------------------------------------------------
        # 1. Primary OCR Reading (OBSERVED or OCR_PRED)
        # ---------------------------------------------------------
        if raw_text:
            # Validate Tamil script content before accepting as a candidate.
            # If OCR is in English-fallback mode (Tamil model broken/missing),
            # the raw_text will be Latin/ASCII noise and must NOT be promoted
            # as a valid Tamil inscription reading.
            ocr_engine_name = (ocr_result.engine or "") if ocr_result else ""
            is_fallback_engine = "english-fallback" in ocr_engine_name or "none-tamil" in ocr_engine_name
            text_has_tamil = _is_tamil_text(raw_text)

            if is_fallback_engine and not text_has_tamil:
                # Non-Tamil text from a broken/English-fallback OCR engine.
                # Tag as script noise — do NOT create a reading candidate from this.
                log.warning(
                    "CandidateGenerator: OCR engine '%s' produced non-Tamil text '%s...'; "
                    "treating as script noise — not generating a reading candidate.",
                    ocr_engine_name, raw_text[:40],
                )
                # We still record the OCR noise for transparency in the report,
                # but the candidate text is [UNKNOWN].
                candidates.append(ReconstructionCandidate(
                    candidate_id="CAND-NOISE-01",
                    candidate_text=UNKNOWN_MARKER,
                    provenance=Provenance.UNKNOWN,
                    ocr_evidence=raw_text,
                    confidence=0.0,
                    confidence_note=(
                        f"OCR engine '{ocr_engine_name}' produced Latin/ASCII text '{raw_text[:60]}' "
                        "which is not Tamil Unicode. This is script noise from a broken Tamil OCR "
                        "model. No reading candidate generated from this output."
                    ),
                    uncertainty_reasons=[
                        f"Script mismatch: OCR output '{raw_text[:30]}' contains 0% Tamil Unicode characters.",
                        "EasyOCR Tamil model checkpoint is broken (state_dict size mismatch). "
                        "Install Tesseract with tam.traineddata or downgrade easyocr to 1.6.x.",
                    ],
                ))
            elif raw_text and raw_text != UNKNOWN_MARKER:
                # Either: text has valid Tamil chars, OR engine is a proper Tamil engine.
                prov = Provenance.OBSERVED if ocr_conf >= 75.0 else Provenance.OCR_PRED
                candidates.append(ReconstructionCandidate(
                    candidate_id="CAND-OCR-01",
                    candidate_text=raw_text,
                    provenance=prov,
                    ocr_evidence=raw_text,
                    confidence=round(max(0.1, min(1.0, ocr_conf / 100.0)), 3),
                    confidence_note=f"Raw OCR output from {ocr_result.engine} (mean conf={ocr_conf:.1f}%).",
                    uncertainty_reasons=["Low OCR confidence"] if ocr_conf < 50.0 else [],
                ))

        # ---------------------------------------------------------
        # 2. Corpus-retrieved Candidates (CORPUS_RETRIEVED)
        # ---------------------------------------------------------
        if retrieved_records:
            # Metadata-fallback records are preserved for historical context in the report,
            # but MUST NOT be promoted as candidate readings when no textual match exists.
            valid_corpus = [
                r for r in retrieved_records
                if r.get("retrieval_method") != "metadata_fallback"
                and float(r.get("score", 0)) >= self.MIN_CORPUS_SCORE
            ]
            corpus_cands = _extract_corpus_candidates(valid_corpus, raw_text)
            for i, cc in enumerate(corpus_cands):
                cc.candidate_id = f"CAND-CORP-{i+1:02d}"
            candidates.extend(corpus_cands)

        # ---------------------------------------------------------
        # 3. Tamil Stone-Erosion Confusion Model (MODEL_INFERRED)
        # ---------------------------------------------------------
        # Only apply erosion confusions when the base text actually contains
        # Tamil Unicode characters — applying them to Latin/ASCII noise
        # produces meaningless variants.
        base_for_variants = raw_text or mapped_text
        if base_for_variants and base_for_variants != UNKNOWN_MARKER and _is_tamil_text(base_for_variants):
            erosion_variants = _apply_tamil_erosion_confusions(base_for_variants)
            for i, (var_text, reason) in enumerate(erosion_variants[:2]):
                candidates.append(ReconstructionCandidate(
                    candidate_id=f"CAND-INF-{i+1:02d}",
                    candidate_text=var_text,
                    provenance=Provenance.MODEL_INFERRED,
                    ocr_evidence=raw_text,
                    confidence=round(max(0.05, min(0.70, (ocr_conf / 100.0) * 0.85)), 3),
                    confidence_note="Plausible Tamil stone-erosion variant; not an independently verified reading.",
                    uncertainty_reasons=[reason],
                ))

        # ---------------------------------------------------------
        # 4. Gap / Damage Markers (UNKNOWN)
        # ---------------------------------------------------------
        damage_count = len(damage_regions or [])
        if not candidates or damage_count > 0 or ocr_conf < 40.0:
            candidates.append(_make_unknown_candidate(
                cid=f"CAND-UNK-{len(candidates)+1:02d}",
                reason=f"{damage_count} potential erosion/damage regions detected; characters unverified."
            ))

        # ---------------------------------------------------------
        # Fallback guarantee: Never return an empty list or only single guess
        # ---------------------------------------------------------
        if not candidates:
            candidates = [
                _make_unknown_candidate("CAND-UNK-01", reason="No readable characters extracted by OCR."),
                _make_unknown_candidate("CAND-UNK-02", reason="Awaiting manual epigraphist transcription."),
            ]
        elif len(candidates) == 1:
            # Add explicit gap alternative so user always has multiple readings
            candidates.append(_make_unknown_candidate(
                cid="CAND-UNK-02",
                reason="Unresolved characters in weathered regions."
            ))

        # Deduplicate candidates by text, but preserve distinct UNKNOWN candidates by ID
        seen: Dict[str, ReconstructionCandidate] = {}
        for c in candidates:
            if c.candidate_text == UNKNOWN_MARKER:
                key = c.candidate_id
            else:
                key = c.candidate_text.strip()
            if key not in seen or c.confidence > seen[key].confidence:
                seen[key] = c

        unique_cands = list(seen.values())
        unique_cands.sort(key=lambda c: c.confidence, reverse=True)
        return unique_cands[: self._max]
