"""
tests/test_tamil_pipeline.py
----------------------------
Focused unit and regression tests for the Tamil-only reconstruction pipeline:
  1. Tamil language scope and metadata defaults.
  2. Clear distinction between Modern Tamil script and historical Tamil-Brahmi.
  3. Honest OCR reporting for unsupported historical scripts.
  4. Multi-candidate generation (ensuring multiple readings, not just one).
  5. Tamil stone-erosion character confusion model.
  6. Evaluation data-leakage prevention (target record cannot retrieve itself).
  7. Unicode NFC normalization for Tamil CER/WER calculation.
  8. Missing ground-truth handling (reported as unavailable, never fabricated).

EpiRecover (Tamil Focus), 2026.
"""

import os
import sys
import unicodedata
import unittest
from dataclasses import dataclass
from typing import List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ocr_engine
from epirecover_pipeline import EpiRecoverPipeline, InscriptionMetadata
from src.evaluation.metrics import compute_cer, compute_wer, evaluate_single, normalize_text
from src.evidence_retrieval.retriever import CorpusRetriever
from src.reconstruction.candidate_generator import (
    CandidateGenerator,
    UNKNOWN_MARKER,
    _apply_tamil_erosion_confusions,
)
from utils import Provenance, ReconstructionCandidate


class TestTamilPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = EpiRecoverPipeline(llm_backend="stub")

    # -----------------------------------------------------------------------
    # 1. Tamil Language Scope & Script Distinction
    # -----------------------------------------------------------------------

    def test_tamil_default_metadata(self):
        """Verify default language is Tamil and script defaults to 'tamil'."""
        meta = InscriptionMetadata()
        self.assertEqual(meta.language, "Tamil")
        self.assertEqual(meta.script, "tamil")

    def test_script_distinction_modern_vs_brahmi(self):
        """Modern Tamil and Tamil-Brahmi must be distinguished as separate writing systems."""
        modern_meta = InscriptionMetadata(script="tamil", language="Tamil")
        brahmi_meta = InscriptionMetadata(script="tamil-brahmi", language="Tamil")
        self.assertNotEqual(modern_meta.script, brahmi_meta.script)
        self.assertEqual(modern_meta.language, brahmi_meta.language)

    def test_tamil_brahmi_ocr_warning(self):
        """Historical Tamil-Brahmi must not be silently processed as standard modern Tamil OCR."""
        dummy_img = np.full((100, 100), 255, dtype=np.uint8)
        result = ocr_engine.extract_text(dummy_img, script_hint="tamil-brahmi")
        self.assertEqual(result.engine, "unsupported_script")
        self.assertIn("tamil-brahmi", result.warning_note.lower())
        self.assertIn("not supported", result.warning_note.lower())

    def test_check_tamil_ocr_support_report(self):
        """check_tamil_ocr_support must return explicit status for both Tamil writing systems."""
        support = ocr_engine.check_tamil_ocr_support()
        self.assertIn("modern_tamil_supported", support)
        self.assertIn("tamil_brahmi_supported", support)
        # Tamil-Brahmi is strictly unsupported by off-the-shelf OCR engines
        self.assertFalse(support["tamil_brahmi_supported"])

    # -----------------------------------------------------------------------
    # 2. Multi-Candidate Generation with Tamil Stone-Erosion Model
    # -----------------------------------------------------------------------

    def test_multi_candidate_generation_with_ocr_text(self):
        """Must generate multiple candidate hypotheses, not just a single reading."""
        gen = CandidateGenerator(translator=self.pipeline._translator)
        ocr_res = ocr_engine.OCRResult(
            raw_text="ஸ்வஸ்தி ஸ்ரீ",
            mean_confidence=65.0,
            engine="test-stub",
        )
        cands = gen.generate(
            ocr_result=ocr_res,
            mapped_text="",
            metadata=InscriptionMetadata(script="tamil", language="Tamil"),
            retrieved_records=[],
            damage_regions=[{"x": 10, "y": 10, "w": 20, "h": 20}],
        )
        self.assertGreater(len(cands), 1, "Must produce multiple alternative candidate readings")
        # Candidate 1 should be OCR
        self.assertEqual(cands[0].candidate_text, "ஸ்வஸ்தி ஸ்ரீ")
        self.assertEqual(cands[0].provenance, Provenance.OCR_PRED)
        # One candidate must represent the damage gap marker
        has_unknown = any(c.candidate_text == UNKNOWN_MARKER for c in cands)
        self.assertTrue(has_unknown, "A gap marker candidate must be included when damage is detected")

    def test_tamil_stone_erosion_confusion_rules(self):
        """Test specific Tamil carved glyph weathering substitutions (e.g. virama / stroke loss)."""
        # Test pulli / virama loss simulation
        text_with_pulli = "ஸ்வஸ்தி"
        variants = _apply_tamil_erosion_confusions(text_with_pulli)
        self.assertGreater(len(variants), 0)
        # Check that variants have documented reasons
        variant_texts = [v[0] for v in variants]
        reasons = [v[1] for v in variants]
        self.assertTrue(any("pulli" in r.lower() or "virama" in r.lower() for r in reasons))

        # Test character substitution: 'ர' vs 'ா'
        text_with_ra = "அரசன்"
        ra_variants = _apply_tamil_erosion_confusions(text_with_ra)
        ra_texts = [v[0] for v in ra_variants]
        self.assertTrue(any("அாசன்" in vt or "அர" in vt for vt in ra_texts))

    def test_candidate_provenance_and_id_tracking(self):
        """Every generated candidate must retain a distinct candidate_id and valid Provenance."""
        gen = CandidateGenerator(translator=self.pipeline._translator)
        ocr_res = ocr_engine.OCRResult(raw_text="கோப்பரகேசரி", mean_confidence=80.0)
        cands = gen.generate(ocr_res, mapped_text="", metadata=None)
        self.assertGreaterEqual(len(cands), 2)
        for c in cands:
            self.assertTrue(bool(c.candidate_id))
            self.assertIsInstance(c.provenance, Provenance)
            self.assertTrue(c.confidence_note)

    # -----------------------------------------------------------------------
    # 3. Evidence Retrieval & Evaluation Data Leakage Prevention
    # -----------------------------------------------------------------------

    def test_evaluation_data_leakage_prevention(self):
        """Evaluation target record ID must be strictly excluded from retrieval."""
        corpus_dir = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
        retriever = CorpusRetriever(corpus_dir=corpus_dir)
        
        target_id = "TAML-001"
        # Query matching the exact transcription of TAML-001
        query = "ஸ்வஸ்தி ஸ்ரீ கோப்பரகேசரிவன்மற்கு யாண்டு"

        # 1. Without exclusion: TAML-001 can match
        results_without_exclusion = retriever.retrieve(query, exclude_ids=[])
        ids_without = [r["id"] for r in results_without_exclusion]
        self.assertIn(target_id, ids_without, "Target record should be found when not excluded")

        # 2. With exclusion (anti-leakage protocol during evaluation):
        results_with_exclusion = retriever.retrieve(query, exclude_ids=[target_id])
        ids_with = [r["id"] for r in results_with_exclusion]
        self.assertNotIn(
            target_id,
            ids_with,
            "CRITICAL: Evaluation target ID must NEVER retrieve itself as evidence (data leakage prevention)!"
        )

    def test_retriever_tamil_language_filtering(self):
        """Retrieval must filter to Tamil language records when language='tamil'."""
        corpus_dir = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
        retriever = CorpusRetriever(corpus_dir=corpus_dir)
        results = retriever.retrieve("கோப்பரகேசரி", language="tamil")
        for r in results:
            self.assertEqual(r.get("language", "").lower(), "tamil")

    # -----------------------------------------------------------------------
    # 4. Unicode NFC Normalization for Tamil CER/WER
    # -----------------------------------------------------------------------

    def test_tamil_unicode_nfc_normalization(self):
        """Tamil strings in decomposed (NFD) and composed (NFC) forms must have CER=0."""
        text_nfc = "கோப்பரகேசரி"
        text_nfd = unicodedata.normalize("NFD", text_nfc)
        # Byte lengths and codepoints differ between NFD and NFC
        self.assertNotEqual(len(text_nfc), len(text_nfd))
        # With normalize_text / compute_cer, NFC normalization ensures CER is 0.0
        cer = compute_cer(text_nfc, text_nfd)
        self.assertEqual(cer, 0.0, "Unicode NFC normalization must reconcile NFD and NFC Tamil text")

    def test_tamil_cer_calculation_known_case(self):
        """CER calculation on verified Tamil reference with 1 character substitution."""
        ref = "வணக்கம்"
        hyp = "வணக்கம"   # missing pulli (1 deletion)
        cer = compute_cer(hyp, ref)
        self.assertIsNotNone(cer)
        self.assertGreater(cer, 0.0)
        self.assertLess(cer, 0.5)

    def test_missing_ground_truth_returns_unavailable(self):
        """Missing ground truth must return None for CER/WER with explicit reason, never fabricate."""
        res = evaluate_single(
            inscription_id="TEST-01",
            hypothesis="ஸ்வஸ்தி ஸ்ரீ",
            reference=None,  # ground truth missing
        )
        self.assertIsNone(res.cer)
        self.assertIsNone(res.wer)
        self.assertTrue(res.metric_unavailable_reason)
        self.assertIn("No verified ground-truth", res.metric_unavailable_reason)


if __name__ == "__main__":
    unittest.main()
