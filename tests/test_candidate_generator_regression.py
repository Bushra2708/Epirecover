"""
tests/test_candidate_generator_regression.py
---------------------------------------------
Regression tests for the Tamil script validation fix in CandidateGenerator.

Tests verify:
  1. Non-Tamil ASCII noise ('44 4') is NOT promoted as a valid reading candidate
     when the OCR engine is in English-fallback mode.
  2. Actual Tamil Unicode text IS accepted and produces CAND-OCR-01.
  3. Tamil stone-erosion confusions are NOT applied to Latin/ASCII noise.
  4. [UNKNOWN] gap candidate is always returned even when OCR produces noise.
  5. Metadata fallback in CorpusRetriever works when query is non-Tamil.
  6. Visual score is penalised for non-Tamil candidate text in fusion.

EpiRecover, 2026.
"""

from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import Provenance, ReconstructionCandidate
from src.reconstruction.candidate_generator import (
    CandidateGenerator,
    _is_tamil_text,
    UNKNOWN_MARKER,
)


class _FakeOCRResult:
    def __init__(self, raw_text, mean_confidence, engine="easyocr"):
        self.raw_text = raw_text
        self.mean_confidence = mean_confidence
        self.engine = engine
        self.boxes = []
        self.language_hint = None
        self.script_hint = None
        self.warning_note = None


class TestIsTamilText:
    def test_tamil_unicode_detected(self):
        assert _is_tamil_text("ஸ்வஸ்தி ஸ்ரீ") is True

    def test_latin_ascii_not_tamil(self):
        assert _is_tamil_text("44 4") is False
        assert _is_tamil_text("DEVANAMPN A ASOKASA") is False

    def test_empty_string(self):
        assert _is_tamil_text("") is False

    def test_unknown_marker(self):
        assert _is_tamil_text("[UNKNOWN]") is False

    def test_mixed_below_threshold(self):
        assert _is_tamil_text("க abc") is False

    def test_tamil_dominant_mixed(self):
        assert _is_tamil_text("கம்ப a") is True


class TestCandidateGeneratorNoiseRejection:
    def setup_method(self):
        self.gen = CandidateGenerator()

    def test_english_fallback_ascii_noise_not_promoted(self):
        ocr = _FakeOCRResult("44 4", 33.78, "easyocr-english-fallback")
        candidates = self.gen.generate(ocr_result=ocr)
        texts = [c.candidate_text for c in candidates]
        assert "44 4" not in texts, f"OCR noise must not appear as candidate; got: {texts}"

    def test_english_fallback_produces_unknown_candidate(self):
        ocr = _FakeOCRResult("44 4", 33.78, "easyocr-english-fallback")
        candidates = self.gen.generate(ocr_result=ocr)
        unk = [c for c in candidates if c.candidate_text == UNKNOWN_MARKER]
        assert len(unk) >= 1

    def test_noise_candidate_has_warning(self):
        ocr = _FakeOCRResult("44 4", 33.78, "easyocr-english-fallback")
        candidates = self.gen.generate(ocr_result=ocr)
        noise = [c for c in candidates if c.candidate_id == "CAND-NOISE-01"]
        assert len(noise) == 1
        assert len(noise[0].uncertainty_reasons) > 0

    def test_valid_tamil_text_accepted(self):
        tamil = "ஸ்வஸ்தி ஸ்ரீ கோப்பரகேசரிவன்மற்கு"
        ocr = _FakeOCRResult(tamil, 80.0, "tesseract")
        candidates = self.gen.generate(ocr_result=ocr)
        assert "CAND-OCR-01" in [c.candidate_id for c in candidates]

    def test_erosion_confusions_not_applied_to_ascii(self):
        ocr = _FakeOCRResult("44 4", 33.78, "easyocr-english-fallback")
        candidates = self.gen.generate(ocr_result=ocr)
        inf = [c for c in candidates if c.candidate_id.startswith("CAND-INF-")]
        assert len(inf) == 0

    def test_corpus_candidates_included(self):
        ocr = _FakeOCRResult("44 4", 33.78, "easyocr-english-fallback")
        records = [{"id": "TAML-001", "transcription": "ஸ்வஸ்தி ஸ்ரீ", "score": 0.6}]
        candidates = self.gen.generate(ocr_result=ocr, retrieved_records=records)
        corp = [c for c in candidates if c.candidate_id.startswith("CAND-CORP-")]
        assert len(corp) >= 1

    def test_always_returns_at_least_two_candidates(self):
        ocr = _FakeOCRResult("", 0.0, "none-tamil-ocr-unavailable")
        candidates = self.gen.generate(ocr_result=ocr)
        assert len(candidates) >= 2


class TestMetadataFallbackRetrieval:
    def test_metadata_fallback_for_non_tamil_query(self):
        from src.evidence_retrieval.retriever import CorpusRetriever

        class FakeMeta:
            historical_period = "10th century CE (Chola)"
            region = "Thanjavur, Tamil Nadu"
            monument = "Brihadisvara Temple"
            script = "tamil"

        corpus_dir = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
        if not os.path.isdir(corpus_dir):
            pytest.skip("Corpus directory not found")

        retriever = CorpusRetriever(corpus_dir=corpus_dir)
        results = retriever.retrieve(query="44 4", language="tamil", metadata=FakeMeta(), min_score=0.01)
        assert isinstance(results, list)
        for r in results:
            assert r.get("score", 0) <= 0.5 + 1e-9
            assert "retrieval_method" in r

    def test_text_retrieval_for_tamil_query(self):
        from src.evidence_retrieval.retriever import CorpusRetriever

        corpus_dir = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
        if not os.path.isdir(corpus_dir):
            pytest.skip("Corpus directory not found")

        retriever = CorpusRetriever(corpus_dir=corpus_dir)
        results = retriever.retrieve(query="ஸ்வஸ்தி ஸ்ரீ", language="tamil", min_score=0.01)
        assert isinstance(results, list)
        for r in results:
            assert r.get("retrieval_method") == "text"

    def test_empty_query_no_metadata_returns_empty(self):
        from src.evidence_retrieval.retriever import CorpusRetriever

        corpus_dir = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
        if not os.path.isdir(corpus_dir):
            pytest.skip("Corpus directory not found")

        retriever = CorpusRetriever(corpus_dir=corpus_dir)
        assert retriever.retrieve(query="", language="tamil", metadata=None) == []


class TestFusionVisualScorePenalty:
    def test_non_tamil_ocr_pred_penalized(self):
        from src.candidate_ranking.fusion import MultimodalFeatureExtractor
        cand = ReconstructionCandidate(
            candidate_id="CAND-OCR-01", candidate_text="44 4",
            provenance=Provenance.OCR_PRED, ocr_evidence="44 4", confidence=0.34,
        )
        feats = MultimodalFeatureExtractor().extract_features(cand, 33.78, [])
        assert feats["V"] < 0.40, f"V for non-Tamil OCR_PRED should be < 0.40, got {feats['V']}"

    def test_tamil_text_ocr_pred_full_v(self):
        from src.candidate_ranking.fusion import MultimodalFeatureExtractor
        cand = ReconstructionCandidate(
            candidate_id="CAND-OCR-01", candidate_text="ஸ்வஸ்தி ஸ்ரீ",
            provenance=Provenance.OCR_PRED, confidence=0.55,
        )
        feats = MultimodalFeatureExtractor().extract_features(cand, 55.0, [])
        assert abs(feats["V"] - 0.65) < 0.01

    def test_unknown_v_zero(self):
        from src.candidate_ranking.fusion import MultimodalFeatureExtractor
        cand = ReconstructionCandidate(
            candidate_id="CAND-UNK-01", candidate_text="[UNKNOWN]",
            provenance=Provenance.UNKNOWN, confidence=0.0,
        )
        feats = MultimodalFeatureExtractor().extract_features(cand, 0.0, [])
        assert feats["V"] == 0.0

    def test_corpus_not_penalized(self):
        from src.candidate_ranking.fusion import MultimodalFeatureExtractor
        cand = ReconstructionCandidate(
            candidate_id="CAND-CORP-01", candidate_text="piyadasi laja",
            provenance=Provenance.CORPUS, corpus_evidence="SYNTH-001", confidence=0.6,
        )
        feats = MultimodalFeatureExtractor().extract_features(
            cand, 50.0, [{"id": "SYNTH-001", "score": 0.6}],
        )
        assert abs(feats["V"] - 0.40) < 0.01
