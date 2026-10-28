"""
tests/test_fusion.py
--------------------
Tests for Multimodal Evidence Fusion (V + O + L + H + R).
"""

import unittest
from utils import ReconstructionCandidate, Provenance
from src.candidate_ranking.fusion import MultimodalFeatureExtractor, LearnedEvidenceFusion


class TestMultimodalFusion(unittest.TestCase):

    def setUp(self):
        self.extractor = MultimodalFeatureExtractor()
        self.fusion = LearnedEvidenceFusion()

    def test_feature_extraction_dimensions(self):
        """Must extract exactly the 5 PPT dimensions [V, O, L, H, R] in [0, 1]."""
        cand = ReconstructionCandidate(
            candidate_id="CAND-01",
            candidate_text="ஸ்வஸ்தி ஸ்ரீ",
            provenance=Provenance.OBSERVED,
        )
        features = self.extractor.extract_features(
            candidate=cand,
            ocr_confidence=85.0,
            retrieved_records=[],
            metadata=None,
        )
        self.assertIn("V", features)
        self.assertIn("O", features)
        self.assertIn("L", features)
        self.assertIn("H", features)
        self.assertIn("R", features)
        for dim, val in features.items():
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_unknown_candidate_visual_score_zero(self):
        """Explicit UNKNOWN gap marker must receive V=0 and O=0."""
        cand = ReconstructionCandidate(
            candidate_id="CAND-UNK-01",
            candidate_text="[UNKNOWN]",
            provenance=Provenance.UNKNOWN,
        )
        features = self.extractor.extract_features(
            candidate=cand,
            ocr_confidence=80.0,
            retrieved_records=[],
        )
        self.assertEqual(features["V"], 0.0)
        self.assertEqual(features["O"], 0.0)

    def test_score_fusion_sum_matches_weights(self):
        """Fused score must equal weighted sum of individual dimensions."""
        cand = ReconstructionCandidate(
            candidate_id="CAND-01",
            candidate_text="ஸ்வஸ்தி ஸ்ரீ கோப்பரகேசரி",
            provenance=Provenance.CORPUS,
            corpus_evidence="TAML-001",
        )
        score, feats = self.fusion.score_candidate(
            candidate=cand,
            ocr_confidence=70.0,
            retrieved_records=[{"id": "TAML-001", "score": 0.85}],
        )
        w = self.fusion.weights
        expected = sum(feats[k] * w[k] for k in ["V", "O", "L", "H", "R"])
        self.assertAlmostEqual(score, round(expected, 4), places=3)


if __name__ == "__main__":
    unittest.main()
