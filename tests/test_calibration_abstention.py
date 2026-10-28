"""
tests/test_calibration_abstention.py
------------------------------------
Tests for confidence calibration (temperature scaling, ECE, Brier score)
and abstention (Chow reject rule, risk-coverage curves).
"""

import unittest
import numpy as np
from utils import ReconstructionCandidate, Provenance
from src.candidate_ranking.calibration import temperature_scale, compute_ece, compute_brier_score, TemperatureScaler
from src.candidate_ranking.abstention import AbstentionEngine, compute_risk_coverage_curve, UNKNOWN_MARKER


class TestCalibrationAndAbstention(unittest.TestCase):

    def test_temperature_scaling_monotonicity(self):
        """Temperature scaling preserves ranking order."""
        scores = [0.2, 0.5, 0.8]
        calibrated = temperature_scale(scores, temperature=1.5)
        self.assertLess(calibrated[0], calibrated[1])
        self.assertLess(calibrated[1], calibrated[2])

    def test_ece_perfect_calibration(self):
        """ECE should be near 0 for perfectly calibrated predictions."""
        # 10 samples with 0.9 confidence and 90% accuracy
        confs = [0.9] * 10
        accs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  # 9/10 = 0.9
        ece = compute_ece(confs, accs, n_bins=10)
        self.assertAlmostEqual(ece, 0.0, places=2)

    def test_brier_score_bounds(self):
        """Brier score must be between 0 and 1."""
        confs = [0.8, 0.7, 0.4]
        accs = [1, 1, 0]
        bs = compute_brier_score(confs, accs)
        self.assertGreaterEqual(bs, 0.0)
        self.assertLessEqual(bs, 1.0)

    def test_abstention_on_low_confidence(self):
        """System must abstain when top candidate score is below threshold."""
        engine = AbstentionEngine(confidence_threshold=0.50)
        cand = ReconstructionCandidate(
            candidate_id="CAND-01",
            candidate_text="ஸ்வஸ்தி",
            provenance=Provenance.OCR_PRED,
            rank_score=0.30,
        )
        dec = engine.evaluate([cand])
        self.assertTrue(dec.should_abstain)
        self.assertEqual(dec.effective_text, UNKNOWN_MARKER)

    def test_abstention_on_contradictory_margin(self):
        """System must abstain when top-2 candidates have conflicting near-identical scores."""
        engine = AbstentionEngine(confidence_threshold=0.30, margin_threshold=0.05)
        cand1 = ReconstructionCandidate(
            candidate_id="CAND-01",
            candidate_text="ஸ்வஸ்தி",
            provenance=Provenance.CORPUS,
            rank_score=0.75,
        )
        cand2 = ReconstructionCandidate(
            candidate_id="CAND-02",
            candidate_text="திருமகள்",
            provenance=Provenance.CORPUS,
            rank_score=0.73,  # margin = 0.02 < 0.05
        )
        dec = engine.evaluate([cand1, cand2])
        self.assertTrue(dec.should_abstain)
        self.assertIn("Contradictory evidence", dec.reason)

    def test_accept_on_high_confidence(self):
        """System accepts candidate when confidence is high and margin is clear."""
        engine = AbstentionEngine(confidence_threshold=0.35, margin_threshold=0.05)
        cand1 = ReconstructionCandidate(
            candidate_id="CAND-01",
            candidate_text="ஸ்வஸ்தி ஸ்ரீ",
            provenance=Provenance.OBSERVED,
            rank_score=0.88,
        )
        cand2 = ReconstructionCandidate(
            candidate_id="CAND-02",
            candidate_text="[UNKNOWN]",
            provenance=Provenance.UNKNOWN,
            rank_score=0.10,
        )
        dec = engine.evaluate([cand1, cand2])
        self.assertFalse(dec.should_abstain)
        self.assertEqual(dec.effective_text, "ஸ்வஸ்தி ஸ்ரீ")

    def test_risk_coverage_curve(self):
        """Risk coverage curve should return non-empty coverages and risks."""
        confs = [0.9, 0.8, 0.7, 0.4, 0.2]
        corr = [1, 1, 1, 0, 0]
        curve = compute_risk_coverage_curve(confs, corr, num_thresholds=10)
        self.assertIn("coverages", curve)
        self.assertIn("risks", curve)
        self.assertIn("aurc", curve)
        self.assertEqual(len(curve["coverages"]), 10)


if __name__ == "__main__":
    unittest.main()
