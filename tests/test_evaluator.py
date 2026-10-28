"""
tests/test_evaluator.py
-----------------------
Unit tests for src/evaluation/evaluator.py.

Uses the synthetic fixture data/annotations/sample_eval_set.json.
No external network or GPU models required.
"""

import json
import os
import shutil
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.evaluator import ExperimentConfig, InscriptionEvaluator


class TestInscriptionEvaluator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.evaluator = InscriptionEvaluator(output_dir=self.temp_dir)
        self.fixture_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "annotations", "sample_eval_set.json"
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_dataset(self):
        records = self.evaluator.load_dataset(self.fixture_path)
        self.assertIsInstance(records, list)
        self.assertGreaterEqual(len(records), 3)
        self.assertEqual(records[0]["id"], "EVAL-001")

    def test_run_evaluation_perfect_match(self):
        records = self.evaluator.load_dataset(self.fixture_path)
        hypotheses = {r["id"]: r["ground_truth_transcription"] for r in records}

        config = ExperimentConfig(
            experiment_id="test_perfect",
            description="Perfect match test run",
        )
        report = self.evaluator.run_evaluation(records, hypotheses, config)

        self.assertEqual(report.total_samples, len(records))
        self.assertEqual(report.evaluated_samples, len(records))
        self.assertEqual(report.mean_cer, 0.0)
        self.assertEqual(report.mean_wer, 0.0)

        # Check export files exist
        summary_json = os.path.join(self.temp_dir, "test_perfect_summary.json")
        details_csv = os.path.join(self.temp_dir, "test_perfect_details.csv")
        self.assertTrue(os.path.exists(summary_json))
        self.assertTrue(os.path.exists(details_csv))

    def test_run_evaluation_partial_errors(self):
        records = [
            {
                "id": "T-1",
                "ground_truth_transcription": "hello world",
                "script": "roman",
                "damage_level": "low",
            }
        ]
        hypotheses = {"T-1": "helo world"}

        config = ExperimentConfig(
            experiment_id="test_error",
            description="1 char deletion test",
        )
        report = self.evaluator.run_evaluation(records, hypotheses, config)
        self.assertIsNotNone(report.mean_cer)
        self.assertGreater(report.mean_cer, 0.0)


if __name__ == "__main__":
    unittest.main()
