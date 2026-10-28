"""
tests/test_api.py
-----------------
Tests for FastAPI backend endpoints.
"""

import unittest
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient
from src.api.main import app


class TestFastAPIBackend(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        """GET /api/health must return 200 with Tamil scope and non-generative guarantee."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("Tamil", data["language_scope"])
        self.assertIn("Strict Non-LLM", data["non_generative_guarantee"])
        self.assertIn("PyTorch", data["technologies"])

    def test_corpus_search_endpoint(self):
        """GET /api/corpus/search must return matching records."""
        res = self.client.get("/api/corpus/search?query=கோப்பரகேசரி")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("records", data)

    def test_save_and_list_review_decision(self):
        """POST /api/review/decision and GET /api/review/decisions must record and retrieve."""
        req_body = {
            "inscription_id": "TEST-INS-01",
            "candidate_id": "CAND-01",
            "candidate_text": "ஸ்வஸ்தி ஸ்ரீ",
            "decision": "ACCEPTED",
            "notes": "Verified against South Indian Inscriptions citation.",
        }
        res = self.client.post("/api/review/decision", json=req_body)
        self.assertEqual(res.status_code, 200)
        saved = res.json()["decision"]
        self.assertEqual(saved["inscription_id"], "TEST-INS-01")
        self.assertEqual(saved["decision"], "ACCEPTED")

        # Verify listing
        list_res = self.client.get("/api/review/decisions")
        self.assertEqual(list_res.status_code, 200)
        all_decs = list_res.json()["decisions"]
        self.assertTrue(any(d["inscription_id"] == "TEST-INS-01" for d in all_decs))

    def test_evaluation_run_endpoint(self):
        """POST /api/evaluation/run calculates CER/WER for hypothesis vs reference."""
        req_body = {
            "inscription_id": "EVAL-01",
            "hypothesis": "வணக்கம்",
            "reference": "வணக்கம்",
            "script": "tamil",
        }
        res = self.client.post("/api/evaluation/run", json=req_body)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["cer"], 0.0)
        self.assertEqual(data["wer"], 0.0)

    def test_pipeline_process_endpoint(self):
        """POST /api/pipeline/process runs Stages A to G on an uploaded synthetic fixture."""
        # Create a small 100x100 synthetic image
        img = np.full((100, 100, 3), 220, dtype=np.uint8)
        _, img_bytes = cv2.imencode(".png", img)

        files = {"file": ("test_inscription.png", io.BytesIO(img_bytes.tobytes()), "image/png")}
        data = {
            "inscription_id": "API-TEST-01",
            "script": "tamil",
            "historical_period": "10th century CE",
            "region": "Thanjavur",
            "monument": "Brihadisvara",
            "verified_transcription": "ஸ்வஸ்தி ஸ்ரீ",
        }
        res = self.client.post("/api/pipeline/process", files=files, data=data)
        self.assertEqual(res.status_code, 200)
        report = res.json()
        self.assertEqual(report["inscription_id"], "API-TEST-01")
        self.assertIn("candidates", report)
        self.assertIn("evaluation", report)


if __name__ == "__main__":
    unittest.main()
