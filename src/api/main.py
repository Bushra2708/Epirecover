"""
src/api/main.py
----------------
FastAPI Backend for EpiRecover: Evidence-Grounded Inscription Reconstruction.

As specified in EpiRecover PPT Slide 13:
  Software requirements: "FastAPI + React/Vite (review interface)"

Endpoints:
  - GET  /api/health              : System status, Python version, GPU & OCR engine status
  - POST /api/pipeline/process    : Run end-to-end multimodal reconstruction pipeline (Tamil only)
  - GET  /api/corpus/search       : Retrieve historical evidence from BharatSHRI / SIDDHAM corpus
  - POST /api/review/decision     : Record human epigraphist review decision
  - GET  /api/review/decisions    : Retrieve saved human review provenance
  - POST /api/evaluation/run      : Compute CER, WER, Top-1/Top-5 accuracy, ECE, Brier score
  - POST /api/damage/synthetic    : Generate D0-D6 synthetic damage benchmarks with exact ground truth

EpiRecover, 2026.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import cv2
import numpy as np
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Core pipeline imports
from epirecover_pipeline import EpiRecoverPipeline, InscriptionMetadata
import ocr_engine
from utils import Provenance, RecoveryReport
from src.evidence_retrieval.retriever import CorpusRetriever
from src.review.review_store import ReviewStore, ReviewDecision
from src.evaluation.metrics import evaluate_single, aggregate_results, compute_cer, compute_wer
from src.evaluation.synthetic_damage import SyntheticDamageGenerator

app = FastAPI(
    title="EpiRecover API",
    description="Backend API for Evidence-Grounded Reconstruction of Damaged Tamil Inscriptions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline instance
pipeline = EpiRecoverPipeline(llm_backend="stub")
corpus_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus")
retriever = CorpusRetriever(corpus_dir=corpus_dir)
review_store = ReviewStore(store_path="outputs/reports/review_decisions.json")
damage_gen = SyntheticDamageGenerator()


class ReviewDecisionRequest(BaseModel):
    decision_id: Optional[str] = None
    inscription_id: str
    candidate_id: str
    candidate_text: str
    decision: str  # "ACCEPTED", "EDITED", "REJECTED", "MARK_UNKNOWN"
    notes: Optional[str] = None
    reviewer_id: Optional[str] = "expert_epigraphist"


class EvaluationRequest(BaseModel):
    hypothesis: str
    reference: Optional[str] = None
    inscription_id: str = "EVAL-01"
    script: str = "tamil"
    damage_level: Optional[str] = None


@app.get("/api/health")
def health_check():
    ocr_support = ocr_engine.check_tamil_ocr_support()
    return {
        "status": "healthy",
        "service": "EpiRecover",
        "python_version": sys.version,
        "language_scope": "Tamil (Modern Tamil & Tamil-Brahmi)",
        "ocr_status": ocr_support,
        "non_generative_guarantee": "Strict Non-LLM Evidence Fusion",
        "technologies": [
            "PyTorch", "OpenCV", "scikit-learn", "Sentence-Transformers",
            "FastAPI", "React/Vite", "Streamlit"
        ],
    }


@app.post("/api/pipeline/process")
async def process_inscription(
    file: UploadFile = File(...),
    inscription_id: Optional[str] = Form(None),
    script: str = Form("tamil"),
    historical_period: Optional[str] = Form(None),
    region: Optional[str] = Form(None),
    monument: Optional[str] = Form(None),
    verified_transcription: Optional[str] = Form(None),
    apply_perspective: bool = Form(False),
):
    """
    Process an uploaded monument inscription image through Stages A to G.
    Target language is strictly Tamil in this version.
    """
    # Enforce script warning for Tamil-Brahmi
    if "tamil-brahmi" in script.lower():
        script_warning = "Historical Tamil-Brahmi selected. Standard OCR engines cannot process ancient lithic glyphs; relying on corpus evidence."
    else:
        script_warning = None

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img)

    try:
        metadata = InscriptionMetadata(
            inscription_id=inscription_id,
            script=script,
            language="Tamil",
            historical_period=historical_period,
            region=region,
            monument=monument,
        )
        report = pipeline.process(tmp_path, metadata=metadata, apply_perspective=apply_perspective)
        report_dict = report.as_dict()
        if script_warning:
            report_dict["script_warning"] = script_warning

        # Add CER evaluation if ground truth transcription was supplied
        if verified_transcription:
            cer = compute_cer(report.cleaned_text, verified_transcription)
            wer = compute_wer(report.cleaned_text, verified_transcription)
            report_dict["evaluation"] = {
                "ground_truth": verified_transcription,
                "cer": cer,
                "wer": wer,
            }
        return report_dict
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/api/corpus/search")
def search_corpus(
    query: str = Query(..., min_length=1),
    script: Optional[str] = Query(None),
    top_k: int = Query(5, ge=1, le=20),
):
    """Search historical Tamil inscription records."""
    records = retriever.retrieve(query=query, script=script, language="tamil", top_k=top_k)
    return {"query": query, "total_found": len(records), "records": records}


@app.post("/api/review/decision")
def save_review_decision(req: ReviewDecisionRequest):
    """Record an expert review decision with full provenance."""
    decision = ReviewDecision(
        decision_id=req.decision_id or f"DEC-{req.inscription_id}",
        inscription_id=req.inscription_id,
        candidate_id=req.candidate_id,
        candidate_text=req.candidate_text,
        decision=req.decision.upper(),
        reviewer_notes=req.notes or "",
        reviewer_id=req.reviewer_id or "expert_epigraphist",
    )
    saved = review_store.save_decision(decision)
    return {"status": "saved", "decision": saved.as_dict()}


@app.get("/api/review/decisions")
def list_review_decisions():
    """List all saved human review decisions."""
    decisions = review_store.get_all_decisions()
    return {"count": len(decisions), "decisions": [d.as_dict() for d in decisions]}


@app.post("/api/evaluation/run")
def run_evaluation(req: EvaluationRequest):
    """Compute CER, WER, and metrics for hypothesis vs reference."""
    res = evaluate_single(
        inscription_id=req.inscription_id,
        hypothesis=req.hypothesis,
        reference=req.reference,
        script=req.script,
        damage_level=req.damage_level,
    )
    return res.as_dict()


@app.post("/api/damage/synthetic")
async def generate_synthetic_damage(
    file: UploadFile = File(...),
    damage_level: str = Form("D1"),
):
    """Generate reproducible D0-D6 synthetic damage benchmark with ground truth mask."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    damage_out = damage_gen.apply_damage(img, level=damage_level)
    return {
        "damage_level": damage_out.damage_level,
        "damage_ratio": damage_out.damage_ratio,
        "applied_operations": damage_out.applied_operations,
        "mask_shape": list(damage_out.damage_mask.shape),
    }
