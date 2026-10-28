# EpiRecover — Implementation Status Matrix

**Project**: EpiRecover: Evidence-Grounded Multimodal Reconstruction of Damaged Ancient Indian Inscriptions from Monument Imagery  
**Document**: Implementation & Verification Status  
**Date**: October 2026 (Verified Baseline v1.0)  
**Target Language Scope**: Tamil Only (Modern Tamil & Historical Tamil-Brahmi)  
**Methodology Constraint**: Strictly Non-Generative (No Generative LLMs; Multimodal Evidence Fusion Only)

---

## 1. Feature Status Matrix

| Module / Feature | Status | Implementation File | Verification / Test |
|---|---|---|---|
| **Stage A: Upload & Metadata** | **Completed** | `epirecover_pipeline.py`, `utils.py` | Validated via CLI, FastAPI, & Streamlit; Tamil-only metadata container |
| **Stage B: OpenCV Preprocessing** | **Completed** | `preprocessing.py` | 13 unit tests passing in `tests/test_preprocessing.py` |
| **Stage C: Damage Localization** | **Completed** (Heuristic baseline + U-Net++ architecture) | `src/damage_analysis/damage_detector.py`, `unet_plus_plus.py`, `yolo_detector.py` | Classical OpenCV baseline labeled `HEURISTIC_BASELINE`; PyTorch UNet++ & YOLO modules ready |
| **Stage A4: Restoration Branch (Ablation)** | **Completed** (NAFNet architecture + Bilateral fallback) | `src/restoration/nafnet_restoration.py` | PyTorch NAFNet module; PSNR & SSIM evaluation in `tests/test_unet_nafnet.py` |
| **Stage D: Line OCR & Uncertainty** | **Completed** | `ocr_engine.py`, `src/ocr/transformer_line_ocr.py` | Multi-engine OCR adapter + Transformer Line OCR; explicit `[UNKNOWN]` markers |
| **Stage E: Non-Generative Reconstruction** | **Completed** | `src/reconstruction/candidate_generator.py` | Candidates preserve provenance tags; strictly non-LLM; Tamil confusion model |
| **Stage F: Corpus Evidence Retrieval** | **Completed** | `src/evidence_retrieval/` | 14 unit tests in `tests/test_evidence_retrieval.py`; strict anti-leakage exclusion |
| **Stage B2: PostgreSQL Historical Storage** | **Completed** | `src/evidence_retrieval/postgres_adapter.py` | Relational schema with automatic SQLite local fallback |
| **Stage G: Multimodal Evidence Fusion** | **Completed** (V + O + L + H + R) | `src/candidate_ranking/fusion.py`, `ranker.py` | 3 unit tests in `tests/test_fusion.py`; scikit-learn interface for learned weights |
| **Stage C3: Confidence Calibration** | **Completed** | `src/candidate_ranking/calibration.py` | Temperature scaling, ECE, and Brier score tested in `tests/test_calibration_abstention.py` |
| **Stage C4: Abstention Decision Engine** | **Completed** | `src/candidate_ranking/abstention.py` | Chow reject rule & risk-coverage curve tested in `tests/test_calibration_abstention.py` |
| **Human Review Workflow** | **Completed** | `src/review/review_store.py` | Integrated with review store, Streamlit panel, & FastAPI review endpoints |
| **FastAPI Backend (11 Endpoints)** | **Completed** | `src/api/main.py` | 5 unit tests in `tests/test_api.py`; runs on Uvicorn |
| **React/Vite Expert Review Frontend** | **Completed** | `frontend/` (`App.jsx`, `index.css`) | Full visual comparison, Top-K candidate score breakdown, and review ledger |
| **Streamlit Research Dashboard** | **Completed** | `streamlit_app.py` | 10 research sections, non-generative fusion display, runs on port 8501 |
| **Synthetic Damage Benchmark (D0–D6)** | **Completed** | `src/evaluation/synthetic_damage.py` | 3 unit tests in `tests/test_synthetic_damage.py`; exact ground-truth masks |
| **Quantitative Evaluation (CER/WER/Top-K)** | **Completed** | `src/evaluation/metrics.py`, `evaluator.py` | 15 unit tests in `tests/test_evaluation_metrics.py`; NFC normalization; Top-1/Top-5 accuracy |

---

## 2. Test Verification Summary

Automated testing executed via `pytest` in Python 3.13.2 environment:
- **Total Test Cases**: 117
- **Passed**: 117
- **Failed**: 0
- **Skipped / Errored**: 0
- **Execution Time**: ~7.0 seconds
- **Dependencies Required**: Zero paid APIs; uses synthetic fixtures and pure-Python fallbacks.

```
tests/test_api.py                     : 5 passed
tests/test_calibration_abstention.py  : 7 passed
tests/test_candidate_generator_regression.py : 20 passed
tests/test_candidate_ranking.py       : 7 passed
tests/test_dataset_matcher.py         : 10 passed
tests/test_evaluation_metrics.py      : 15 passed
tests/test_evaluator.py               : 3 passed
tests/test_evidence_retrieval.py      : 14 passed
tests/test_fusion.py                  : 3 passed
tests/test_preprocessing.py           : 13 passed
tests/test_synthetic_damage.py        : 3 passed
tests/test_tamil_pipeline.py          : 12 passed
tests/test_unet_nafnet.py             : 5 passed
============================= 117 passed in 7.01s =============================
```

---

## 3. Research Integrity Commitments Audit

- [x] **No Generative LLMs**: All prompt-based reconstruction has been removed. Candidates are strictly attributed to visual observations, OCR hypotheses, historical corpus records, or documented stone-erosion confusion rules.
- [x] **Tamil-Only Scope**: Target language is strictly Tamil. Modern Tamil script and historical Tamil-Brahmi script are explicitly separated.
- [x] **Clear Damage Baselines**: Heuristic damage masks are explicitly stamped `HEURISTIC_BASELINE`, never mislabeled as deep learning predictions.
- [x] **Provenance Tracking**: Every reconstruction candidate preserves its source tag (`OBSERVED`, `OCR_PRED`, `CORPUS_RETRIEVED`, `MODEL_INFERRED`, `HUMAN_REVIEWED`, `UNKNOWN`).
- [x] **Data Leakage Immunity**: Evaluated records are systematically excluded from the evidence retrieval index.
- [x] **Synthetic Fixture Transparency**: All fixture files carry clear warnings stating they are synthetic testing data and not actual epigraphs.
- [x] **No Hallucinated Benchmarks**: All evaluation reports strictly output `unavailable` when ground-truth files are absent.
