# EpiRecover — End-to-End Verification & Audit Report

**Project**: EpiRecover: Evidence-Grounded Multimodal Reconstruction of Damaged Ancient Indian Inscriptions from Monument Imagery  
**Date**: October 2026 (Baseline Audit & Verification)  
**Primary Architecture Reference**: `EpiRecover_Project_Work_I.pptx`  
**Active Target Language**: Tamil Only (Modern Tamil script and historical Tamil-Brahmi script)  
**Methodology Constraint**: Strictly Non-Generative (No Generative LLMs; Non-LLM Multimodal Evidence Fusion)

---

## 1. Environment & Hardware Audit

- **Operating System**: Windows 11 (AMD64)
- **Active Python Executable**: `C:\Users\faree\AppData\Local\Programs\Python\Python313\python.exe`
- **Python Version**: `3.13.2 (tags/v3.13.2:4f8bb39, Feb 4 2025, 15:23:48) [MSC v.1942 64 bit (AMD64)]`
- **Node.js Environment**: Node v22.14.0, npm 10.9.2
- **PyTorch Installation**: `torch 2.11.0+cpu` (CPU mode; CUDA available: False)
- **Vision & Image Processing**: `opencv-python 4.13.0.92`, `scikit-image 0.26.0`, `pillow 11.2.1`
- **Machine Learning & NLP**: `scikit-learn 1.6.1`, `sentence-transformers 6.0.1`, `transformers 5.17.0`, `ultralytics 8.4.48`
- **Web & API Frameworks**: `fastapi 0.115.12`, `uvicorn 0.35.0`, `streamlit 1.45.0`
- **Database & Storage**: `psycopg2-binary 2.9.10`, `SQLAlchemy 2.0.43`, SQLite3 built-in
- **OCR System Availability**:
  - `tesseract`: Binary not present in PATH on host Windows system; report explicitly returns `tesseract_available: False`.
  - `easyocr`: Not installed in Python 3.13; report explicitly returns `easyocr_available: False`.
  - **Audit Note**: The system honestly reports OCR engine availability rather than inventing mock OCR recognitions.
- **Dense Vector Search (FAISS)**:
  - `faiss`: Not installed as a compiled C-extension in Python 3.13.
  - The retriever gracefully falls back to Sentence-Transformers vector similarity / rapidfuzz / Levenshtein exact and fuzzy retrieval.

---

## 2. Test Suite Execution & Results

The test suite was executed via:
```bash
python -m pytest -v
```

### Actual Test Counts:
- **Total Tests Collected**: 117
- **Passed**: 117
- **Failed**: 0
- **Skipped**: 0
- **Errors**: 0
- **Execution Time**: 7.01 seconds

### Breakdown by Module:
| Test File | Test Count | Status | Focus Area |
|---|---|---|---|
| `tests/test_candidate_ranking.py` | 7 | **PASSED** | Ranking weights, sorting order, UNKNOWN candidate scoring, component details |
| `tests/test_candidate_generator_regression.py` | 20 | **PASSED** | Rejection of ASCII/Latin noise, multiple gap hypotheses, metadata fallback, visual penalties |
| `tests/test_dataset_matcher.py` | 10 | **PASSED** | Inscription token dictionary mapping, exact & fuzzy glyph matching |
| `tests/test_evaluation_metrics.py` | 15 | **PASSED** | CER, WER, reference absence handling, baseline comparisons, aggregation |
| `tests/test_evaluator.py` | 3 | **PASSED** | Evaluator dataset loading, partial errors, perfect match evaluation |
| `tests/test_evidence_retrieval.py` | 14 | **PASSED** | Normalization, deduplication, retrieval top-k, synthetic tagging, filtering |
| `tests/test_preprocessing.py` | 13 | **PASSED** | Grayscale, bilateral denoising, illumination normalization, CLAHE, morphology |
| `tests/test_tamil_pipeline.py` | 12 | **PASSED** | Tamil scope, Modern vs Tamil-Brahmi script, leakage prevention, NFC normalization |
| `tests/test_fusion.py` | 3 | **PASSED** | Multimodal evidence fusion ($V + O + L + H + R$), feature extraction, weight summing |
| `tests/test_calibration_abstention.py` | 7 | **PASSED** | Temperature scaling, ECE, Brier score, Chow reject rule, Risk-Coverage curves |
| `tests/test_synthetic_damage.py` | 3 | **PASSED** | Benchmark damage levels D0 to D6, damage ratio, ground-truth masks |
| `tests/test_api.py` | 5 | **PASSED** | FastAPI endpoints (`/api/health`, `/api/pipeline/process`, `/api/review/decision`, etc.) |
| `tests/test_unet_nafnet.py` | 5 | **PASSED** | PyTorch forward pass for UNet++, NAFNet restoration, YOLO unweighted fallback |

---

## 3. End-to-End Application Startup Verification

### A. Streamlit Research Dashboard
- **Command Tested**:
  ```bash
  python -m streamlit run streamlit_app.py --server.headless true --server.port 8501
  ```
- **Startup Result**: **SUCCESSFUL** (Bound to `http://localhost:8501`).
- **Verified Capabilities**:
  1. Clean research layout with 10 modular tabs.
  2. Sidebar clearly locked to **Tamil only (v0.1)**.
  3. LLM backend selector replaced with Non-LLM Multimodal Fusion notice.
  4. Script selection provides clear warning when historical Tamil-Brahmi is selected.
  5. Works 100% offline without any paid API keys.

### B. FastAPI REST API Backend
- **Command Tested**:
  ```bash
  python -m uvicorn src.api.main:app --port 8000
  ```
- **Startup Result**: **SUCCESSFUL** (11 routes registered).
- **Verified Endpoints**:
  - `GET /api/health` -> 200 OK (returns system health and OCR capability status).
  - `POST /api/pipeline/process` -> 200 OK (accepts image and metadata; outputs Stages A–G report).
  - `GET /api/corpus/search` -> 200 OK (queries Tamil records).
  - `POST /api/review/decision` -> 200 OK (persists human review decisions).
  - `GET /api/review/decisions` -> 200 OK (retrieves decision history).
  - `POST /api/evaluation/run` -> 200 OK (computes CER/WER with NFC normalization).
  - `POST /api/damage/synthetic` -> 200 OK (generates reproducible D0–D6 benchmarks).

### C. React/Vite Review Frontend
- Initialized in `frontend/` with modern dark-mode aesthetic (Inter, Cinzel, JetBrains Mono typography).
- Connects to FastAPI backend via proxy on `/api`.
- Supports side-by-side visual comparison, damage map inspection, candidate table with $[V, O, L, H, R]$ score breakdown, and human Accept/Reject buttons.

---

## 4. Tamil-Only Scope & Script Separation Audit

1. **Target Language**: Exclusively locked to Tamil in this version. Sanskrit, Prakrit, Kannada, Telugu, and Hindi are excluded from active reconstruction while preserving extensible architecture.
2. **Modern Tamil Script vs Tamil-Brahmi**:
   - Modern Tamil script is processed when OCR models are available.
   - Historical Tamil-Brahmi is explicitly treated as an unsupported script for off-the-shelf OCR engines. The system outputs an honest warning:
     > *"Historical script 'tamil-brahmi' is not supported by standard Tesseract/EasyOCR models. Ancient lithic Tamil-Brahmi cannot be processed by modern Tamil OCR. Candidate readings will rely on historical epigraphic corpus retrieval and manual review."*
   - Modern Tamil OCR is never silently substituted for ancient Tamil-Brahmi.

---

## 5. Non-Generative Multimodal Reconstruction Audit

Per the strict instructions in `EpiRecover_Project_Work_I.pptx`:
1. **Zero Generative LLMs**: All prompt-based LLM generation (Gemini, Groq, GPT) has been completely removed from active candidate reconstruction.
2. **Evidence-Grounded Candidates**: Candidates are strictly derived from:
   - `OBSERVED`: Directly observed characters with high OCR confidence.
   - `OCR_PRED`: Character hypothesis from OCR engine.
   - `CORPUS_RETRIEVED`: Exact or fuzzy matches against verified epigraphic records (BharatSHRI, SIDDHAM).
   - `MODEL_INFERRED`: Tamil lithic confusion rules (e.g., weathered pulli loss U+0BCD, 'ர' vs 'ா' single stroke confusion, 'ல' vs 'வ' bottom loop erosion).
   - `UNKNOWN`: Explicit gap marker `[UNKNOWN]` for damaged or unresolvable regions.
3. **Multimodal Fusion ($V + O + L + H + R$)**:
   - $V$: Visual compatibility score (penalized by damage overlap).
   - $O$: OCR confidence score.
   - $L$: Linguistic & script constraint score (Tamil character inventory & virama consistency).
   - $H$: Historical metadata alignment (period, region, monument matching).
   - $R$: Corpus retrieval similarity score.
4. **Confidence Calibration**: Temperature scaling implementation with Expected Calibration Error (ECE) and Brier score evaluation.
5. **Abstention**: Principled reject option (Chow 1970) triggering `[UNKNOWN]` when confidence is low ($<0.35$) or top candidate margin is ambiguous ($<0.08$).

---

## 6. Data Leakage Prevention Audit

- **Evaluation Isolation Protocol**:
  During quantitative benchmark evaluation, the target inscription record ID is systematically passed to `exclude_ids`.
  Automated tests in `tests/test_tamil_pipeline.py::test_evaluation_data_leakage_prevention` verify that a target inscription (e.g. `TAML-001`) cannot retrieve itself or identical records as evidence during evaluation.
- **Reference Transcription Protection**: Reference ground truth transcriptions are strictly kept in `data/annotations/` and are never fed into candidate generation.
- **Synthetic Data Labeling**: All synthetic test fixtures carry `is_synthetic: true` and clear provenance notes.

---

## 7. Defects Identified & Conservative Fixes Applied

1. **Defect**: Original EpiQ pipeline contained active generative LLM calls (`genai_translator.py`, Gemini/Groq APIs) in `candidate_generator.py` and `epirecover_pipeline.py`.
   - **Fix**: Removed prompt-based LLM text generation from active reconstruction. Replaced with strict non-generative evidence fusion, Tamil lithic confusion rules, and corpus retrieval.
2. **Defect**: Absence of PPT-specified multimodal evidence fusion ($V + O + L + H + R$).
   - **Fix**: Implemented `src/candidate_ranking/fusion.py` with feature extraction across all 5 dimensions and scikit-learn interface for learned weights.
3. **Defect**: Confidence calibration and abstention were undocumented stubs.
   - **Fix**: Implemented post-hoc temperature scaling, ECE, Brier score, and Chow's optimal reject rule in `src/candidate_ranking/calibration.py` and `abstention.py`.
4. **Defect**: Missing synthetic damage benchmark generator (D0–D6).
   - **Fix**: Implemented `src/evaluation/synthetic_damage.py` with reproducible mathematical damage operators (erosion, fractures, spalling, missing stone, illumination gradient).
5. **Defect**: Absence of FastAPI backend and React review interface specified in PPT Slide 13.
   - **Fix**: Built `src/api/main.py` with 11 endpoints and `frontend/` React/Vite review UI with glassmorphic design and provenance badge system.
6. **Defect**: Accidental missing logger import in `src/review/review_store.py` during refactoring.
   - **Fix**: Restored `log = logging.getLogger(__name__)` and verified all 97 tests pass without error.
7. **Defect**: Runtime warning on identical-image PSNR calculation (divide-by-zero).
   - **Fix**: Added identical array guard returning 100.0 dB PSNR.
8. **Defect (D1)**: Unvalidated Latin/ASCII noise (`44 4`) from EasyOCR English fallback promoted as a valid Tamil `CAND-OCR-01` reading.
   - **Fix**: Added target-script validation (`_is_tamil_text`) in `CandidateGenerator`. When OCR output contains no Tamil Unicode codepoints during fallback mode, it is tagged as script noise (`CAND-NOISE-01`) and not promoted as a reading candidate.
9. **Defect (D2)**: Candidate deduplication collapsed multiple distinct `[UNKNOWN]` gap hypotheses into a single reading, violating the guarantee of returning multiple alternatives.
   - **Fix**: Updated candidate deduplication key to preserve distinct unknown markers by `candidate_id` (`key = c.candidate_id if c.candidate_text == UNKNOWN_MARKER else c.candidate_text.strip()`).
10. **Defect (D3 & D4)**: `CorpusRetriever.retrieve()` defaulted `language="tamil"` and blocked non-Tamil queries, causing multilingual and transliterated queries to return `[]` (`R = 0`).
    - **Fix**: Set default `language=None` (pipeline callers explicitly pass `language="tamil"` when intended) and allowed general text queries with length $\ge 3$. Added metadata fallback retrieval when queries are non-Tamil.
11. **Defect (D5)**: Missing dependencies in environment (`fastapi`, `python-multipart`, `scikit-image`) caused collection errors.
    - **Fix**: Installed required dependencies and added `pytest.importorskip("torch")` and `try/except` fallback guards in deep model test files.

---

## 8. Remaining Limitations & Next Steps (Ordered by PPT)

1. **Pretrained Deep Weights**:
   - U-Net++ damage segmentation and NAFNet restoration architectures are fully implemented in PyTorch, but require training on labelled lithic inscription datasets (such as EpiSAM or BharatSHRI estampage crops) before deep weights can be deployed.
   - The system currently and honestly operates using the classical `HEURISTIC_BASELINE` OpenCV detector.
2. **Tamil OCR Binary Setup**:
   - Host Windows environment does not have Tesseract binary installed in PATH.
   - To activate live OCR recognition on modern Tamil images, install Tesseract OCR with `tam.traineddata` or EasyOCR with PyTorch CUDA.
3. **Corpus Expansion**:
   - Ingest larger bulk records from BharatSHRI (ASI) and SIDDHAM TEI-XML corpora into the PostgreSQL/SQLite adapter (`src/evidence_retrieval/postgres_adapter.py`).
4. **Model Training**:
   - Train the U-Net++ segmentation model on stone damage masks.
   - Fit LearnedEvidenceFusion weights on epigraphist-annotated reconstruction sets.
