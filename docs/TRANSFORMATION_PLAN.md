# EpiRecover — Transformation Plan

**Upstream project**: EpiQ (Ancient Inscription Translator, `sam-r-148/EpiQ`)
**Target project**: EpiRecover — Evidence-Grounded Multimodal Reconstruction of Damaged Ancient Indian Inscriptions from Monument Imagery
**Author**: B.Tech AI & Data Science Research Project
**Document status**: Phase 1 complete (repository inspection and migration plan)

---

## 1. Upstream Codebase Audit

### 1.1 Files Inspected

| File | Size | Status | Notes |
|---|---|---|---|
| `main.py` | 170 lines | Functional | CLI + orchestrator (InscriptionTranslator) |
| `preprocessing.py` | 202 lines | Functional | Full OpenCV pipeline; all 7 stages working |
| `ocr_engine.py` | 290 lines | Functional (with caveats) | Tesseract + EasyOCR ensemble; missing Brahmi models |
| `dataset_matcher.py` | 182 lines | Functional | Exact + Levenshtein fuzzy match; no FAISS |
| `genai_translator.py` | 412 lines | Functional (API-keyed) | Gemini/Groq/Stub backends; StubTranslator works offline |
| `streamlit_app.py` | 343 lines | Functional | Full Streamlit UI; single-page |
| `utils.py` | 82 lines | Functional | PipelineReport dataclass + image I/O |
| `data/script_mappings.json` | 272 lines | Data: seed only | 63 Brahmi glyph/token entries |
| `requirements.txt` | 26 lines | Needs extension | Missing: rapidfuzz, scipy, FAISS |
| `.gitignore` | 6 lines | Needs extension | Missing model weights, datasets patterns |

### 1.2 What Currently Works (No API Key Required)

- Full OpenCV preprocessing pipeline (grayscale -> CLAHE -> adaptive threshold -> morphology)
- Tesseract OCR (if binary + language packs installed)
- EasyOCR (if installed; heavy PyTorch dependency)
- Script-mapping exact and fuzzy match against data/script_mappings.json
- StubTranslator offline mode
- Streamlit UI in offline/stub mode

### 1.3 What Requires External APIs or Models

| Feature | Dependency | Risk |
|---|---|---|
| Gemini translation | GEMINI_API_KEY | Medium — stub fallback exists |
| Groq translation | GROQ_API_KEY | Medium — stub fallback exists |
| Sanskrit Tesseract (san) | tessdata san.traineddata | Low — degrades to eng only |
| EasyOCR Devanagari/Tamil | PyTorch + model download | Medium — large download |
| IndicTrans2 offline HF | HuggingFace + GPU/RAM | High — ~4 GB model |

### 1.4 Identified Gaps vs EpiRecover Requirements

| EpiRecover Requirement | Gap in EpiQ |
|---|---|
| Stage C: Damage localization | Missing entirely |
| Stage E: Candidate reconstruction with provenance | Partial — LLM only, no structured candidates |
| Stage F: BharatSHRI / corpus retrieval | Missing entirely |
| Stage G: Evidence-grounded ranking | Missing entirely |
| Stage H: Evaluation (CER/WER) | Missing entirely |
| Human review workflow | Missing entirely |
| Metadata collection (script, period, region) | Missing from upload stage |
| Per-character uncertainty markers | Partial — notation in LLM prompts only |
| Structured candidate output with provenance | Missing |
| Unit tests | Missing entirely |
| Structured logging | Partial |
| .env config / path abstraction | Missing |

---

## 2. Architecture Migration Plan

### 2.1 Modules to Reuse Unchanged

- preprocessing.py — keep as-is; minor additions for intermediate saving
- ocr_engine.py — keep; add adapter interface and uncertainty markers
- dataset_matcher.py — keep; extend with FAISS embedding index path

### 2.2 Modules to Extend

| Module | Extensions |
|---|---|
| main.py | Extend InscriptionTranslator to EpiRecoverPipeline; add metadata parameter |
| utils.py | Add RecoveryReport dataclass with candidates, damage mask, evidence |
| genai_translator.py | Extend StubTranslator for offline reconstruction; add EpiRecover prompt |
| streamlit_app.py | Full dashboard rewrite with 10 sections; retain pipeline calls |

### 2.3 New Modules (in src/)

| Module | Path | Purpose |
|---|---|---|
| Damage analysis | src/damage_analysis/damage_detector.py | Heuristic damage mask + annotation; U-Net plug-in |
| Reconstruction | src/reconstruction/candidate_generator.py | Structured candidate list with evidence provenance |
| Evidence retrieval | src/evidence_retrieval/corpus_adapter.py | BharatSHRI / CSV / JSON corpus ingestion |
| Evidence retrieval | src/evidence_retrieval/retriever.py | Exact + fuzzy + FAISS retrieval |
| Candidate ranking | src/candidate_ranking/ranker.py | Transparent scoring with configurable weights |
| Evaluation | src/evaluation/metrics.py | CER, WER, aggregate reporting |
| Evaluation | src/evaluation/evaluator.py | Per-inscription experiment runner |
| Review | src/review/review_store.py | JSON-backed human-review decision store |

### 2.4 Target Directory Structure

```
EpiRecover/
  main.py                        # kept for backward-compat CLI
  epirecover_pipeline.py         # main orchestrator (new)
  preprocessing.py               # reused
  ocr_engine.py                  # reused
  dataset_matcher.py             # reused
  genai_translator.py            # extended
  streamlit_app.py               # full rewrite
  utils.py                       # extended
  config/
    epirecover.yaml              # runtime configuration
  data/
    raw/                         # uploaded images (original, unmodified)
    processed/                   # preprocessing intermediates
    annotations/                 # damage masks, ground-truth transcriptions
    corpus/                      # BharatSHRI records, CSVs, JSONL
      bharatshri_sample.jsonl    # synthetic test fixture
  models/                        # model weights (gitignored)
  src/
    damage_analysis/
    reconstruction/
    evidence_retrieval/
    candidate_ranking/
    evaluation/
    review/
  tests/
  outputs/
    predictions/
    visualizations/
    reports/
  docs/
  requirements.txt
  README.md
  .env.example
  .gitignore
```

---

## 3. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Tesseract not on PATH | Medium | Detect and show clear install guide in UI |
| No Brahmi OCR model exists | High | Label limitation explicitly; heuristic/stub with warning |
| BharatSHRI data not publicly downloadable | High | Documented import workflow + synthetic fixture |
| EasyOCR CUDA/CPU conflict | Medium | Wrap with exception; skip gracefully |
| LLM generates fabricated inscriptions | High | Strict prompts; separate [UNKNOWN] from inferred |
| FAISS install conflicts | Low | Make optional; fall back to pure-Python search |

---

## 4. Implementation Order

| Phase | Deliverable | Key Files |
|---|---|---|
| 1 | Inspection + this plan | docs/TRANSFORMATION_PLAN.md |
| 2 | Directory scaffold, config, .env.example, requirements | config/, requirements.txt, .env.example |
| 3 | Core pipeline (Stages A-D) | epirecover_pipeline.py, extended utils.py |
| 4 | Damage analysis, reconstruction, evidence retrieval (C, E, F) | src/damage_analysis/, src/reconstruction/, src/evidence_retrieval/ |
| 5 | Candidate ranking, human review (G, human) | src/candidate_ranking/, src/review/ |
| 6 | Full Streamlit dashboard | streamlit_app.py |
| 7 | Evaluation module + tests | src/evaluation/, tests/ |
| 8 | Final documentation | README.md, docs/ |

---

## 5. Research Integrity Commitments

- Damage masks produced by classical image analysis are labelled "heuristic baseline", never "deep-learning prediction".
- No training results or accuracy percentages will be stated unless a training run artifact exists.
- Every reconstruction candidate retains a provenance chain: observed | ocr_predicted | corpus_retrieved | model_inferred | human_reviewed.
- [UNKNOWN] is used for regions where no evidence is sufficient; the system never fills gaps silently.
- BharatSHRI fixture records in data/corpus/bharatshri_sample.jsonl are synthetic and clearly labelled as such.
- Ground-truth transcriptions used in evaluation are kept in data/annotations/ and excluded from corpus retrieval during evaluation.

---

*Document created: 2026-10-25 | EpiRecover v0.1 (pre-release)*
