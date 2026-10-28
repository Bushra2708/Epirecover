# EpiRecover — Root Cause Analysis: Repeated 0.48 Score & Static Corpus Output

**Project**: EpiRecover: Evidence-Grounded Multimodal Reconstruction of Damaged Ancient Indian Inscriptions from Monument Imagery  
**Document**: Diagnostic & Root Cause Report for Repeated Output Bug  
**Date**: October 2026  
**Scope**: Tamil Inscriptions (Modern Tamil & Historical Tamil-Brahmi)  
**Constraint**: Strictly Non-Generative (No Generative LLMs; Multimodal Evidence Fusion Only)

---

## 1. Bug Description & Symptom

When running the EpiRecover reconstruction pipeline via the React frontend or FastAPI backend on any input image:
- Several different candidates repeatedly appear with rank score **0.48**.
- Their evidence scores are identical: **$V = 0.40, O = 0.00, L = 1.00, H = 1.00, R = 0.50$**.
- Their provenance is tagged as **`CORPUS_RETRIEVED`**.
- The top-candidate score margin is **$0.000$** ($0.4800 - 0.4800 = 0.000$), triggering ambiguity abstention.
- The same candidate set (`TAML-001`, `TAML-002`, `TAML-005`) is returned regardless of the uploaded image.

---

## 2. Complete Request Tracing (Step 1)

Tracing one full execution cycle from image upload to frontend display:

1. **Frontend Request Initiation (`frontend/src/App.jsx`)**:
   - User selects an image file.
   - The form has hardcoded default state values:
     `inscription_id = "INS-TAMIL-001"`, `historical_period = "10th century CE (Chola)"`,
     `region = "Thanjavur, Tamil Nadu"`, `monument = "Brihadisvara Temple"`.
   - `handleProcess()` packages `file` and metadata into `FormData` and `POST`s to `/api/pipeline/process`.

2. **FastAPI Route (`src/api/main.py`)**:
   - `process_inscription()` decodes the uploaded image bytes via `cv2.imdecode(nparr, cv2.IMREAD_COLOR)`.
   - The decoded image is genuine and varies with the input file.
   - The image is saved to a temporary file `tmp_path` and passed to `pipeline.process(tmp_path, metadata=metadata)`.

3. **Preprocessing & Damage Detection (`epirecover_pipeline.py`)**:
   - `stage_b_preprocess(img)`: Runs OpenCV grayscale, bilateral filtering, illumination normalization, and CLAHE.
   - `stage_c_damage(img, cleaned)`: Runs heuristic damage detection, producing damage regions and mask.
   - *Audit finding*: These stages execute on the actual image, and their intermediate masks/regions vary with the input image.

4. **OCR Inference (`epirecover_pipeline.py` -> `ocr_engine.py`)**:
   - `stage_d_ocr()` calls `ocr_engine.extract_text(cleaned, script_hint=metadata.script)`.
   - In `ocr_engine.py`:
     - Tesseract is not installed in the Windows PATH (`_HAS_TESS = False`).
     - EasyOCR's Tamil model has a PyTorch class count mismatch (`size mismatch: shape [143, 512] vs [127, 512]`), so `_EASY_TAM_OK = False`.
     - Consequently, `extract_text()` returns `raw_text = ""`, `mean_confidence = 0.0`, `engine = "none-tamil-ocr-unavailable"`.
   - *Audit finding*: Because OCR detected 0 characters, the text track has zero image-derived characters.

5. **Corpus Evidence Retrieval (`epirecover_pipeline.py` -> `src/evidence_retrieval/retriever.py`)**:
   - Line 488: `query = mapped_text or ocr_result.raw_text`.
   - Since `raw_text == ""`, `query = ""`.
   - `stage_f_retrieve("", metadata)` calls `CorpusRetriever.retrieve(query="", metadata=metadata)`.
   - In `retriever.py` (lines 386–395): Because `query` is empty, it bypasses text retrieval and executes `_metadata_fallback_retrieve()`.
   - `_metadata_fallback_retrieve()` compares the request metadata (`Chola`, `Thanjavur`, `Brihadisvara`) against `data/corpus/tamil_inscriptions_sample.jsonl`.
   - In `tamil_inscriptions_sample.jsonl`, all records share those keywords, so each receives `meta_score = min(0.50, 1.0) = 0.50`.
   - Records `TAML-001`, `TAML-002`, and `TAML-005` are returned, all with `score = 0.50` and `retrieval_method = "metadata_fallback"`.

6. **Candidate Reconstruction (`src/reconstruction/candidate_generator.py`)**:
   - `generate()` receives `ocr_result` (empty) and `retrieved_records` (`TAML-001`, `TAML-002`, `TAML-005`).
   - Line 248: `valid_corpus = [r for r in retrieved_records if float(r.get("score", 0)) >= self.MIN_CORPUS_SCORE]`.
   - Because `MIN_CORPUS_SCORE = 0.35` and all 3 records have `score = 0.50`, all 3 are converted into `ReconstructionCandidate`s:
     - `CAND-CORP-01`: `"ஸ்வஸ்தி ஸ்ரீ கோப்பரகேசரிவன்மற்கு யாண்டு"` (`Provenance.CORPUS`)
     - `CAND-CORP-02`: `"திருமகள் போல பெருநிலச் செல்வியும்"` (`Provenance.CORPUS`)
     - `CAND-CORP-03`: `"கோவிராஜகேசரி வர்மரான உடையார் ஸ்ரீராஜராஜ தேவர்"` (`Provenance.CORPUS`)
   - Plus `CAND-UNK-01`: `[UNKNOWN]`.
   - *Audit finding*: Metadata-matched historical documents from the database were silently promoted as "reconstructed readings" of an image where 0 characters were observed!

7. **Evidence Feature Extraction (`src/candidate_ranking/fusion.py`)**:
   - `MultimodalFeatureExtractor.extract_features()` scores each candidate:
     - **Visual Evidence ($V$)**: Line 67 sets `prov_weights[Provenance.CORPUS] = 0.40`. Even though 0 characters were seen in the image, the candidate gets $V = 0.40$.
     - **OCR Evidence ($O$)**: Line 95 computes `ocr_confidence / 100 = 0.0`. So $O = 0.00$.
     - **Linguistic Evidence ($L$)**: Line 108 computes $0.3 + 0.7 \times \text{tamil\_ratio}$. Since corpus text is 100% Tamil Unicode, $L = 1.00$.
     - **Historical Evidence ($H$)**: Lines 111–134 loop over `retrieved_records[:3]` against `metadata` and compute a single number: $0.3 + 0.7 \times (2/2) = 1.00$. **This identical score is assigned to EVERY candidate**, completely ignoring individual candidate metadata.
     - **Retrieval Evidence ($R$)**: Line 141 looks up `rec["score"]`, which is `0.50` for all 3 records. So $R = 0.50$.

8. **Multimodal Evidence Fusion Formula**:
   Active weights in `config/epirecover.yaml`:
   $$w_V = 0.20,\; w_O = 0.25,\; w_L = 0.15,\; w_H = 0.10,\; w_R = 0.30$$
   For each of the three corpus candidates:
   $$S(c) = 0.20(0.40) + 0.25(0.00) + 0.15(1.00) + 0.10(1.00) + 0.30(0.50)$$
   $$S(c) = 0.0800 + 0.0000 + 0.1500 + 0.1000 + 0.1500 = \mathbf{0.4800}$$

9. **Ambiguity Abstention**:
   In `src/candidate_ranking/abstention.py`:
   - Top-1 candidate score: $0.4800$
   - Top-2 candidate score: $0.4800$
   - Score margin: $0.4800 - 0.4800 = \mathbf{0.000}$
   - Since $0.000 < 0.080$ (margin threshold), the engine rejects with reason:
     `"Margin 0.000 < threshold 0.080 between Top-1 and Top-2; ambiguous candidates."`

---

## 3. Confirmed Defects & Root Causes

| Defect ID | Component | Defect Description | Impact |
|---|---|---|---|
| **RC-1** | `candidate_generator.py` | Promoting metadata fallback corpus records as image reading candidates when OCR observed 0 characters | Fabricates full transcription readings for images with no detected text |
| **RC-2** | `fusion.py` | Global broadcast of $H$: compares `retrieved_records[:3]` to metadata instead of candidate-specific metadata | Destroys score variance; every candidate receives identical $H = 1.00$ |
| **RC-3** | `fusion.py` | Double-counting metadata fallback in $R$: assigning $R = 0.50$ when text retrieval similarity was zero | Fabricates retrieval evidence for candidates that have zero text overlap |
| **RC-4** | `fusion.py` | Static $V = 0.40$ for corpus candidates when no visual character overlap exists | Assigns visual evidence to unobserved text |
| **RC-5** | `App.jsx` | Static default form inputs (`Chola`, `Thanjavur`, `Brihadisvara`) | Binds every upload to identical metadata query |
| **RC-6** | `epirecover_pipeline.py` | Missing image hash and request tracing | Requests cannot be uniquely verified in diagnostic logs |

---

## 4. Fix Strategy (Preserving PPT Architecture & Non-LLM Scope)

1. **Grounded Candidate Generation**:
   - In `CandidateGenerator`: If OCR detected no readable characters (`raw_text` is empty or noise), corpus records retrieved via metadata MUST NOT be emitted as inscription reading candidates. They are reference documents, not image readings.
   - The primary candidate when OCR is empty must be `[UNKNOWN]` (honestly preserving uncertainty).
   - When OCR *does* detect partial text, corpus candidates are generated only if they corroborate or overlap the observed text.
2. **Candidate-Specific Historical Evidence ($H$)**:
   - In `fusion.py`: Evaluate $H$ by checking the *specific candidate's* record metadata against the query metadata, not a global broadcast across the top 3 retrieved records.
3. **Strict Text-Retrieval Scoring ($R$)**:
   - In `fusion.py`: $R$ reflects textual similarity between candidate text and OCR/query text. If a record was retrieved solely via metadata fallback (no text match), its text retrieval score $R$ is $0.0$.
4. **Visual Grounding ($V$)**:
   - For corpus candidates, $V$ must be proportional to the visual overlap between the candidate text and the OCR/observed strokes. If zero characters were observed in OCR, $V = 0.0$.
5. **Deterministic Request Tracing**:
   - Compute SHA-256 hash of the input image and assign a unique `request_id` logged across all pipeline stages.
