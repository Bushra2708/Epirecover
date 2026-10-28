"""
streamlit_app.py
-----------------
EpiRecover Research Dashboard.

Run:
    streamlit run streamlit_app.py

Sections:
  1. Project Overview
  2. Upload Inscription (Stage A)
  3. Image Preprocessing (Stage B)
  4. Damage Analysis (Stage C)
  5. OCR Results (Stage D)
  6. Candidate Reconstruction (Stage E)
  7. Evidence and Source Records (Stage F)
  8. Human Review (Stage G + review)
  9. Evaluation Dashboard (Stage H)
  10. Dataset and Experiment Management

Original upstream UI: sam-r-148/EpiQ (MIT licence preserved).
Full rewrite for EpiRecover research dashboard, 2026.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Core pipeline
import preprocessing
import ocr_engine
import dataset_matcher
import genai_translator
import utils
from utils import RecoveryReport, ReconstructionCandidate, Provenance, setup_logging, load_config
from epirecover_pipeline import EpiRecoverPipeline, InscriptionMetadata

# Source modules — imported gracefully
try:
    from src.damage_analysis.damage_detector import DamageDetector
    _HAS_DAMAGE = True
except ImportError:
    _HAS_DAMAGE = False

try:
    from src.evidence_retrieval.corpus_adapter import CorpusAdapter
    from src.evidence_retrieval.retriever import CorpusRetriever
    _HAS_RETRIEVAL = True
except ImportError:
    _HAS_RETRIEVAL = False

try:
    from src.evaluation.metrics import evaluate_single, aggregate_results, InscriptionResult
    _HAS_EVAL = True
except ImportError:
    _HAS_EVAL = False

try:
    from src.review.review_store import ReviewStore
    _HAS_REVIEW = True
except ImportError:
    _HAS_REVIEW = False

# ---------------------------------------------------------------------------
# Logging + page setup
# ---------------------------------------------------------------------------

setup_logging("INFO")
log = logging.getLogger("epirecover.dashboard")

st.set_page_config(
    page_title="EpiRecover — Inscription Reconstruction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean research aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header strip */
.epirecover-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 4px solid #e94560;
}
.epirecover-header h1 {
    color: #ffffff;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.epirecover-header p {
    color: #a0aec0;
    font-size: 0.9rem;
    margin: 0.4rem 0 0 0;
}

/* Section cards */
.section-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

/* Provenance badges */
.prov-observed { background: #d1fae5; color: #065f46; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.prov-ocr { background: #dbeafe; color: #1e40af; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.prov-corpus { background: #ede9fe; color: #5b21b6; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.prov-inferred { background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.prov-unknown { background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.prov-human { background: #d1fae5; color: #065f46; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }

/* Warning banner */
.research-warning {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #78350f;
    margin: 0.5rem 0;
}

/* Metric blocks */
.metric-block {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.9rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _pil_from_bgr(bgr: Optional[np.ndarray]) -> Optional[Image.Image]:
    if bgr is None:
        return None
    if len(bgr.shape) == 2:
        return Image.fromarray(bgr)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _bgr_from_upload(uploaded) -> np.ndarray:
    data = uploaded.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image. Is it a valid PNG/JPEG?")
    uploaded.seek(0)
    return img


def _provenance_badge(prov: str) -> str:
    cls_map = {
        "observed": "prov-observed",
        "ocr_predicted": "prov-ocr",
        "corpus_retrieved": "prov-corpus",
        "model_inferred": "prov-inferred",
        "unknown": "prov-unknown",
        "human_reviewed": "prov-human",
    }
    label_map = {
        "observed": "OBSERVED",
        "ocr_predicted": "OCR PREDICTED",
        "corpus_retrieved": "CORPUS RETRIEVED",
        "model_inferred": "MODEL INFERRED",
        "unknown": "UNKNOWN",
        "human_reviewed": "HUMAN REVIEWED",
    }
    cls = cls_map.get(prov, "prov-unknown")
    label = label_map.get(prov, prov.upper())
    return f'<span class="{cls}">{label}</span>'


@st.cache_resource(show_spinner=False)
def _get_pipeline(backend: str, tess_langs: str, easy_langs_key: tuple) -> EpiRecoverPipeline:
    cfg = load_config()
    return EpiRecoverPipeline(
        config=cfg,
        llm_backend=backend,
        tesseract_langs=tess_langs,
        easyocr_langs=list(easy_langs_key) if easy_langs_key else None,
    )


@st.cache_resource(show_spinner=False)
def _get_review_store() -> Optional[Any]:
    if not _HAS_REVIEW:
        return None
    return ReviewStore("outputs/reports/review_decisions.json")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### EpiRecover")
    st.caption("Evidence-Grounded Inscription Reconstruction")
    st.divider()

    st.subheader("Reconstruction Engine")
    st.info("Non-LLM Multimodal Fusion (V + O + L + H + R). Generative LLMs are strictly excluded per PPT specification.", icon=None)
    backend = "stub"

    st.divider()
    st.subheader("Tamil Scope & OCR Status")
    st.caption("Language scope: Tamil only (v0.1)")
    from ocr_engine import check_tamil_ocr_support
    ocr_status = check_tamil_ocr_support()

    if ocr_status["modern_tamil_supported"]:
        if ocr_status.get("tesseract_tam_pack"):
            st.success("✓ Tesseract + tam pack: OK")
        if ocr_status.get("easyocr_tam_model_ok"):
            st.success("✓ EasyOCR Tamil model: OK")
    else:
        st.error("⚠ Tamil OCR Unavailable — results will not reflect actual Tamil script")
        if ocr_status.get("tesseract_available") and not ocr_status.get("tesseract_tam_pack"):
            st.warning("Tesseract installed but 'tam.traineddata' missing")
        elif not ocr_status.get("tesseract_available"):
            st.warning("Tesseract not found in PATH")
        if ocr_status.get("easyocr_available") and not ocr_status.get("easyocr_tam_model_ok"):
            err = ocr_status.get("easyocr_tam_error", "")
            st.warning(f"EasyOCR Tamil model checkpoint error: {err[:120]}")
            st.caption("Fix: pip install easyocr==1.6.2  OR  replace cached ta.pth checkpoint")

    st.info("Tamil-Brahmi: Requires corpus retrieval / manual transcription (standard OCR unsupported)")

    tess_langs = st.text_input(
        "Tesseract languages",
        value="tam+eng",
        help="Configured for Tamil scope. Requires 'tam.traineddata' in tessdata.",
    )
    easy_langs = st.multiselect(
        "EasyOCR languages",
        options=["ta", "en"],
        default=["ta", "en"],
        help="Configured for Tamil scope.",
    )
    apply_perspective = st.toggle("Try perspective correction", value=False)
    save_intermediates = st.toggle("Save preprocessing stages to disk", value=False)

    st.divider()
    st.subheader("Display Options")
    show_preprocessing = st.toggle("Show all preprocessing stages", value=True)
    show_damage_overlay = st.toggle("Show damage overlay", value=True)
    show_ocr_boxes = st.toggle("Show OCR bounding boxes", value=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="epirecover-header">
  <h1>EpiRecover</h1>
  <p>Evidence-Grounded Multimodal Reconstruction of Damaged Ancient Indian Inscriptions from Monument Imagery</p>
</div>
""", unsafe_allow_html=True)

# Navigation tabs
tabs = st.tabs([
    "1. Overview",
    "2. Upload",
    "3. Preprocessing",
    "4. Damage Analysis",
    "5. OCR Results",
    "6. Reconstruction",
    "7. Evidence",
    "8. Human Review",
    "9. Evaluation",
    "10. Dataset",
])

# ===========================================================================
# Tab 1 — Project Overview
# ===========================================================================

with tabs[0]:
    st.header("Project Overview")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("""
**EpiRecover** is a research prototype for the evidence-grounded reconstruction of damaged
ancient Indian inscriptions from monument photographs.

**Research objectives:**
- Process photographs of stone inscriptions (Brahmi, Tamil-Brahmi, Sanskrit, Prakrit, Pali, Devanagari)
- Improve image readability using classical computer vision
- Detect and localize damaged, eroded, or obscured character regions
- Extract visible characters using OCR engines
- Generate candidate readings backed by visual and corpus evidence
- Rank candidates using transparent, auditable scoring
- Retrieve supporting records from inscription corpora (BharatSHRI and others)
- Allow researchers to review, accept, edit, or reject proposed readings
- Evaluate system performance with Character Error Rate and Word Error Rate

**Pipeline stages:**

| Stage | Description |
|---|---|
| A | Image upload and metadata collection |
| B | OpenCV preprocessing (CLAHE, adaptive threshold, morphology) |
| C | Damage localization (heuristic baseline; U-Net plug-in point) |
| D | OCR ensemble (Tesseract + EasyOCR) with uncertainty markers |
| E | Structured candidate reconstruction with provenance |
| F | Corpus evidence retrieval (BharatSHRI, CSV, JSONL) |
| G | Evidence-grounded candidate ranking |
| H | CER/WER evaluation with aggregate reporting |
        """)

    with col_b:
        st.markdown("""
**Research integrity notice:**

This system is an academic prototype. It:
- Distinguishes observed, predicted, retrieved, and inferred evidence
- Labels heuristic outputs as baselines, not model predictions
- Uses [UNKNOWN] for insufficient evidence
- Does not fabricate citations or historical facts
- Reports metrics as unavailable when ground truth is missing
        """)
        st.markdown('<div class="research-warning">Reconstruction candidates are hypotheses, not established historical facts. All outputs require expert review.</div>', unsafe_allow_html=True)

    st.subheader("Module Status")
    status_data = {
        "Module": ["preprocessing.py", "ocr_engine.py", "dataset_matcher.py",
                   "genai_translator.py", "src/damage_analysis", "src/reconstruction",
                   "src/evidence_retrieval", "src/candidate_ranking", "src/evaluation",
                   "src/review"],
        "Status": ["Functional", "Functional (no Brahmi model)",
                   "Functional (seed dataset)", "Functional (stub offline)",
                   "Implemented (heuristic baseline)" if _HAS_DAMAGE else "Import error",
                   "Implemented" if True else "Import error",
                   "Implemented (synthetic fixture)" if _HAS_RETRIEVAL else "Import error",
                   "Implemented (heuristic weights)" if True else "Import error",
                   "Implemented (CER/WER)" if _HAS_EVAL else "Import error",
                   "Implemented" if _HAS_REVIEW else "Import error"],
        "API Key Required": ["No","No","No","Only for Gemini/Groq",
                             "No","No","No","No","No","No"],
    }
    st.dataframe(status_data, use_container_width=True)


# ===========================================================================
# Tab 2 — Upload Inscription
# ===========================================================================

with tabs[1]:
    st.header("Upload Inscription")
    st.caption("Stage A: Image upload and metadata collection")

    st.markdown('<div class="research-warning">The original image is never modified. Preprocessing creates separate copies.</div>', unsafe_allow_html=True)

    col_up, col_meta = st.columns([1, 1])

    with col_up:
        st.subheader("Image")
        uploaded = st.file_uploader(
            "Upload an inscription photograph",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
            key="main_upload",
        )
        if uploaded:
            try:
                img_bgr = _bgr_from_upload(uploaded)
                h, w = img_bgr.shape[:2]
                st.image(_pil_from_bgr(img_bgr), caption=f"Uploaded: {uploaded.name} ({w}x{h}px)", use_column_width=True)
                st.session_state["img_bgr"] = img_bgr
                st.session_state["img_name"] = uploaded.name
                img_id = utils.image_hash(img_bgr)
                st.session_state["img_id"] = img_id
                st.success(f"Image loaded. ID: {img_id}")
            except Exception as e:
                st.error(f"Image decode failed: {e}")

    with col_meta:
        st.subheader("Metadata (Tamil Scope)")
        inscription_id = st.text_input("Inscription / Record ID", placeholder="e.g. ASI-TN-TAML-001")
        
        language_val = st.text_input(
            "Target Language",
            value="Tamil",
            disabled=True,
            help="Current version (v0.1) is scoped exclusively for Tamil inscriptions.",
        )
        
        script_options = ["tamil", "tamil-brahmi", "uncertain"]
        script_labels = {
            "tamil": "Modern Tamil Script (தமிழ் எழுத்து)",
            "tamil-brahmi": "Tamil-Brahmi (தமிழி) — Historical Lithic Script",
            "uncertain": "(Uncertain — Manual classification required)",
        }
        script_choice = st.selectbox(
            "Writing System / Script",
            script_options,
            format_func=lambda x: script_labels[x],
            help="Record the script separately from language. Do not assume ancient inscriptions are modern Tamil.",
        )
        
        if script_choice == "tamil-brahmi":
            st.warning(
                "Historical Script Notice: 'Tamil-Brahmi' (தமிழி) is an ancient writing system (3rd c. BCE - 5th c. CE). "
                "Standard OCR engines (Tesseract / EasyOCR) do not support automatic recognition of ancient Tamil-Brahmi glyphs. "
                "Candidate readings will rely on epigraphic corpus records and manual review. "
                "Do not assume ancient Tamil-Brahmi is identical to modern Tamil."
            )
        elif script_choice == "uncertain":
            st.warning(
                "Script is marked uncertain. Please determine whether the inscription is in Modern Tamil script or historical Tamil-Brahmi. "
                "EpiRecover does not silently classify inscriptions."
            )

        historical_period = st.text_input("Historical period", placeholder="e.g. 10th century CE (Chola) / 2nd c. BCE")
        region = st.text_input("Region / Location", placeholder="e.g. Thanjavur / Madurai, Tamil Nadu")
        monument = st.text_input("Monument / Site", placeholder="e.g. Brihadisvara Temple / Mangulam Cave")
        dataset_source = st.text_input("Source / Citation", placeholder="e.g. South Indian Inscriptions / Early Tamil Epigraphy")
        researcher_notes = st.text_area("Notes", placeholder="Observations regarding stone weathering, surface condition, or fracture")

        if st.button("Save metadata", type="secondary"):
            meta = InscriptionMetadata(
                inscription_id=inscription_id or None,
                script=script_choice if script_choice != "uncertain" else None,
                language="Tamil",
                historical_period=historical_period or None,
                region=region or None,
                monument=monument or None,
                dataset_source=dataset_source or None,
                researcher_notes=researcher_notes or None,
            )
            st.session_state["metadata"] = meta
            st.success("Metadata saved for Tamil inscription.")


# ===========================================================================
# Tab 3 — Image Preprocessing
# ===========================================================================

with tabs[2]:
    st.header("Image Preprocessing")
    st.caption("Stage B: OpenCV preprocessing pipeline")

    if "img_bgr" not in st.session_state:
        st.info("Upload an image in tab 2 first.")
    else:
        img_bgr = st.session_state["img_bgr"]
        with st.spinner("Running preprocessing pipeline..."):
            try:
                stages = preprocessing.preprocess_inscription(
                    img_bgr,
                    apply_perspective=apply_perspective,
                    return_intermediates=True,
                )
                cleaned = stages["cleaned"]
                st.session_state["cleaned"] = cleaned
                st.session_state["stages"] = stages
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                stages, cleaned = {}, None

        if cleaned is not None:
            # Side-by-side comparison
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption("Original")
                st.image(_pil_from_bgr(img_bgr), use_column_width=True)
            with c2:
                st.caption("Preprocessed (binary)")
                st.image(_pil_from_bgr(cleaned), use_column_width=True, clamp=True)
            with c3:
                st.caption("Edge map")
                st.image(_pil_from_bgr(stages.get("edges")), use_column_width=True, clamp=True)

            st.markdown(f"**Stages applied:** {', '.join(stages.keys())}")

            if show_preprocessing:
                with st.expander("All preprocessing stages", expanded=False):
                    cols = st.columns(4)
                    for i, (name, arr) in enumerate(stages.items()):
                        with cols[i % 4]:
                            st.caption(name)
                            st.image(_pil_from_bgr(arr), use_column_width=True, clamp=True)


# ===========================================================================
# Tab 4 — Damage Analysis
# ===========================================================================

with tabs[3]:
    st.header("Damage Analysis")
    st.caption("Stage C: Damage localization and uncertainty")
    st.markdown('<div class="research-warning">Current method: HEURISTIC BASELINE (classical image analysis). This is NOT a trained deep-learning model prediction. The U-Net integration point is documented in src/damage_analysis/damage_detector.py.</div>', unsafe_allow_html=True)

    if "cleaned" not in st.session_state:
        st.info("Complete preprocessing (tab 3) first.")
    else:
        img_bgr = st.session_state["img_bgr"]
        cleaned = st.session_state["cleaned"]

        if not _HAS_DAMAGE:
            st.warning("Damage analysis module not available. Check src/damage_analysis/damage_detector.py.")
        else:
            with st.spinner("Running heuristic damage analysis..."):
                try:
                    cfg = load_config()
                    detector = DamageDetector(cfg)
                    damage_result = detector.analyze(img_bgr, cleaned)
                    st.session_state["damage_result"] = damage_result
                except Exception as e:
                    st.error(f"Damage analysis failed: {e}")
                    damage_result = {"method": "error", "label": str(e), "regions": [], "mask": None, "overlay": None}

            d1, d2 = st.columns(2)
            with d1:
                st.caption("Original")
                st.image(_pil_from_bgr(img_bgr), use_column_width=True)
            with d2:
                overlay = damage_result.get("overlay")
                if overlay is not None and show_damage_overlay:
                    st.caption(f"Damage overlay [{damage_result.get('label', '')}]")
                    st.image(_pil_from_bgr(overlay), use_column_width=True)
                else:
                    mask = damage_result.get("mask")
                    if mask is not None:
                        st.caption("Damage mask")
                        st.image(_pil_from_bgr(mask), use_column_width=True, clamp=True)

            regions = damage_result.get("regions", [])
            st.metric("Candidate damage regions", len(regions))
            if regions:
                st.dataframe(regions, use_container_width=True)


# ===========================================================================
# Tab 5 — OCR Results
# ===========================================================================

with tabs[4]:
    st.header("OCR Results")
    st.caption("Stage D: Multi-engine OCR with uncertainty markers")

    if "cleaned" not in st.session_state:
        st.info("Complete preprocessing (tab 3) first.")
    else:
        cleaned = st.session_state["cleaned"]
        img_bgr = st.session_state["img_bgr"]

        run_ocr = st.button("Run OCR", type="primary")
        if run_ocr or "ocr_result" in st.session_state:
            if run_ocr:
                with st.spinner("Running OCR ensemble..."):
                    try:
                        pipe = _get_pipeline(backend, tess_langs, tuple(easy_langs))
                        meta = st.session_state.get("metadata")
                        script_h = meta.script if meta else "tamil"
                        damage_res = st.session_state.get("damage_result", {})
                        ocr_result, illegible = pipe.stage_d_ocr(
                            cleaned,
                            damage_regions=damage_res.get("regions"),
                            script_hint=script_h,
                        )
                        st.session_state["ocr_result"] = ocr_result
                        st.session_state["illegible"] = illegible
                    except Exception as e:
                        st.error(f"OCR failed: {e}")
                        ocr_result = None

            if "ocr_result" in st.session_state:
                ocr_result = st.session_state["ocr_result"]
                illegible = st.session_state.get("illegible", [])

                if ocr_result.warning_note:
                    st.warning(ocr_result.warning_note)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("OCR engine", ocr_result.engine or "none")
                m2.metric("Mean confidence", f"{ocr_result.mean_confidence:.1f}")
                m3.metric("Tokens detected", len(ocr_result.boxes))
                m4.metric("Illegible regions", len(illegible))

                if show_ocr_boxes and ocr_result.boxes:
                    vis = ocr_engine.draw_boxes(img_bgr, ocr_result)
                    st.image(_pil_from_bgr(vis), caption="OCR bounding boxes", use_column_width=True)

                tab_raw, tab_boxes, tab_illegible = st.tabs(["Raw text", "Token details", "Illegible regions"])
                with tab_raw:
                    if ocr_result.raw_text:
                        st.code(ocr_result.raw_text, language="text")
                    else:
                        st.info("No raw text produced by OCR engine for this image/script.")
                with tab_boxes:
                    if ocr_result.boxes:
                        rows = [{"text": b.text, "confidence": round(b.confidence, 1),
                                 "x": b.bbox[0], "y": b.bbox[1], "w": b.bbox[2], "h": b.bbox[3]}
                                for b in ocr_result.boxes]
                        st.dataframe(rows, use_container_width=True)
                    else:
                        st.info("No token-level bounding boxes available.")
                with tab_illegible:
                    if illegible:
                        st.caption("Regions with OCR confidence below threshold:")
                        for marker in illegible:
                            st.code(marker, language="text")
                    else:
                        st.info("No illegible markers generated.")


# ===========================================================================
# Tab 6 — Candidate Reconstruction
# ===========================================================================

with tabs[5]:
    st.header("Candidate Reconstruction")
    st.caption("Stage E: Evidence-grounded reconstruction candidates")
    st.markdown('<div class="research-warning">Candidates are hypotheses. Provenance labels distinguish observed text from OCR output, corpus evidence, and model inference. [UNKNOWN] is returned when evidence is insufficient.</div>', unsafe_allow_html=True)

    if "ocr_result" not in st.session_state:
        st.info("Run OCR (tab 5) first.")
    else:
        ocr_result = st.session_state["ocr_result"]
        metadata = st.session_state.get("metadata")

        run_recon = st.button("Generate candidates", type="primary")
        if run_recon or "candidates" in st.session_state:
            if run_recon:
                with st.spinner("Running reconstruction pipeline..."):
                    try:
                        pipe = _get_pipeline(backend, tess_langs, tuple(easy_langs))
                        # Dictionary mapping
                        mapped_text = ""
                        if pipe._mapping and ocr_result.raw_text:
                            tokens = ocr_result.raw_text.split()
                            mapped_text, _ = pipe._mapping.translate_tokens(tokens)
                        st.session_state["mapped_text"] = mapped_text

                        # Evidence retrieval
                        retrieved = pipe.stage_f_retrieve(
                            mapped_text or ocr_result.raw_text,
                            metadata,
                        )
                        st.session_state["retrieved"] = retrieved

                        # Reconstruction
                        damage_res = st.session_state.get("damage_result", {})
                        candidates = pipe.stage_e_reconstruct(
                            ocr_result,
                            mapped_text,
                            metadata,
                            retrieved,
                            damage_regions=damage_res.get("regions", []),
                        )
                        # Ranking
                        ranked, abstention = pipe.stage_g_rank(
                            candidates, ocr_result, retrieved, metadata,
                            damage_regions=damage_res.get("regions", []),
                        )
                        st.session_state["candidates"] = ranked
                        st.session_state["abstention"] = abstention

                        # Full LLM output for English/Hindi
                        script_h = metadata.script if metadata else None
                        llm_out = pipe._translator.translate(
                            ocr_result.raw_text, mapped_text, script_hint=script_h
                        )
                        st.session_state["llm_out"] = llm_out
                    except Exception as e:
                        st.error(f"Reconstruction failed: {e}")
                        import traceback; st.text(traceback.format_exc())

            if "candidates" in st.session_state:
                candidates = st.session_state["candidates"]
                llm_out = st.session_state.get("llm_out")
                mapped_text = st.session_state.get("mapped_text", "")

                st.subheader("Dictionary mapping")
                st.code(mapped_text or "(no mapping)", language="text")

                st.subheader(f"Reconstruction candidates ({len(candidates)})")

                # OCR fallback banner — shown when Tamil OCR was unavailable
                ocr_st = check_tamil_ocr_support()
                if not ocr_st.get("modern_tamil_supported"):
                    st.error(
                        "⚠ Tamil OCR unavailable — candidates below are based on CORPUS and METADATA "
                        "retrieval only, NOT on OCR character recognition. "
                        "Install Tesseract + tam.traineddata or downgrade easyocr to 1.6.2 to enable OCR."
                    )

                for i, cand in enumerate(candidates):
                    # Compute Tamil Unicode ratio for transparency
                    cand_text = cand.candidate_text or ""
                    stripped = cand_text.replace(" ", "")
                    tamil_cnt = sum(1 for c in stripped if "\u0b80" <= c <= "\u0bff") if stripped else 0
                    tamil_ratio = (tamil_cnt / len(stripped)) if stripped else 0.0
                    script_ok = tamil_ratio >= 0.30 or cand_text == "[UNKNOWN]"

                    label_suffix = " ✓" if script_ok and cand_text != "[UNKNOWN]" else " ⚠ non-Tamil" if not script_ok else ""
                    with st.expander(
                        f"Candidate {i+1}: {cand_text[:80]}...{label_suffix}",
                        expanded=(i == 0),
                    ):
                        st.markdown(
                            _provenance_badge(cand.provenance.value if hasattr(cand.provenance, 'value') else cand.provenance),
                            unsafe_allow_html=True,
                        )
                        st.code(cand_text, language="text")

                        # Script content warning
                        if not script_ok and cand_text != "[UNKNOWN]":
                            st.warning(
                                f"⚠ Tamil Unicode content: {tamil_ratio:.0%} — this text is not Tamil script. "
                                "Likely produced by a broken EasyOCR Tamil model (English fallback). "
                                "Do not interpret as an inscription reading."
                            )

                        # Score metrics: raw heuristic confidence vs calibrated rank score
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Heuristic confidence", f"{cand.confidence:.3f}")
                        rank_s = cand.rank_score if cand.rank_score is not None else 0.0
                        c2.metric("Fused rank score S(c)", f"{rank_s:.4f}")
                        c3.metric("Tamil Unicode ratio", f"{tamil_ratio:.0%}")

                        st.caption(f"Confidence note: {cand.confidence_note}")
                        if cand.corpus_evidence:
                            st.caption(f"Corpus source: {cand.corpus_evidence}")
                        if cand.uncertainty_reasons:
                            st.warning("Uncertainty: " + "; ".join(cand.uncertainty_reasons))

                        # V/O/L/H/R breakdown
                        if cand.rank_details:
                            v = cand.rank_details.get("V", 0.0)
                            o = cand.rank_details.get("O", 0.0)
                            l = cand.rank_details.get("L", 0.0)
                            h = cand.rank_details.get("H", 0.0)
                            r = cand.rank_details.get("R", 0.0)
                            with st.expander("Evidence score breakdown (V + O + L + H + R)", expanded=False):
                                cols = st.columns(5)
                                cols[0].metric("V (Visual)", f"{v:.3f}")
                                cols[1].metric("O (OCR)", f"{o:.3f}")
                                cols[2].metric("L (Linguistic)", f"{l:.3f}")
                                cols[3].metric("H (Historical)", f"{h:.3f}")
                                cols[4].metric("R (Retrieval)", f"{r:.3f}")
                                if r == 0.0:
                                    st.caption(
                                        "R = 0: No supporting corpus match found. "
                                        "Possible causes: (1) OCR query was non-Tamil text — text similarity against "
                                        "Tamil Unicode corpus records returned 0; (2) no metadata (period/region) was "
                                        "supplied for fallback retrieval; (3) corpus is small (5 synthetic fixtures only)."
                                    )
                                if l <= 0.30:
                                    st.caption(
                                        f"L = {l:.2f} (≤ 0.30): Candidate text contains less than 30% Tamil Unicode characters. "
                                        "Neutral baseline applies. Provide correct Tamil script text to improve this score."
                                    )
                                st.caption(
                                    "Weights: V=0.20, O=0.25, L=0.15, H=0.15, R=0.25 "
                                    "(heuristic defaults — not calibrated on labelled pairs)."
                                )

                if llm_out:
                    st.subheader("LLM translation output")
                    tab_eng, tab_hin, tab_notes = st.tabs(["English", "Hindi", "Notes"])
                    with tab_eng:
                        st.markdown(f"> {llm_out.english}" if llm_out.english else "(no output)")
                        for alt in llm_out.alternatives:
                            st.markdown(f"- Alternative: {alt}")
                    with tab_hin:
                        st.markdown(f"> {llm_out.hindi}" if llm_out.hindi else "(no output)")
                    with tab_notes:
                        st.write(llm_out.notes or "(no notes)")
                        st.caption(f"Backend: {llm_out.backend} | Self-reported confidence: {llm_out.confidence:.2f}")
                        st.markdown('<div class="research-warning">Self-reported LLM confidence is not a calibrated probability.</div>', unsafe_allow_html=True)


# ===========================================================================
# Tab 7 — Evidence and Source Records
# ===========================================================================

with tabs[6]:
    st.header("Evidence and Source Records")
    st.caption("Stage F: Corpus evidence retrieval")

    if not _HAS_RETRIEVAL:
        st.warning("Evidence retrieval module not available.")
    else:
        cfg = load_config()
        corpus_dir = os.path.join(
            cfg.get("paths", {}).get("data_root", "data"), "corpus"
        )

        @st.cache_resource(show_spinner=False)
        def _get_retriever(corpus_dir_key: str):
            return CorpusRetriever(corpus_dir=corpus_dir_key, cfg=load_config())

        retriever = _get_retriever(corpus_dir)
        adapter = retriever._adapter

        st.metric("Corpus records loaded", adapter.count)
        if adapter.count == 0:
            st.info("No corpus records loaded. Place JSONL/JSON/CSV files in data/corpus/. A synthetic fixture is included.")

        retrieved = st.session_state.get("retrieved", [])
        if retrieved:
            st.subheader(f"Retrieved evidence ({len(retrieved)} records)")
            for rec in retrieved:
                with st.expander(f"{rec.get('id', '?')} — score={rec.get('score', 0):.3f}", expanded=False):
                    st.markdown(f"**Script:** {rec.get('script', '—')}  |  **Language:** {rec.get('language', '—')}")
                    st.markdown(f"**Period:** {rec.get('historical_period', '—')}  |  **Region:** {rec.get('region', '—')}")
                    st.markdown(f"**Monument:** {rec.get('monument', '—')}")
                    st.code(rec.get('transcription', '(no transcription)'), language="text")
                    st.caption(f"Source: {rec.get('source', '—')}")
                    if rec.get("synthetic") == "true":
                        st.markdown('<div class="research-warning">SYNTHETIC FIXTURE — not a real inscription record. Do not cite.</div>', unsafe_allow_html=True)
        else:
            st.info("Run reconstruction (tab 6) to retrieve supporting evidence.")

        st.subheader("Manual corpus query")
        manual_query = st.text_input("Search corpus", placeholder="e.g. piyadasi dhamma brahmi")
        if manual_query and st.button("Search"):
            results = retriever.retrieve(manual_query, top_k=5)
            if results:
                for rec in results:
                    st.write(f"**{rec.get('id')}** (score={rec.get('score', 0):.3f}): {rec.get('transcription', '—')}")
            else:
                st.info("No matching records found.")


# ===========================================================================
# Tab 8 — Human Review
# ===========================================================================

with tabs[7]:
    st.header("Human Review")
    st.caption("Accept, edit, or reject reconstruction candidates")

    review_store = _get_review_store()

    if "candidates" not in st.session_state:
        st.info("Generate candidates (tab 6) first.")
    elif not _HAS_REVIEW:
        st.warning("Review module not available.")
    else:
        candidates = st.session_state["candidates"]
        img_id = st.session_state.get("img_id", "unknown")
        img_name = st.session_state.get("img_name", "unknown")
        metadata = st.session_state.get("metadata")
        insc_id = (metadata.inscription_id if metadata and metadata.inscription_id else img_name)

        st.subheader("Select a candidate to review")
        cand_labels = [f"Candidate {i+1}: {c.candidate_text[:60]}..." for i, c in enumerate(candidates)]
        selected_idx = st.selectbox("Candidate", range(len(cand_labels)), format_func=lambda i: cand_labels[i])
        selected_cand = candidates[selected_idx]

        st.code(selected_cand.candidate_text, language="text")
        st.markdown(
            _provenance_badge(selected_cand.provenance.value if hasattr(selected_cand.provenance, 'value') else selected_cand.provenance),
            unsafe_allow_html=True,
        )

        decision = st.radio("Decision", ["accepted", "rejected", "edited"], horizontal=True)
        reviewed_text = selected_cand.candidate_text
        if decision == "edited":
            reviewed_text = st.text_area("Edited text", value=selected_cand.candidate_text)
        reviewer_id = st.text_input("Reviewer ID (optional)")
        review_notes = st.text_area("Review notes (optional)")

        if st.button("Submit review", type="primary"):
            try:
                review_store.add_decision(
                    inscription_id=insc_id,
                    image_id=img_id,
                    candidate_text=selected_cand.candidate_text,
                    decision=decision,
                    reviewed_text=reviewed_text if decision != "rejected" else "",
                    reviewer_id=reviewer_id or None,
                    notes=review_notes or None,
                )
                st.success(f"Review decision '{decision}' recorded.")
            except Exception as e:
                st.error(f"Could not save review: {e}")

        st.subheader("Review history")
        past = review_store.get_decisions(insc_id) if review_store else []
        if past:
            st.dataframe(past, use_container_width=True)
            csv_buf = io.StringIO()
            import csv as _csv
            writer = _csv.DictWriter(csv_buf, fieldnames=list(past[0].keys()))
            writer.writeheader()
            writer.writerows(past)
            st.download_button(
                "Download review decisions (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"review_{insc_id}.csv",
                mime="text/csv",
            )
        else:
            st.info("No review decisions recorded yet.")


# ===========================================================================
# Tab 9 — Evaluation Dashboard
# ===========================================================================

with tabs[8]:
    st.header("Evaluation Dashboard")
    st.caption("Stage H: Reproducible evaluation with CER and WER")
    st.markdown('<div class="research-warning">Metrics are only computed when verified ground-truth transcriptions are available in data/annotations/. If unavailable, the metric is reported as unavailable.</div>', unsafe_allow_html=True)

    if not _HAS_EVAL:
        st.warning("Evaluation module not available. Install jiwer: pip install jiwer")
    else:
        ocr_result = st.session_state.get("ocr_result")
        candidates = st.session_state.get("candidates", [])
        metadata = st.session_state.get("metadata")
        img_name = st.session_state.get("img_name", "unknown")
        insc_id = (metadata.inscription_id if metadata and metadata.inscription_id else img_name)

        st.subheader("Ground-truth reference")
        ref_text = st.text_area(
            "Paste a verified ground-truth transcription (for evaluation only)",
            placeholder="This must be independently verified. Do not enter a generated candidate.",
            help="Source: data/annotations/ or a trusted epigraphic reference.",
        )
        hyp_text = candidates[0].candidate_text if candidates else ""
        if ocr_result:
            st.caption(f"OCR baseline text: {ocr_result.raw_text[:120]}")

        if st.button("Compute metrics", type="primary"):
            if not ref_text.strip():
                st.warning("Please enter a reference transcription to compute metrics.")
            elif not (hyp_text or (ocr_result and ocr_result.raw_text)):
                st.warning("No hypothesis text available. Run OCR and reconstruction first.")
            else:
                result = evaluate_single(
                    inscription_id=insc_id,
                    hypothesis=hyp_text,
                    reference=ref_text,
                    ocr_raw=ocr_result.raw_text if ocr_result else None,
                    script=metadata.script if metadata else None,
                    damage_level="unknown",
                )
                m1, m2, m3, m4 = st.columns(4)
                if result.cer is not None:
                    m1.metric("CER (reconstruction)", f"{result.cer:.4f}")
                else:
                    m1.metric("CER", "unavailable")
                if result.wer is not None:
                    m2.metric("WER (reconstruction)", f"{result.wer:.4f}")
                else:
                    m2.metric("WER", "unavailable")
                if result.ocr_cer is not None:
                    m3.metric("CER (OCR baseline)", f"{result.ocr_cer:.4f}")
                else:
                    m3.metric("CER (OCR baseline)", "unavailable")
                if result.ocr_wer is not None:
                    m4.metric("WER (OCR baseline)", f"{result.ocr_wer:.4f}")
                else:
                    m4.metric("WER (OCR baseline)", "unavailable")

                st.caption("CER = Character Error Rate | WER = Word Error Rate. Lower is better.")
                if result.metric_unavailable_reason:
                    st.info(result.metric_unavailable_reason)

        st.subheader("Export results")
        if "candidates" in st.session_state and st.session_state.get("img_id"):
            report_dict = {
                "inscription_id": insc_id,
                "image_id": st.session_state.get("img_id"),
                "ocr_text": ocr_result.raw_text if ocr_result else "",
                "mapped_text": st.session_state.get("mapped_text", ""),
                "candidates": [
                    c.as_dict() for c in candidates
                ] if candidates else [],
                "retrieved_records": st.session_state.get("retrieved", []),
            }
            st.download_button(
                "Download full report as JSON",
                data=json.dumps(report_dict, ensure_ascii=False, indent=2),
                file_name=f"epirecover_{insc_id}.json",
                mime="application/json",
            )


# ===========================================================================
# Tab 10 — Dataset and Experiment Management
# ===========================================================================

with tabs[9]:
    st.header("Dataset and Experiment Management")

    st.subheader("Corpus status")
    if _HAS_RETRIEVAL:
        cfg = load_config()
        corpus_dir = os.path.join(cfg.get("paths", {}).get("data_root", "data"), "corpus")
        adapter = CorpusAdapter()
        n = adapter.load_directory(corpus_dir)
        st.metric("Total corpus records", n)
        if n > 0:
            synthetic_count = sum(1 for r in adapter.records if r.get("synthetic") == "true")
            st.metric("Synthetic (fixture) records", synthetic_count)
            real_count = n - synthetic_count
            st.metric("Real records", real_count)
            if real_count == 0:
                st.info("Only synthetic fixture records are present. Import real BharatSHRI data by following docs/DATASET_GUIDE.md.")

    st.subheader("BharatSHRI import workflow")
    st.markdown("""
To import BharatSHRI or other inscription data:

1. Obtain the dataset files from your institution (CSV, JSON, or JSONL).
2. Place them in **data/corpus/**.
3. Each file must contain at minimum an `id` field.
4. Supported additional fields: `script`, `language`, `transcription`, `iast`, `historical_period`, `region`, `monument`, `source`, `notes`.
5. The corpus adapter auto-loads all files in that directory on startup.
6. Evaluation annotation files (ground-truth transcriptions) belong in **data/annotations/**, not in the corpus.

See **docs/DATASET_GUIDE.md** for the full import workflow.
    """)

    st.subheader("Annotation files")
    ann_dir = "data/annotations"
    if os.path.isdir(ann_dir):
        ann_files = [f for f in os.listdir(ann_dir) if not f.startswith(".")]
        st.metric("Annotation files", len(ann_files))
        if ann_files:
            for f in ann_files:
                st.write(f"- {f}")
    else:
        st.info("data/annotations/ not found. Create it and add ground-truth transcriptions as <inscription_id>.txt files.")

    st.subheader("Quick links")
    st.markdown("""
- [TRANSFORMATION_PLAN.md](docs/TRANSFORMATION_PLAN.md)
- [DATASET_GUIDE.md](docs/DATASET_GUIDE.md)
- [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)
- [IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md)
    """)
