"""
epirecover_pipeline.py
----------------------
Main orchestrator for the EpiRecover research pipeline.

Extends the upstream EpiQ InscriptionTranslator with:
  - Stage A : image upload + metadata collection
  - Stage B : image preprocessing (reuses preprocessing.py)
  - Stage C : damage localization (src/damage_analysis/)
  - Stage D : OCR with uncertainty markers (reuses ocr_engine.py)
  - Stage E : candidate reconstruction (src/reconstruction/)
  - Stage F : corpus evidence retrieval (src/evidence_retrieval/)
  - Stage G : evidence-grounded candidate ranking (src/candidate_ranking/)

CLI usage:
    python epirecover_pipeline.py path/to/inscription.jpg \
        --script-hint brahmi \
        --backend stub \
        --out outputs/predictions/

Programmatic:
    from epirecover_pipeline import EpiRecoverPipeline, InscriptionMetadata
    pipe = EpiRecoverPipeline()
    report = pipe.process("inscription.jpg", metadata=InscriptionMetadata(script="brahmi"))
    print(report.to_json())

Original upstream code: sam-r-148/EpiQ (MIT licence preserved)
EpiRecover extensions, 2026.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Ensure UTF-8 output on Windows for Indic scripts and Unicode epigraphy
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np

import preprocessing
import ocr_engine
import dataset_matcher
import genai_translator
from utils import (
    RecoveryReport,
    ReconstructionCandidate,
    Provenance,
    PipelineReport,
    load_image,
    save_image,
    image_hash,
    setup_logging,
    load_config,
)

log = logging.getLogger("epirecover")


# ---------------------------------------------------------------------------
# Metadata container (Stage A)
# ---------------------------------------------------------------------------

@dataclass
class InscriptionMetadata:
    """
    Metadata supplied by the researcher at upload time.
    Scoped for Tamil-language inscriptions in current version (v0.1).
    """
    inscription_id: Optional[str] = None
    script: Optional[str] = "tamil"        # 'tamil' (modern) | 'tamil-brahmi' (historical)
    language: Optional[str] = "Tamil"      # Fixed to Tamil for this version
    historical_period: Optional[str] = None
    region: Optional[str] = None
    monument: Optional[str] = None
    dataset_source: Optional[str] = None
    researcher_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class EpiRecoverPipeline:
    """
    End-to-end EpiRecover pipeline (Tamil Focus).

    Each stage is a separate method so it can be called independently,
    replaced in subclasses, or tested in isolation.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        data_dir: Optional[str] = None,
        llm_backend: str = "auto",
        tesseract_langs: str = "tam+eng",
        easyocr_langs: Optional[List[str]] = None,
        unknown_marker: str = "[UNKNOWN]",
        uncertainty_threshold: float = 30.0,
    ) -> None:
        self._cfg = config or load_config()
        self._data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self._unknown_marker = unknown_marker
        self._uncertainty_threshold = uncertainty_threshold

        # Load the script-mapping dataset
        try:
            self._mapping = dataset_matcher.load_default_dataset(self._data_dir)
        except FileNotFoundError as e:
            log.warning("Script mapping dataset not found: %s — dictionary matching disabled.", e)
            self._mapping = None

        # Build LLM translator
        self._translator = genai_translator.make_translator(llm_backend)

        # Tesseract / EasyOCR language settings for Tamil scope
        self._tess_langs = tesseract_langs
        self._easy_langs = easyocr_langs or ["ta", "en"]

        # Lazily-imported sub-modules to avoid hard-fail on missing dependencies
        self._damage_detector = None
        self._candidate_gen = None
        self._retriever = None
        self._ranker = None

        log.info(
            "EpiRecoverPipeline initialised (language=Tamil, llm_backend=%s, tesseract=%s)",
            llm_backend, tesseract_langs,
        )

    # ------------------------------------------------------------------ #
    # Stage A — Validate upload + metadata                                 #
    # ------------------------------------------------------------------ #

    def stage_a_validate(
        self,
        img: np.ndarray,
        source_name: str,
        metadata: Optional[InscriptionMetadata] = None,
    ) -> str:
        """
        Validate the uploaded image and collect metadata.

        Returns the image hash ID used to key all downstream artifacts.
        """
        if img is None:
            raise ValueError("Image is None — could not decode the uploaded file.")
        h, w = img.shape[:2]
        max_dim = 4096
        if max(h, w) > max_dim:
            log.warning(
                "Image %s is very large (%dx%d); downstream OCR may be slow.", source_name, w, h
            )
        img_id = image_hash(img)
        log.info("Stage A: image=%s id=%s shape=%dx%d", source_name, img_id, w, h)
        return img_id

    # ------------------------------------------------------------------ #
    # Stage B — Preprocessing                                              #
    # ------------------------------------------------------------------ #

    def stage_b_preprocess(
        self,
        img: np.ndarray,
        apply_perspective: bool = False,
        save_dir: Optional[str] = None,
        img_id: str = "",
    ) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run the full OpenCV preprocessing pipeline.

        Returns the final cleaned binary image and a dict of all intermediate stages.
        Saves intermediates to `save_dir` if provided.
        Original image is never overwritten.
        """
        log.info("Stage B: preprocessing (perspective=%s)", apply_perspective)
        stages: Dict[str, np.ndarray] = preprocessing.preprocess_inscription(
            img,
            apply_perspective=apply_perspective,
            return_intermediates=True,
        )
        cleaned = stages["cleaned"]

        if save_dir and img_id:
            stage_dir = os.path.join(save_dir, img_id)
            os.makedirs(stage_dir, exist_ok=True)
            for name, arr in stages.items():
                save_image(os.path.join(stage_dir, f"{name}.png"), arr)
            log.debug("Stage B: saved %d intermediates to %s", len(stages), stage_dir)

        return cleaned, stages

    # ------------------------------------------------------------------ #
    # Stage C — Damage analysis                                            #
    # ------------------------------------------------------------------ #

    def stage_c_damage(
        self,
        img: np.ndarray,
        cleaned: np.ndarray,
        save_dir: Optional[str] = None,
        img_id: str = "",
    ) -> Dict:
        """
        Produce a damage mask and identify uncertain regions.

        Uses the heuristic baseline unless a trained U-Net model is configured.
        Returns a dict with keys: method, mask, regions, label, overlay_path.
        """
        log.info("Stage C: damage analysis")
        try:
            from src.damage_analysis.damage_detector import DamageDetector
            if self._damage_detector is None:
                self._damage_detector = DamageDetector(self._cfg)
            result = self._damage_detector.analyze(img, cleaned)
        except ImportError as e:
            log.warning("DamageDetector import failed (%s); returning empty damage result.", e)
            result = {
                "method": "unavailable",
                "mask": None,
                "regions": [],
                "label": "DAMAGE_ANALYSIS_UNAVAILABLE",
            }

        if save_dir and img_id and result.get("mask") is not None:
            overlay_path = os.path.join(save_dir, img_id, "damage_overlay.png")
            os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
            save_image(overlay_path, result.get("overlay", result["mask"]))
            result["overlay_path"] = overlay_path

        return result

    # ------------------------------------------------------------------ #
    # Stage D — OCR with uncertainty markers                               #
    # ------------------------------------------------------------------ #

    def stage_d_ocr(
        self,
        cleaned: np.ndarray,
        damage_regions: Optional[List[Dict]] = None,
        script_hint: Optional[str] = "tamil",
    ) -> tuple[ocr_engine.OCRResult, List[str]]:
        """
        Run the OCR ensemble and insert [UNKNOWN] markers for low-confidence
        or damaged regions.

        Returns (OCRResult, list_of_illegible_region_markers).
        """
        log.info("Stage D: OCR ensemble (script_hint=%s)", script_hint)
        ocr_result = ocr_engine.extract_text(
            cleaned,
            tesseract_langs=self._tess_langs,
            easyocr_langs=self._easy_langs,
            script_hint=script_hint,
        )

        illegible: List[str] = []
        for box in ocr_result.boxes:
            if box.confidence < self._uncertainty_threshold:
                illegible.append(
                    f"{self._unknown_marker}@({box.bbox[0]},{box.bbox[1]})"
                )

        log.info(
            "Stage D: engine=%s mean_conf=%.1f tokens=%d illegible=%d",
            ocr_result.engine, ocr_result.mean_confidence,
            len(ocr_result.boxes), len(illegible),
        )
        return ocr_result, illegible

    # ------------------------------------------------------------------ #
    # Stage E — Candidate reconstruction                                   #
    # ------------------------------------------------------------------ #

    def stage_e_reconstruct(
        self,
        ocr_result: ocr_engine.OCRResult,
        mapped_text: str,
        metadata: Optional[InscriptionMetadata] = None,
        retrieved_records: Optional[List[Dict]] = None,
        damage_regions: Optional[List[Dict]] = None,
    ) -> List[ReconstructionCandidate]:
        """
        Generate candidate readings from OCR output, dictionary mapping,
        and retrieved corpus records.

        Returns a list of ReconstructionCandidate objects ordered by
        descending provisional rank (Stage G will re-rank with full evidence).
        """
        log.info("Stage E: candidate reconstruction")
        try:
            from src.reconstruction.candidate_generator import CandidateGenerator
            if self._candidate_gen is None:
                self._candidate_gen = CandidateGenerator(
                    translator=self._translator, cfg=self._cfg
                )
            candidates = self._candidate_gen.generate(
                ocr_result=ocr_result,
                mapped_text=mapped_text,
                metadata=metadata,
                retrieved_records=retrieved_records or [],
                damage_regions=damage_regions or [],
            )
        except ImportError as e:
            log.warning("CandidateGenerator import failed (%s); returning stub candidate.", e)
            hint = metadata.script if metadata else None
            llm_out = self._translator.translate(
                ocr_result.raw_text, mapped_text, script_hint=hint
            )
            candidates = [
                ReconstructionCandidate(
                    candidate_id="CAND-STUB-01",
                    candidate_text=llm_out.cleaned_text or mapped_text or ocr_result.raw_text,
                    provenance=Provenance.INFERRED,
                    ocr_evidence=ocr_result.raw_text,
                    confidence=llm_out.confidence,
                    confidence_note="Self-reported LLM confidence; not a calibrated probability.",
                    uncertainty_reasons=["CandidateGenerator module unavailable"],
                )
            ]

        return candidates

    # ------------------------------------------------------------------ #
    # Stage F — Evidence retrieval                                         #
    # ------------------------------------------------------------------ #

    def stage_f_retrieve(
        self,
        query_text: str,
        metadata: Optional[InscriptionMetadata] = None,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Retrieve supporting corpus records for a given query text.

        Returns a list of dicts: {id, score, script, transcription, source, ...}.
        Never returns records from the evaluation set (data leakage prevention).
        """
        log.info("Stage F: evidence retrieval (query='%s...')", query_text[:40])
        try:
            from src.evidence_retrieval.retriever import CorpusRetriever
            if self._retriever is None:
                self._retriever = CorpusRetriever(
                    corpus_dir=os.path.join(self._data_dir, "corpus"),
                    cfg=self._cfg,
                )
            records = self._retriever.retrieve(
                query=query_text,
                script=metadata.script if metadata else None,
                language="tamil",
                exclude_ids=exclude_ids or [],
                top_k=5,
                metadata=metadata,  # enables metadata fallback when OCR query is non-Tamil
            )
        except ImportError as e:
            log.warning("CorpusRetriever import failed (%s); returning empty evidence.", e)
            records = []
        except Exception as e:
            log.warning("Evidence retrieval error: %s", e)
            records = []
        return records

    # ------------------------------------------------------------------ #
    # Stage G — Evidence-grounded ranking                                  #
    # ------------------------------------------------------------------ #

    def stage_g_rank(
        self,
        candidates: List[ReconstructionCandidate],
        ocr_result: ocr_engine.OCRResult,
        retrieved_records: List[Dict],
        metadata: Optional[InscriptionMetadata] = None,
        damage_regions: Optional[List[Dict]] = None,
    ) -> tuple[List[ReconstructionCandidate], Optional[Any]]:
        """
        Re-rank candidates using multimodal evidence fusion (V + O + L + H + R).

        Returns (ranked_candidates, abstention_decision).
        """
        log.info("Stage G: ranking %d candidates", len(candidates))
        abstention_decision = None
        try:
            from src.candidate_ranking.ranker import CandidateRanker
            if self._ranker is None:
                self._ranker = CandidateRanker(cfg=self._cfg)
            ranked = self._ranker.rank(
                candidates=candidates,
                ocr_result=ocr_result,
                retrieved_records=retrieved_records,
                metadata=metadata,
                damage_regions=damage_regions,
            )
            abstention_decision = self._ranker.decide_abstention(ranked)
        except ImportError as e:
            log.warning("CandidateRanker import failed (%s); returning original order.", e)
            ranked = candidates
        return ranked, abstention_decision

    # ------------------------------------------------------------------ #
    # Full orchestration                                                   #
    # ------------------------------------------------------------------ #

    def process(
        self,
        image_path: str,
        metadata: Optional[InscriptionMetadata] = None,
        apply_perspective: bool = False,
        save_intermediates: bool = False,
        intermediates_dir: Optional[str] = None,
        eval_exclude_ids: Optional[List[str]] = None,
    ) -> RecoveryReport:
        """
        Run the complete EpiRecover pipeline on a single image.

        Parameters
        ----------
        image_path : str
            Path to the inscription image.
        metadata : InscriptionMetadata, optional
            Researcher-supplied metadata; all fields optional.
        apply_perspective : bool
            Whether to attempt perspective correction.
        save_intermediates : bool
            Save preprocessing stages to disk for reproducibility.
        intermediates_dir : str, optional
            Directory for intermediate files; defaults to data/processed/.
        eval_exclude_ids : list of str, optional
            Corpus record IDs to exclude from retrieval (data-leakage prevention).

        Returns
        -------
        RecoveryReport
        """
        img = load_image(image_path)

        # Enforce Tamil language scope for this version
        if metadata is None:
            metadata = InscriptionMetadata(language="Tamil", script="tamil")
        elif metadata.language and metadata.language.lower() != "tamil":
            log.warning(
                "EpiRecover is configured for Tamil-language inscriptions only in this version. "
                "Target language '%s' is not supported in this version. Extensible to future versions.",
                metadata.language
            )

        # Stage A
        img_id = self.stage_a_validate(img, os.path.basename(image_path), metadata)

        save_dir = (
            intermediates_dir
            if intermediates_dir
            else os.path.join(self._data_dir, "processed")
        ) if save_intermediates else None

        # Stage B
        cleaned, stages = self.stage_b_preprocess(img, apply_perspective, save_dir, img_id)

        # Stage C
        damage_info = self.stage_c_damage(img, cleaned, save_dir, img_id)

        # Stage D
        ocr_result, illegible = self.stage_d_ocr(
            cleaned,
            damage_regions=damage_info.get("regions"),
            script_hint=metadata.script if metadata else "tamil",
        )

        # Dictionary mapping
        mapped_text = ""
        if self._mapping and ocr_result.raw_text:
            tokens = ocr_result.raw_text.split()
            mapped_text, _ = self._mapping.translate_tokens(tokens)

        # Stage F (retrieve before reconstruction so evidence can inform candidates)
        query = mapped_text or ocr_result.raw_text
        retrieved = self.stage_f_retrieve(query, metadata, eval_exclude_ids)

        # Stage E
        candidates = self.stage_e_reconstruct(
            ocr_result,
            mapped_text,
            metadata,
            retrieved,
            damage_regions=damage_info.get("regions"),
        )

        # Stage G
        ranked, abstention = self.stage_g_rank(
            candidates, ocr_result, retrieved, metadata,
            damage_regions=damage_info.get("regions"),
        )
        top = ranked[0] if ranked else None

        # Build report
        report = RecoveryReport(
            image_id=img_id,
            source_image=image_path,
            inscription_id=metadata.inscription_id if metadata else None,
            script_hint=metadata.script if metadata else "tamil",
            language_hint="Tamil",
            historical_period=metadata.historical_period if metadata else None,
            region=metadata.region if metadata else None,
            monument=metadata.monument if metadata else None,
            dataset_source=metadata.dataset_source if metadata else None,
            researcher_notes=metadata.researcher_notes if metadata else None,
            preprocessing_stages=list(stages.keys()),
            damage_method=damage_info.get("method", "none"),
            damage_mask_path=damage_info.get("mask_path"),
            damage_regions=damage_info.get("regions", []),
            damage_label=damage_info.get("label", ""),
            ocr_raw_text=ocr_result.raw_text,
            ocr_engine=ocr_result.engine,
            ocr_mean_confidence=ocr_result.mean_confidence,
            ocr_boxes=[
                {"text": b.text, "confidence": b.confidence,
                 "bbox": list(b.bbox)}
                for b in ocr_result.boxes
            ],
            ocr_script_hint=ocr_result.language_hint,
            illegible_regions=illegible,
            candidates=ranked,
            top_candidate=top,
            should_abstain=abstention.should_abstain if abstention else False,
            abstention_reason=abstention.reason if abstention else "",
            calibrated_confidence=top.rank_score if top and top.rank_score is not None else 0.0,
            fusion_features=top.rank_details if top and top.rank_details else {},
            mapped_text=mapped_text,
            cleaned_text=(abstention.effective_text if abstention else (top.candidate_text if top else "")),
            retrieved_records=retrieved,
        )

        # Populate backward-compat fields from the LLM output embedded in top candidate
        if top:
            report.english = top.visual_evidence or ""   # overridden by full LLM call below
        # For the full UI flow we also get English/Hindi from LLM translator
        script_hint = metadata.script if metadata else None
        try:
            llm_out = self._translator.translate(
                ocr_result.raw_text, mapped_text, script_hint=script_hint
            )
            report.english = llm_out.english
            report.hindi = llm_out.hindi
            report.alternatives = llm_out.alternatives
            report.notes = llm_out.notes
            report.llm_confidence = llm_out.confidence
            report.llm_backend = llm_out.backend
            if top:
                report.cleaned_text = llm_out.cleaned_text or top.candidate_text
        except Exception as e:
            log.warning("LLM translation failed: %s", e)
            report.notes = f"LLM translation failed: {e}"

        log.info(
            "Pipeline complete for %s — overall_confidence=%.2f",
            image_path, report.overall_confidence,
        )
        return report


# ---------------------------------------------------------------------------
# Backward-compatible InscriptionTranslator wrapper
# Kept so existing CLI / tests that import from main.py still work.
# ---------------------------------------------------------------------------

class InscriptionTranslator(EpiRecoverPipeline):
    """
    Thin alias for EpiRecoverPipeline with the original EpiQ signature.
    Preserved for backward compatibility.
    """

    def process(  # type: ignore[override]
        self,
        image_path: str,
        apply_perspective: bool = False,
        script_hint: Optional[str] = None,
        debug_dir: Optional[str] = None,
    ) -> PipelineReport:
        """Run the pipeline and return the original EpiQ PipelineReport."""
        meta = InscriptionMetadata(script=script_hint) if script_hint else None
        report = super().process(
            image_path,
            metadata=meta,
            apply_perspective=apply_perspective,
            save_intermediates=debug_dir is not None,
            intermediates_dir=debug_dir,
        )
        return PipelineReport(
            source_image=image_path,
            extracted_text=report.ocr_raw_text,
            mapped_text=report.mapped_text,
            cleaned_text=report.cleaned_text,
            english=report.english,
            hindi=report.hindi,
            alternatives=report.alternatives,
            notes=report.notes,
            ocr_confidence=report.ocr_mean_confidence,
            llm_confidence=report.llm_confidence,
        )

    # Keep upstream stage names for any code that calls them directly
    def preprocess(self, img, apply_perspective=False, debug_dir=None):
        cleaned, stages = self.stage_b_preprocess(img, apply_perspective)
        if debug_dir:
            for name, arr in stages.items():
                save_image(os.path.join(debug_dir, f"{name}.png"), arr)
        return stages["cleaned"], stages

    def ocr(self, binary_img):
        result, _ = self.stage_d_ocr(binary_img)
        return result

    def dictionary_map(self, raw_text: str):
        if self._mapping is None:
            return raw_text, []
        tokens = raw_text.split()
        return self._mapping.translate_tokens(tokens)

    def translate(self, raw_text, mapped_text, script_hint=None):
        return self._translator.translate(raw_text, mapped_text, script_hint=script_hint)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EpiRecover - ancient inscription reconstruction pipeline"
    )
    p.add_argument("image", help="Path to the inscription image.")
    p.add_argument("--perspective", action="store_true",
                   help="Try perspective correction before OCR.")
    p.add_argument("--script-hint", default=None,
                   help="Script hint: brahmi|tamil-brahmi|sanskrit|prakrit|pali|devanagari|tamil")
    p.add_argument("--region", default=None, help="Geographic region metadata.")
    p.add_argument("--period", default=None, help="Historical period metadata.")
    p.add_argument("--monument", default=None, help="Monument or site name.")
    p.add_argument("--save-debug", default=None,
                   help="Directory to save intermediate preprocessing images.")
    p.add_argument("--backend", default="auto",
                   help="LLM backend: auto|gemini|groq|local|stub")
    p.add_argument("--out", default=None,
                   help="Directory to save the JSON report.")
    p.add_argument("--json", action="store_true",
                   help="Print full JSON report instead of summary.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    setup_logging("DEBUG" if args.verbose else "INFO")

    pipe = EpiRecoverPipeline(llm_backend=args.backend)
    meta = InscriptionMetadata(
        script=args.script_hint,
        region=args.region,
        historical_period=args.period,
        monument=args.monument,
    )
    report = pipe.process(
        args.image,
        metadata=meta,
        apply_perspective=args.perspective,
        save_intermediates=bool(args.save_debug),
        intermediates_dir=args.save_debug,
    )

    if args.json:
        print(report.to_json())
    else:
        print(f"=== EpiRecover Report ===")
        print(f"Image ID      : {report.image_id}")
        print(f"OCR engine    : {report.ocr_engine}  conf={report.ocr_mean_confidence:.1f}")
        print(f"OCR text      : {report.ocr_raw_text}")
        print(f"Mapped text   : {report.mapped_text}")
        print(f"Damage method : {report.damage_method} [{report.damage_label}]")
        print(f"Candidates    : {len(report.candidates)}")
        if report.top_candidate:
            print(f"Top candidate : {report.top_candidate.candidate_text}")
            print(f"  Provenance  : {report.top_candidate.provenance.value}")
            print(f"  Confidence  : {report.top_candidate.confidence:.2f}")
        print(f"English       : {report.english}")
        print(f"Hindi         : {report.hindi}")
        print(f"Overall conf  : {report.overall_confidence:.2f}")
        if report.notes:
            print(f"Notes         : {report.notes}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, f"{report.image_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report.to_json())
        print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
