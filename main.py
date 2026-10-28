"""
main.py
-------
CLI entry point for the EpiRecover research pipeline, with backward-compatibility
for upstream EpiQ (Ancient Inscription Translator).

Usage (EpiRecover research pipeline):
    python main.py path/to/inscription.jpg \
        --script-hint brahmi \
        --backend stub \
        --out outputs/predictions/

Usage (Legacy EpiQ translator mode):
    python main.py path/to/inscription.jpg \
        --mode epiq \
        --backend auto

Programmatic:
    from main import InscriptionTranslator, EpiRecoverPipeline
    pipe = EpiRecoverPipeline()
    report = pipe.process("samples/my_stone.jpg")

Original upstream: sam-r-148/EpiQ (MIT licence preserved)
Extended for EpiRecover, 2026.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

# Ensure UTF-8 output on Windows for Indic scripts and Unicode epigraphy
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2

import preprocessing
import ocr_engine
import dataset_matcher
import genai_translator
import utils
from epirecover_pipeline import EpiRecoverPipeline, InscriptionMetadata

log = logging.getLogger("epirecover")


# ---------------------------------------------------------------------------
# Legacy Upstream Pipeline orchestrator (preserved for backward compatibility)
# ---------------------------------------------------------------------------

class InscriptionTranslator:
    """Upstream EpiQ orchestrator preserved for backward compatibility."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        llm_backend: str = "auto",
        tesseract_langs: str = "eng+san+tam+hin",
        easyocr_langs=None,
    ):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.mapping = dataset_matcher.load_default_dataset(self.data_dir)
        self.translator = genai_translator.make_translator(llm_backend)
        self.tesseract_langs = tesseract_langs
        self.easyocr_langs = easyocr_langs

    def preprocess(self, img, apply_perspective: bool = False, debug_dir: Optional[str] = None):
        stages = preprocessing.preprocess_inscription(
            img, apply_perspective=apply_perspective, return_intermediates=True
        )
        if debug_dir:
            for name, arr in stages.items():
                utils.save_image(os.path.join(debug_dir, f"{name}.png"), arr)
        return stages["cleaned"], stages

    def ocr(self, binary_img):
        return ocr_engine.extract_text(
            binary_img,
            tesseract_langs=self.tesseract_langs,
            easyocr_langs=self.easyocr_langs,
        )

    def dictionary_map(self, raw_text: str):
        tokens = raw_text.split()
        mapped, details = self.mapping.translate_tokens(tokens)
        return mapped, details

    def translate(self, raw_text: str, mapped_text: str, script_hint: Optional[str] = None):
        return self.translator.translate(raw_text, mapped_text, script_hint=script_hint)

    def process(
        self,
        image_path: str,
        apply_perspective: bool = False,
        script_hint: Optional[str] = None,
        debug_dir: Optional[str] = None,
    ) -> utils.PipelineReport:
        log.info("Loading %s", image_path)
        img = utils.load_image(image_path)

        log.info("Preprocessing …")
        cleaned, stages = self.preprocess(img, apply_perspective=apply_perspective, debug_dir=debug_dir)

        log.info("Running OCR ensemble …")
        ocr_result = self.ocr(cleaned)

        log.info("Dictionary mapping …")
        mapped_text, _details = self.dictionary_map(ocr_result.raw_text)

        log.info("LLM clean-up + translation …")
        llm_out = self.translate(ocr_result.raw_text, mapped_text, script_hint=script_hint)

        if debug_dir:
            vis = ocr_engine.draw_boxes(img, ocr_result)
            utils.save_image(os.path.join(debug_dir, "detected_regions.png"), vis)

        return utils.PipelineReport(
            source_image=image_path,
            extracted_text=ocr_result.raw_text,
            mapped_text=mapped_text,
            cleaned_text=llm_out.cleaned_text,
            english=llm_out.english,
            hindi=llm_out.hindi,
            alternatives=llm_out.alternatives,
            notes=llm_out.notes,
            ocr_confidence=ocr_result.mean_confidence,
            llm_confidence=llm_out.confidence,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="EpiRecover: Multimodal Reconstruction and Translation of Ancient Indian Inscriptions."
    )
    p.add_argument("image", help="Path to the inscription image.")
    p.add_argument("--mode", default="recover", choices=["recover", "epiq"],
                   help="Pipeline mode: 'recover' (default: full EpiRecover pipeline) or 'epiq' (legacy EpiQ)")
    p.add_argument("--perspective", action="store_true",
                   help="Try to detect and warp the inscription panel.")
    p.add_argument("--script-hint", default=None,
                   help="Optional hint: brahmi | tamil-brahmi | sanskrit | prakrit | pali | devanagari | tamil")
    p.add_argument("--region", default=None, help="Geographical provenance metadata.")
    p.add_argument("--period", default=None, help="Historical period metadata.")
    p.add_argument("--monument", default=None, help="Monument / site name.")
    p.add_argument("--save-debug", default=None,
                   help="Directory to save intermediate preprocessing and visualization images.")
    p.add_argument("--backend", default="auto",
                   help="LLM backend: auto (Gemini->Groq failover) | gemini | groq | local | stub")
    p.add_argument("--out", default=None, help="Directory to save output JSON report.")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of pretty text.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main():
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.mode == "epiq":
        t = InscriptionTranslator(llm_backend=args.backend)
        report = t.process(
            args.image,
            apply_perspective=args.perspective,
            script_hint=args.script_hint,
            debug_dir=args.save_debug,
        )
        if args.json:
            print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.pretty())
    else:
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
            print(report.pretty_summary())

        if args.out:
            os.makedirs(args.out, exist_ok=True)
            out_file = os.path.join(args.out, f"{report.image_id}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(report.to_json())
            print(f"\nReport written to {out_file}")


if __name__ == "__main__":
    main()
