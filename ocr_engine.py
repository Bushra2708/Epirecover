"""
ocr_engine.py
-------------
Multi-engine OCR wrapper for ancient inscriptions.

Strategy
--------
No single OCR engine reliably handles ancient scripts (Brahmi, Tamil-Brahmi,
Prakrit, etc.), so we try several in priority order and keep whichever
extracts the most high-confidence text:

    1. Tesseract (with `san`, `tam`, `hin`, `eng` language packs)
    2. EasyOCR (fallback — handles Devanagari and Tamil well)
    3. (Optional) TrOCR / a fine-tuned transformer model — plug-in point

Handling eroded inscriptions
----------------------------
- We run OCR at multiple scales and keep the best result.
- We also detect line-level bounding boxes so the downstream GenAI step
  knows which fragments sit on which line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Optional imports — we don't hard-fail if a backend is missing.
try:
    import pytesseract
    from pytesseract import Output
    _HAS_TESS = True
except ImportError:
    _HAS_TESS = False

try:
    import easyocr
    _HAS_EASY = True
except ImportError:
    _HAS_EASY = False

# Probe whether the Tamil model checkpoint actually loads.
# EasyOCR 1.7.x ships a Tamil CTC head expecting 127 classes but some
# cached checkpoints were built for 143 classes (older architecture).
# The mismatch raises RuntimeError at Reader() construction time.
_EASY_TAM_OK: bool = False
_EASY_TAM_ERROR: str = ""

def _probe_easyocr_tamil() -> None:
    """Attempt to load EasyOCR Tamil reader; set _EASY_TAM_OK accordingly."""
    global _EASY_TAM_OK, _EASY_TAM_ERROR
    if not _HAS_EASY:
        _EASY_TAM_ERROR = "easyocr package not installed"
        return
    try:
        # Use a tiny in-memory test — we only care whether the model loads.
        import easyocr as _eocr
        _eocr.Reader(["ta", "en"], gpu=False, verbose=False)
        _EASY_TAM_OK = True
        log.info("EasyOCR Tamil model probe: OK")
    except Exception as exc:
        _EASY_TAM_ERROR = str(exc)
        _EASY_TAM_OK = False
        log.warning(
            "EasyOCR Tamil model probe FAILED — Tamil OCR unavailable. "
            "Cause: %s. "
            "Workaround: downgrade easyocr to 1.6.x or replace the cached ta.pth checkpoint.",
            exc,
        )

_probe_easyocr_tamil()


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class OCRBox:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class OCRResult:
    raw_text: str = ""
    boxes: List[OCRBox] = field(default_factory=list)
    engine: str = ""
    mean_confidence: float = 0.0
    language_hint: Optional[str] = None
    script_hint: Optional[str] = None
    warning_note: Optional[str] = None


# ---------------------------------------------------------------------------
# Tesseract backend
# ---------------------------------------------------------------------------

def _tesseract_ocr(img: np.ndarray, langs: str = "eng+san+tam+hin") -> OCRResult:
    """
    Run Tesseract. `langs` is a '+' separated list of installed language packs.
    Install Sanskrit/Tamil/Hindi traineddata into your tessdata directory.
    """
    if not _HAS_TESS:
        return OCRResult(engine="tesseract-missing")

    # PSM 6 = treat as a single uniform block of text (good for inscriptions).
    config = "--oem 3 --psm 6"
    try:
        data = pytesseract.image_to_data(img, lang=langs, config=config, output_type=Output.DICT)
    except pytesseract.TesseractError as e:
        log.warning("Tesseract failed for langs=%s: %s", langs, e)
        return OCRResult(engine="tesseract-error")

    boxes: List[OCRBox] = []
    confs: List[float] = []
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        if not txt:
            continue
        try:
            c = float(data["conf"][i])
        except (ValueError, TypeError):
            c = -1
        if c < 0:
            continue
        boxes.append(OCRBox(
            text=txt,
            confidence=c,
            bbox=(data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
        ))
        confs.append(c)

    raw = " ".join(b.text for b in boxes)
    return OCRResult(
        raw_text=raw,
        boxes=boxes,
        engine="tesseract",
        mean_confidence=float(np.mean(confs)) if confs else 0.0,
        language_hint=langs,
    )


# ---------------------------------------------------------------------------
# EasyOCR backend
# ---------------------------------------------------------------------------

_EASY_READER_CACHE = {}

# EasyOCR compatibility rules.
# Tamil ('ta') can ONLY be combined with English ('en').
# Devanagari-family languages (hi, mr, ne, sa, ...) can be grouped with en.
# To cover the typical Indic inscription case we run multiple *compatible*
# groups sequentially and merge the results.
_EASYOCR_INCOMPATIBLE = {"ta"}  # languages that must be run in isolation (with 'en')


def _split_into_compatible_groups(langs: List[str]) -> List[List[str]]:
    """
    Split a requested language list into EasyOCR-compatible groups.
    Each returned group is safe to pass to easyocr.Reader(...).
    """
    langs = list(dict.fromkeys(langs))  # dedupe, keep order
    isolated = [l for l in langs if l in _EASYOCR_INCOMPATIBLE]
    shared = [l for l in langs if l not in _EASYOCR_INCOMPATIBLE]

    groups: List[List[str]] = []
    if shared:
        groups.append(shared if "en" in shared else shared + ["en"])
    for l in isolated:
        groups.append([l, "en"])  # EasyOCR requires English as the companion
    if not groups:
        groups.append(["en"])
    return groups


def _get_easy_reader(langs: List[str]):
    """Cache EasyOCR readers since model loading is expensive."""
    key = tuple(sorted(langs))
    if key not in _EASY_READER_CACHE:
        log.info("Loading EasyOCR reader for languages: %s", langs)
        _EASY_READER_CACHE[key] = easyocr.Reader(langs, gpu=False)
    return _EASY_READER_CACHE[key]


def _easyocr_ocr(img: np.ndarray, langs: Optional[List[str]] = None) -> OCRResult:
    """
    EasyOCR fallback. Supports Devanagari ('hi'), Tamil ('ta'), English ('en').
    Note: EasyOCR has no native Brahmi model — we use Devanagari / Tamil as
    the closest available approximation.

    Because EasyOCR refuses some language combinations (e.g. Tamil cannot be
    combined with Hindi), we split the requested list into compatible groups
    and run one reader per group, then merge the outputs.
    """
    if not _HAS_EASY:
        return OCRResult(engine="easyocr-missing")

    requested = langs or ["en", "hi", "ta"]
    groups = _split_into_compatible_groups(requested)

    all_boxes: List[OCRBox] = []
    all_confs: List[float] = []
    used_groups: List[str] = []

    for group in groups:
        try:
            reader = _get_easy_reader(group)
        except ValueError as e:
            # EasyOCR rejected the combination — skip this group.
            log.warning("EasyOCR rejected %s: %s", group, e)
            continue
        except Exception as e:
            log.warning("EasyOCR reader init failed for %s: %s", group, e)
            continue

        try:
            result = reader.readtext(img)
        except Exception as e:
            log.warning("EasyOCR readtext failed for %s: %s", group, e)
            continue

        for bbox_pts, text, conf in result:
            if not text.strip():
                continue
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            all_boxes.append(OCRBox(
                text=text.strip(),
                confidence=float(conf) * 100,
                bbox=(x, y, w, h),
            ))
            all_confs.append(float(conf) * 100)
        used_groups.append("+".join(group))

    if not all_boxes:
        return OCRResult(engine="easyocr", language_hint="|".join(used_groups))

    raw = " ".join(b.text for b in all_boxes)
    return OCRResult(
        raw_text=raw,
        boxes=all_boxes,
        engine="easyocr",
        mean_confidence=float(np.mean(all_confs)),
        language_hint="|".join(used_groups),
    )


# ---------------------------------------------------------------------------
# Multi-scale + ensemble wrapper
# ---------------------------------------------------------------------------

def _rescale(img: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return img
    interp = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=interp)


def check_tamil_ocr_support() -> Dict[str, Any]:
    """
    Check environment for Tamil OCR capability.

    Accurately probes whether the Tamil EasyOCR model checkpoint loads
    without error, rather than just checking package installation.
    Returns a status dict that the API /health endpoint and Streamlit UI use.
    """
    tess_installed = _HAS_TESS
    tess_tam = False
    if tess_installed:
        try:
            langs = pytesseract.get_languages()
            tess_tam = "tam" in langs
        except Exception:
            tess_tam = False

    easy_installed = _HAS_EASY
    # _EASY_TAM_OK is set by _probe_easyocr_tamil() at import time.
    easy_tam_model_ok = _EASY_TAM_OK
    # Actual Tamil OCR is available only if either Tesseract+tam or a working
    # EasyOCR Tamil checkpoint is present.
    modern_tamil_supported = tess_tam or easy_tam_model_ok

    notes_parts = [
        "Standard OCR engines (Tesseract/EasyOCR) only support modern Tamil script with appropriate language data.",
        "Historical Tamil-Brahmi lithic script is not supported by off-the-shelf models.",
    ]
    if not modern_tamil_supported:
        if easy_installed and not easy_tam_model_ok:
            notes_parts.append(
                f"EasyOCR is installed but the Tamil (ta) model checkpoint failed to load: "
                f"{_EASY_TAM_ERROR}. "
                "Fix: downgrade easyocr to 1.6.x or replace the cached ta.pth checkpoint."
            )
        if not tess_installed:
            notes_parts.append(
                "Tesseract is not installed or not in PATH. "
                "Install Tesseract 5.x and add to PATH, then install tam.traineddata."
            )
        elif not tess_tam:
            notes_parts.append(
                "Tesseract is installed but the 'tam' language pack is missing. "
                "Download tam.traineddata into your tessdata directory."
            )

    return {
        "tesseract_available": tess_installed,
        "tesseract_tam_pack": tess_tam,
        "easyocr_available": easy_installed,
        "easyocr_tam_model_ok": easy_tam_model_ok,
        "easyocr_tam_error": _EASY_TAM_ERROR if not easy_tam_model_ok else "",
        "modern_tamil_supported": modern_tamil_supported,
        "tamil_brahmi_supported": False,  # Standard OCR does not support historical Tamil-Brahmi
        "notes": " ".join(notes_parts),
    }


def extract_text(
    img: np.ndarray,
    tesseract_langs: str = "tam+eng",
    easyocr_langs: Optional[List[str]] = None,
    scales: Tuple[float, ...] = (1.0, 1.5, 2.0),
    script_hint: Optional[str] = None,
) -> OCRResult:
    """
    Run the OCR ensemble for Tamil inscriptions.

    Parameters
    ----------
    img : np.ndarray
        Cleaned binary or preprocessed inscription image.
    tesseract_langs : str
        Tesseract language codes. Defaults to 'tam+eng' for Tamil scope.
    easyocr_langs : list of str, optional
        EasyOCR languages. Defaults to ['ta', 'en'].
    scales : tuple of float
        Multi-scale factors.
    script_hint : str, optional
        Target script: 'tamil' (modern) or 'tamil-brahmi' (historical).

    Returns
    -------
    OCRResult with honest engine attribution, confidence, and script warnings.
    """
    # 1. Historical Tamil-Brahmi script check:
    # Standard OCR engines do not support Tamil-Brahmi. We report this transparently
    # rather than pretending modern Tamil OCR or English OCR recognized it.
    if script_hint and "tamil-brahmi" in script_hint.lower():
        log.warning(
            "Historical script 'tamil-brahmi' selected. Standard OCR engines "
            "do not support ancient Tamil-Brahmi without specialized model weights."
        )
        return OCRResult(
            raw_text="",
            engine="unsupported_script",
            script_hint="tamil-brahmi",
            language_hint="Tamil",
            warning_note=(
                "Historical script 'tamil-brahmi' is not supported by standard Tesseract/EasyOCR models. "
                "Ancient lithic Tamil-Brahmi cannot be processed by modern Tamil OCR. "
                "Candidate readings will rely on historical epigraphic corpus retrieval and manual review."
            ),
        )

    # 2. Modern Tamil script OCR:
    easy_langs = easyocr_langs or ["ta", "en"]
    best: OCRResult = OCRResult(engine="none", script_hint=script_hint or "tamil", language_hint="Tamil")
    used_tamil_ocr = False  # Track whether a genuine Tamil-capable engine ran

    for s in scales:
        scaled = _rescale(img, s)

        if _HAS_TESS:
            try:
                r = _tesseract_ocr(scaled, langs=tesseract_langs)
                if r.mean_confidence > best.mean_confidence and r.raw_text.strip():
                    best = r
                    best.script_hint = script_hint or "tamil"
                    used_tamil_ocr = True
            except Exception as e:
                log.warning("Tesseract crashed at scale=%.2f: %s", s, e)

        if _HAS_EASY:
            if _EASY_TAM_OK:
                # Tamil model is confirmed working — run Tamil+English reader.
                try:
                    r = _easyocr_ocr(scaled, langs=easy_langs)
                    if r.mean_confidence > best.mean_confidence and r.raw_text.strip():
                        best = r
                        best.script_hint = script_hint or "tamil"
                        used_tamil_ocr = True
                except Exception as e:
                    log.warning("EasyOCR crashed at scale=%.2f: %s", s, e)
            else:
                # Tamil checkpoint broken — do NOT run EasyOCR Tamil.
                # Running English-only would produce Latin noise and mislead the pipeline.
                log.debug(
                    "Skipping EasyOCR at scale=%.2f — Tamil model checkpoint unavailable.", s
                )

    if not best.raw_text:
        support = check_tamil_ocr_support()
        if not support["modern_tamil_supported"]:
            best.warning_note = (
                "Tamil OCR unavailable: Neither Tesseract (with 'tam.traineddata') nor a working "
                "EasyOCR Tamil model is available in the current environment. "
                "Candidates will be based on corpus retrieval and metadata only. "
                "To fix: install Tesseract + tam.traineddata, or downgrade easyocr to 1.6.x."
            )
            best.engine = "none-tamil-ocr-unavailable"
        else:
            best.warning_note = "No Tamil characters detected in the image by available OCR engines."
        log.warning("OCR output empty: %s", best.warning_note)
    elif not used_tamil_ocr:
        # OCR ran but only via an English fallback — flag the output as unreliable for Tamil.
        best.warning_note = (
            "OCR engines ran in English-fallback mode only (Tamil model unavailable). "
            "Text output may not represent Tamil script content and should not be used as "
            "primary evidence. See check_tamil_ocr_support() for details."
        )
        best.engine = best.engine + "-english-fallback" if best.engine else "easyocr-english-fallback"
        log.warning(
            "OCR produced text via English fallback engine (Tamil model unavailable). "
            "Raw text '%s...' should NOT be treated as Tamil evidence.",
            best.raw_text[:40],
        )

    return best


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def draw_boxes(img: np.ndarray, result: OCRResult,
               color: Tuple[int, int, int] = (0, 200, 0), thickness: int = 2) -> np.ndarray:
    """Return a copy of `img` with detected text regions drawn on."""
    vis = img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for b in result.boxes:
        x, y, w, h = b.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        label = f"{b.text} ({b.confidence:.0f})"
        cv2.putText(vis, label, (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis
