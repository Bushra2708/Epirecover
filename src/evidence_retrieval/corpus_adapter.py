"""
src/evidence_retrieval/corpus_adapter.py
-----------------------------------------
Corpus ingestion adapter for EpiRecover.

Supported formats:
  - JSONL (one record per line)
  - JSON (list of records or dict of records)
  - CSV (with header row)

Expected fields per record (all optional except 'id'):
  id                  — unique record identifier
  image_path          — relative path to the inscription image
  script              — script name (brahmi, tamil-brahmi, etc.)
  language            — language of the inscription
  transcription       — text of the inscription (transliterated or in script)
  iast                — IAST romanisation if available
  historical_period   — e.g. "3rd century BCE"
  region              — geographic region
  monument            — monument or site name
  source              — bibliographic citation or source dataset name
  notes               — additional notes

Research integrity:
  - Records loaded from data/corpus/bharatshri_sample.jsonl are SYNTHETIC.
    They are clearly labelled in the file and in log output.
  - Real BharatSHRI data, if imported by the researcher, is NOT fabricated
    by this tool. The import workflow is documented in docs/DATASET_GUIDE.md.
  - Duplicate detection prevents the same inscription from appearing in both
    the evidence corpus and the evaluation set.

EpiRecover, 2026.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Dict, Iterator, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Record normalisation
# ---------------------------------------------------------------------------

REQUIRED_FIELD = "id"
OPTIONAL_FIELDS = [
    "image_path", "script", "language", "transcription", "iast",
    "historical_period", "region", "monument", "source", "notes",
    "synthetic",   # "true" if the record is a synthetic fixture
]


def _normalise(raw: Dict) -> Optional[Dict]:
    """
    Normalise a raw record dict to the canonical EpiRecover schema.

    Returns None if the record cannot be normalised (missing id).
    """
    # Accept common alternative field names
    rec_id = (
        raw.get("id") or raw.get("record_id") or raw.get("inscription_id")
        or raw.get("epigraphic_id")
    )
    if not rec_id:
        return None
    rec = {"id": str(rec_id)}
    for f in OPTIONAL_FIELDS:
        val = raw.get(f) or raw.get(f.replace("_", "")) or ""
        rec[f] = str(val) if val else ""
    # Text index for retrieval
    rec["_text_index"] = " ".join(filter(None, [
        rec.get("transcription"), rec.get("iast"),
        rec.get("script"), rec.get("language"),
        rec.get("region"), rec.get("monument"),
        rec.get("notes"),
    ])).lower()
    return rec


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("JSONL parse error at %s line %d: %s", path, i, e)


def _load_json(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        # Could be {id: record} or {"records": [...]}
        if "records" in data:
            yield from data["records"]
        else:
            yield from data.values()
    else:
        log.warning("Unrecognised JSON structure in %s", path)


def _load_csv(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        yield from reader


# ---------------------------------------------------------------------------
# Corpus adapter
# ---------------------------------------------------------------------------

class CorpusAdapter:
    """
    Loads inscription records from one or more files and exposes them
    as a list of normalised dicts for downstream retrieval.
    """

    def __init__(self) -> None:
        self._records: List[Dict] = []
        self._id_set: set = set()

    def load_file(self, path: str, synthetic: bool = False) -> int:
        """
        Load records from a file (JSONL, JSON, or CSV).

        Parameters
        ----------
        path : str
            Path to the corpus file.
        synthetic : bool
            If True, tag all records as synthetic fixture data.

        Returns
        -------
        Number of records successfully loaded from this file.
        """
        if not os.path.exists(path):
            log.warning("Corpus file not found: %s", path)
            return 0

        ext = os.path.splitext(path)[1].lower()
        if ext == ".jsonl":
            raw_iter = _load_jsonl(path)
        elif ext == ".json":
            raw_iter = _load_json(path)
        elif ext == ".csv":
            raw_iter = _load_csv(path)
        else:
            log.warning("Unsupported corpus file format: %s", path)
            return 0

        count = 0
        for raw in raw_iter:
            rec = _normalise(raw)
            if rec is None:
                continue
            if rec["id"] in self._id_set:
                continue   # deduplicate by ID
            if synthetic:
                rec["synthetic"] = "true"
            self._id_set.add(rec["id"])
            self._records.append(rec)
            count += 1

        label = "SYNTHETIC FIXTURE" if synthetic else "real"
        log.info("Loaded %d %s records from %s", count, label, path)
        return count

    def load_directory(self, directory: str) -> int:
        """Load all supported files in a directory."""
        if not os.path.isdir(directory):
            log.warning("Corpus directory not found: %s", directory)
            return 0
        total = 0
        for fname in sorted(os.listdir(directory)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in (".jsonl", ".json", ".csv"):
                continue
            path = os.path.join(directory, fname)
            is_synthetic = "sample" in fname or "synthetic" in fname or "fixture" in fname
            total += self.load_file(path, synthetic=is_synthetic)
        return total

    @property
    def records(self) -> List[Dict]:
        return list(self._records)

    @property
    def count(self) -> int:
        return len(self._records)

    def get_by_id(self, record_id: str) -> Optional[Dict]:
        for r in self._records:
            if r["id"] == record_id:
                return r
        return None
