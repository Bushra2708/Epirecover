"""
tests/test_evidence_retrieval.py
---------------------------------
Unit tests for the corpus adapter and retriever.

Uses a small in-memory JSONL fixture; no real BharatSHRI data required.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evidence_retrieval.corpus_adapter import CorpusAdapter, _normalise
from src.evidence_retrieval.retriever import CorpusRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_RECORDS = [
    {"id": "T001", "script": "brahmi", "language": "Prakrit",
     "transcription": "piyadasi laja hevam aha", "source": "Synthetic"},
    {"id": "T002", "script": "brahmi", "language": "Prakrit",
     "transcription": "dhammam carati", "source": "Synthetic"},
    {"id": "T003", "script": "tamil-brahmi", "language": "Tamil",
     "transcription": "arahata atavimita", "source": "Synthetic"},
]


@pytest.fixture
def fixture_corpus_dir(tmp_path):
    p = tmp_path / "corpus"
    p.mkdir()
    f = p / "test_fixture.jsonl"
    f.write_text(
        "\n".join(json.dumps(r) for r in FIXTURE_RECORDS),
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture
def adapter(fixture_corpus_dir):
    a = CorpusAdapter()
    a.load_directory(fixture_corpus_dir)
    return a


@pytest.fixture
def retriever(fixture_corpus_dir):
    return CorpusRetriever(corpus_dir=fixture_corpus_dir, enable_faiss=False)


# ---------------------------------------------------------------------------
# CorpusAdapter tests
# ---------------------------------------------------------------------------

def test_normalise_valid():
    rec = _normalise({"id": "X001", "transcription": "test"})
    assert rec is not None
    assert rec["id"] == "X001"


def test_normalise_missing_id():
    rec = _normalise({"transcription": "no id field"})
    assert rec is None


def test_adapter_load_count(adapter):
    assert adapter.count == 3


def test_adapter_deduplication(tmp_path):
    """Duplicate IDs should only be stored once."""
    p = tmp_path / "dedup_fixture.jsonl"
    p.write_text(
        json.dumps({"id": "DUP-001", "transcription": "first"}) + "\n" +
        json.dumps({"id": "DUP-001", "transcription": "second"}),
        encoding="utf-8",
    )
    a = CorpusAdapter()
    a.load_file(str(p))
    assert a.count == 1


def test_adapter_get_by_id(adapter):
    rec = adapter.get_by_id("T001")
    assert rec is not None
    assert "piyadasi" in rec["transcription"]


def test_adapter_get_by_missing_id(adapter):
    assert adapter.get_by_id("NONEXISTENT") is None


def test_adapter_synthetic_tagging(tmp_path):
    p = tmp_path / "sample_fixture.jsonl"
    p.write_text(json.dumps({"id": "S001", "transcription": "test"}), encoding="utf-8")
    a = CorpusAdapter()
    a.load_file(str(p), synthetic=True)
    assert a.records[0]["synthetic"] == "true"


# ---------------------------------------------------------------------------
# CorpusRetriever tests
# ---------------------------------------------------------------------------

def test_retriever_count(retriever):
    assert retriever._adapter.count == 3


def test_retriever_exact_transcription(retriever):
    results = retriever.retrieve("piyadasi laja hevam aha", top_k=3)
    assert len(results) >= 1
    assert results[0]["id"] in {"T001", "T002", "T003"}


def test_retriever_fuzzy_match(retriever):
    results = retriever.retrieve("piyadasi", top_k=3, min_score=0.1)
    assert len(results) >= 1


def test_retriever_empty_query(retriever):
    results = retriever.retrieve("", top_k=5)
    assert results == []


def test_retriever_exclude_ids(retriever):
    """Excluded IDs must not appear in results."""
    results = retriever.retrieve("piyadasi", top_k=5, exclude_ids=["T001", "T002", "T003"])
    assert all(r["id"] not in {"T001", "T002", "T003"} for r in results)


def test_retriever_script_filter(retriever):
    results = retriever.retrieve("arahata", script="tamil-brahmi", top_k=3, min_score=0.1)
    # Should prefer T003
    assert any(r["id"] == "T003" for r in results)


def test_retriever_score_in_result(retriever):
    results = retriever.retrieve("dhammam", top_k=3, min_score=0.0)
    for r in results:
        assert "score" in r
        assert 0.0 <= r["score"] <= 1.0
