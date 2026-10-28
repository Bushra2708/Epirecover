"""
tests/test_dataset_matcher.py
------------------------------
Unit tests for dataset_matcher.py.

Uses a small in-memory fixture instead of data/script_mappings.json
so tests run without any file system dependencies.
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dataset_matcher

# ---------------------------------------------------------------------------
# Minimal fixture data
# ---------------------------------------------------------------------------

FIXTURE = [
    {
        "script": "brahmi",
        "glyph": "\U00011000",
        "devanagari": "क",
        "iast": "ka",
        "english_gloss": "ka"
    },
    {
        "script": "brahmi",
        "token": "piyadasi",
        "devanagari": "पियदसि",
        "iast": "piyadasi",
        "meaning": "beloved of the gods (epithet of Ashoka)"
    },
    {
        "script": "tamil-brahmi",
        "token": "arahata",
        "tamil": "அரஹத",
        "iast": "arahata",
        "meaning": "the worthy one"
    },
]


@pytest.fixture
def fixture_dataset(tmp_path):
    """Write fixture JSON to a temp file and return a ScriptMappingDataset."""
    p = tmp_path / "script_mappings.json"
    p.write_text(json.dumps(FIXTURE), encoding="utf-8")
    return dataset_matcher.ScriptMappingDataset(str(p))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_entries(fixture_dataset):
    assert fixture_dataset.entries is not None
    assert len(fixture_dataset.entries) == 3


def test_exact_match_glyph(fixture_dataset):
    hit = fixture_dataset.exact("\U00011000")
    assert hit is not None
    assert hit.iast == "ka"


def test_exact_match_token(fixture_dataset):
    hit = fixture_dataset.exact("piyadasi")
    assert hit is not None
    assert "पियदसि" in hit.modern


def test_exact_match_miss(fixture_dataset):
    hit = fixture_dataset.exact("xyz_not_in_dataset")
    assert hit is None


def test_fuzzy_match_close(fixture_dataset):
    """'piadasi' is one character away from 'piyadasi'."""
    results = fixture_dataset.fuzzy("piadasi", top_k=1, min_score=0.5)
    assert len(results) >= 1
    assert results[0].entry.iast == "piyadasi"


def test_fuzzy_match_below_threshold(fixture_dataset):
    """Completely unrelated string should return no fuzzy hits above 0.9."""
    results = fixture_dataset.fuzzy("zzzzzzzz", top_k=1, min_score=0.9)
    assert results == []


def test_translate_tokens_exact(fixture_dataset):
    mapped, details = fixture_dataset.translate_tokens(["piyadasi"])
    assert "पियदसि" in mapped
    assert details[0][1] is not None
    assert details[0][1].kind == "exact"


def test_translate_tokens_fuzzy(fixture_dataset):
    mapped, details = fixture_dataset.translate_tokens(["piadasi"])
    # Should fuzzy-match piyadasi
    assert details[0][1] is not None
    assert details[0][1].kind == "fuzzy"


def test_translate_tokens_unmatched(fixture_dataset):
    """Unmatched token passes through verbatim."""
    mapped, details = fixture_dataset.translate_tokens(["zzzzzzzz"])
    assert "zzzzzzzz" in mapped
    assert details[0][1] is None


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        dataset_matcher.ScriptMappingDataset("/nonexistent/path/to/file.json")
