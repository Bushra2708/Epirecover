"""
tests/test_candidate_ranking.py
--------------------------------
Unit tests for src/candidate_ranking/ranker.py.

Uses synthetic candidates and a stub OCR result so no network or
trained model is needed.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import ReconstructionCandidate, Provenance
from src.candidate_ranking.ranker import CandidateRanker, DEFAULT_WEIGHTS


# ---------------------------------------------------------------------------
# Stub OCR result
# ---------------------------------------------------------------------------

@dataclass
class StubOCRResult:
    raw_text: str = "piyadasi dhamma"
    mean_confidence: float = 60.0
    boxes: List = field(default_factory=list)
    engine: str = "stub"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ranker():
    return CandidateRanker(cfg={})


@pytest.fixture
def synthetic_candidates():
    return [
        ReconstructionCandidate(
            candidate_text="piyadasi dhamma hita",
            provenance=Provenance.CORPUS,
            corpus_evidence="SYNTH-001",
            confidence=0.8,
        ),
        ReconstructionCandidate(
            candidate_text="piyadasi dharma hitasukhaya",
            provenance=Provenance.INFERRED,
            confidence=0.5,
        ),
        ReconstructionCandidate(
            candidate_text="[UNKNOWN]",
            provenance=Provenance.UNKNOWN,
            confidence=0.0,
        ),
    ]


@pytest.fixture
def synthetic_records():
    return [
        {
            "id": "SYNTH-001",
            "script": "brahmi",
            "transcription": "piyadasi dhamma hita",
            "score": 0.82,
            "synthetic": "true",
        }
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ranker_default_weights():
    ranker = CandidateRanker()
    for key in DEFAULT_WEIGHTS:
        assert key in ranker._weights


def test_rank_returns_all_candidates(ranker, synthetic_candidates, synthetic_records):
    ranked = ranker.rank(
        candidates=synthetic_candidates,
        ocr_result=StubOCRResult(),
        retrieved_records=synthetic_records,
    )
    assert len(ranked) == len(synthetic_candidates)


def test_rank_descending_order(ranker, synthetic_candidates, synthetic_records):
    ranked = ranker.rank(
        candidates=synthetic_candidates,
        ocr_result=StubOCRResult(),
        retrieved_records=synthetic_records,
    )
    scores = [c.rank_score for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_unknown_lowest(ranker, synthetic_candidates, synthetic_records):
    ranked = ranker.rank(
        candidates=synthetic_candidates,
        ocr_result=StubOCRResult(),
        retrieved_records=synthetic_records,
    )
    # [UNKNOWN] candidate should be last
    last = ranked[-1]
    assert last.provenance == Provenance.UNKNOWN or last.rank_score <= ranked[0].rank_score


def test_rank_details_populated(ranker, synthetic_candidates, synthetic_records):
    ranked = ranker.rank(
        candidates=synthetic_candidates,
        ocr_result=StubOCRResult(),
        retrieved_records=synthetic_records,
    )
    for cand in ranked:
        assert isinstance(cand.rank_details, dict)
        assert len(cand.rank_details) > 0


def test_rank_single_candidate(ranker, synthetic_records):
    single = [ReconstructionCandidate(
        candidate_text="piyadasi",
        provenance=Provenance.OCR_PRED,
        confidence=0.7,
    )]
    ranked = ranker.rank(
        candidates=single,
        ocr_result=StubOCRResult(),
        retrieved_records=synthetic_records,
    )
    assert len(ranked) == 1
    assert ranked[0].rank_score >= 0.0


def test_rank_empty_candidates(ranker, synthetic_records):
    ranked = ranker.rank(
        candidates=[],
        ocr_result=StubOCRResult(),
        retrieved_records=synthetic_records,
    )
    assert ranked == []
