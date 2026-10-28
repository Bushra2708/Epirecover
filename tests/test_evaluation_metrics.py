"""
tests/test_evaluation_metrics.py
----------------------------------
Unit tests for src/evaluation/metrics.py.

Uses synthetic hypothesis/reference pairs.
No external datasets or API access required.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import (
    compute_cer,
    compute_wer,
    evaluate_single,
    aggregate_results,
    InscriptionResult,
)


# ---------------------------------------------------------------------------
# CER tests
# ---------------------------------------------------------------------------

def test_cer_identical():
    cer = compute_cer("piyadasi", "piyadasi")
    assert cer == pytest.approx(0.0, abs=1e-4)


def test_cer_one_char_error():
    # "pyadasi" vs "piyadasi" — 1 insertion / 8 chars = 0.125
    cer = compute_cer("pyadasi", "piyadasi")
    assert cer is not None
    assert 0.0 < cer <= 0.5


def test_cer_completely_wrong():
    cer = compute_cer("zzzzzzzz", "piyadasi")
    assert cer is not None
    assert cer > 0.5


def test_cer_empty_hypothesis():
    cer = compute_cer("", "piyadasi")
    assert cer is not None
    assert cer == pytest.approx(1.0, abs=1e-4)


def test_cer_empty_reference():
    result = compute_cer("something", "")
    # Empty reference returns None (metric unavailable)
    assert result is None


# ---------------------------------------------------------------------------
# WER tests
# ---------------------------------------------------------------------------

def test_wer_identical():
    wer = compute_wer("piyadasi dhamma", "piyadasi dhamma")
    assert wer == pytest.approx(0.0, abs=1e-4)


def test_wer_one_word_error():
    wer = compute_wer("piyadasi dharma", "piyadasi dhamma")
    assert wer is not None
    assert 0.0 < wer <= 1.0


def test_wer_empty_reference():
    result = compute_wer("something", "")
    assert result is None


# ---------------------------------------------------------------------------
# evaluate_single tests
# ---------------------------------------------------------------------------

def test_evaluate_single_with_reference():
    result = evaluate_single(
        inscription_id="TEST-001",
        hypothesis="piyadasi dhamma",
        reference="piyadasi dhamma",
    )
    assert result.cer == pytest.approx(0.0, abs=1e-4)
    assert result.wer == pytest.approx(0.0, abs=1e-4)
    assert result.metric_unavailable_reason == ""


def test_evaluate_single_without_reference():
    result = evaluate_single(
        inscription_id="TEST-002",
        hypothesis="piyadasi dhamma",
        reference=None,
    )
    assert result.cer is None
    assert result.wer is None
    assert result.metric_unavailable_reason != ""


def test_evaluate_single_with_ocr_baseline():
    result = evaluate_single(
        inscription_id="TEST-003",
        hypothesis="piyadasi dhamma",
        reference="piyadasi dhamma hita",
        ocr_raw="piyadis dhama",
    )
    assert result.ocr_cer is not None
    assert result.cer is not None


def test_evaluate_single_metadata():
    result = evaluate_single(
        inscription_id="TEST-004",
        hypothesis="foo",
        reference="foo",
        script="brahmi",
        damage_level="medium",
    )
    assert result.script == "brahmi"
    assert result.damage_level == "medium"


# ---------------------------------------------------------------------------
# aggregate_results tests
# ---------------------------------------------------------------------------

def test_aggregate_empty():
    report = aggregate_results([])
    assert report.total == 0
    assert report.mean_cer is None


def test_aggregate_with_results():
    results = [
        InscriptionResult(
            inscription_id="A", hypothesis="piyadasi", reference="piyadasi",
            cer=0.0, wer=0.0, script="brahmi", damage_level="light"
        ),
        InscriptionResult(
            inscription_id="B", hypothesis="damma", reference="dhamma",
            cer=0.2, wer=1.0, script="brahmi", damage_level="medium"
        ),
    ]
    report = aggregate_results(results)
    assert report.total == 2
    assert report.with_reference == 2
    assert report.mean_cer == pytest.approx(0.1, abs=1e-4)
    assert "brahmi" in report.by_script


def test_aggregate_no_references():
    results = [
        InscriptionResult(inscription_id="X", hypothesis="foo", reference=""),
    ]
    report = aggregate_results(results)
    assert report.mean_cer is None
    assert "unavailable" in report.notes.lower() or report.with_reference == 0
