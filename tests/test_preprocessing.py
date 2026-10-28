"""
tests/test_preprocessing.py
-----------------------------
Unit tests for preprocessing.py.

Tests use synthetic numpy arrays (not real inscription images) so they
run without any external datasets or API access.
"""

import numpy as np
import pytest
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import preprocessing


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gray_image():
    """Simple 64x64 grayscale synthetic image."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[20:44, 20:44] = 180  # bright square
    return img


@pytest.fixture
def bgr_image():
    """Simple 64x64 BGR synthetic image."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[20:44, 20:44] = (200, 150, 100)
    return img


# ---------------------------------------------------------------------------
# Individual stage tests
# ---------------------------------------------------------------------------

def test_to_grayscale_from_bgr(bgr_image):
    gray = preprocessing.to_grayscale(bgr_image)
    assert gray.ndim == 2
    assert gray.shape == (64, 64)


def test_to_grayscale_already_gray(gray_image):
    gray = preprocessing.to_grayscale(gray_image)
    assert gray.ndim == 2
    assert gray.shape == (64, 64)


def test_to_grayscale_none_raises():
    with pytest.raises(ValueError, match="None"):
        preprocessing.to_grayscale(None)


def test_denoise_output_shape(gray_image):
    out = preprocessing.denoise(gray_image)
    assert out.shape == gray_image.shape


def test_normalize_illumination_output_dtype(gray_image):
    out = preprocessing.normalize_illumination(gray_image)
    assert out.dtype == np.uint8
    assert out.shape == gray_image.shape


def test_adaptive_binarize_binary_values(gray_image):
    norm = preprocessing.normalize_illumination(gray_image)
    binary = preprocessing.adaptive_binarize(norm)
    unique = set(binary.flatten().tolist())
    assert unique <= {0, 255}, f"Binary image should only contain 0 and 255, got {unique}"


def test_adaptive_binarize_even_block_corrected(gray_image):
    """Even block_size should be corrected to odd without error."""
    norm = preprocessing.normalize_illumination(gray_image)
    out = preprocessing.adaptive_binarize(norm, block_size=30)
    assert out is not None


def test_morphological_cleanup_shape(gray_image):
    norm = preprocessing.normalize_illumination(gray_image)
    binary = preprocessing.adaptive_binarize(norm)
    cleaned = preprocessing.morphological_cleanup(binary)
    assert cleaned.shape == binary.shape


def test_enhance_edges_output(gray_image):
    edges = preprocessing.enhance_edges(gray_image)
    assert edges.shape == gray_image.shape
    assert edges.dtype == np.uint8


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

def test_preprocess_inscription_returns_dict(bgr_image):
    result = preprocessing.preprocess_inscription(bgr_image, return_intermediates=True)
    assert isinstance(result, dict)
    assert "cleaned" in result
    assert "grayscale" in result
    assert "binary" in result
    assert "edges" in result


def test_preprocess_inscription_returns_array(bgr_image):
    result = preprocessing.preprocess_inscription(bgr_image, return_intermediates=False)
    assert isinstance(result, np.ndarray)


def test_preprocess_inscription_none_raises():
    with pytest.raises(ValueError):
        preprocessing.preprocess_inscription(None)


def test_preprocess_with_perspective(bgr_image):
    """Perspective correction should not crash on a simple image."""
    result = preprocessing.preprocess_inscription(
        bgr_image, apply_perspective=True, return_intermediates=True
    )
    assert "cleaned" in result
