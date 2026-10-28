"""
src/candidate_ranking/calibration.py
------------------------------------
Confidence calibration for EpiRecover reconstruction candidates.

As specified in the EpiRecover architecture (PPT Slides 11, 12, 14):
  - Temperature scaling: Calibrates raw uncalibrated candidate scores.
  - Expected Calibration Error (ECE): Quantifies gap between confidence and empirical accuracy.
  - Brier Score: Mean squared error between predicted probabilities and binary outcomes.

Formula:
  p_calibrated = sigmoid(logit(score) / T)   or   softmax(logits / T)
  where T > 0 is learned via NLL minimization on a held-out validation set.

EpiRecover, 2026.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

log = logging.getLogger(__name__)


def temperature_scale(
    scores: np.ndarray | List[float],
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Apply temperature scaling to raw scores in [0, 1].

    Parameters
    ----------
    scores : array-like of floats in [0, 1]
        Raw candidate confidence / rank scores.
    temperature : float
        Temperature parameter T > 0.
        T > 1 softens overconfident scores; T < 1 sharpens underconfident scores.

    Returns
    -------
    np.ndarray of calibrated probabilities in [0, 1].
    """
    scores_arr = np.clip(np.asarray(scores, dtype=np.float64), eps, 1.0 - eps)
    if temperature <= 0.0:
        raise ValueError("Temperature T must be strictly positive.")

    # Convert probability to logit
    logits = np.log(scores_arr / (1.0 - scores_arr))
    # Scale logits by temperature
    scaled_logits = logits / temperature
    # Convert back via sigmoid
    calibrated = 1.0 / (1.0 + np.exp(-scaled_logits))
    return np.clip(calibrated, 0.0, 1.0)


def compute_ece(
    confidences: np.ndarray | List[float],
    accuracies: np.ndarray | List[int | bool],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Parameters
    ----------
    confidences : array-like of floats in [0, 1]
        Predicted confidence values for Top-1 candidate.
    accuracies : array-like of 0/1 or bool
        Binary ground-truth correctness (1 if candidate matches reference, 0 otherwise).
    n_bins : int
        Number of equal-width confidence bins (default: 10).

    Returns
    -------
    float : ECE in [0, 1].
    """
    confs = np.asarray(confidences, dtype=np.float64)
    accs = np.asarray(accuracies, dtype=np.float64)

    if len(confs) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = float(len(confs))

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples within this bin
        in_bin = (confs > bin_lower) & (confs <= bin_upper) if i > 0 else (confs >= bin_lower) & (confs <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(accs[in_bin])
            bin_conf = np.mean(confs[in_bin])
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return float(np.round(ece, 4))


def compute_brier_score(
    confidences: np.ndarray | List[float],
    accuracies: np.ndarray | List[int | bool],
) -> float:
    """
    Compute the Brier Score: (1 / N) * sum (p_i - y_i)^2.

    Parameters
    ----------
    confidences : array-like of floats in [0, 1]
    accuracies : array-like of 0/1 or bool

    Returns
    -------
    float : Brier score in [0, 1] (lower is better, 0.0 = perfect calibration).
    """
    confs = np.asarray(confidences, dtype=np.float64)
    accs = np.asarray(accuracies, dtype=np.float64)

    if len(confs) == 0:
        return 0.0

    return float(np.round(np.mean((confs - accs) ** 2), 4))


class TemperatureScaler:
    """
    Fits and applies post-hoc temperature scaling to calibrate confidence scores.
    Optimizes negative log-likelihood (NLL) via scipy.optimize or grid search.
    """

    def __init__(self, default_temperature: float = 1.0) -> None:
        self.temperature = float(default_temperature)
        self.is_fitted = False

    def fit(
        self,
        val_scores: np.ndarray | List[float],
        val_labels: np.ndarray | List[int | bool],
    ) -> float:
        """
        Fit temperature parameter T on a held-out validation set.
        """
        from scipy.optimize import minimize_scalar

        scores = np.asarray(val_scores, dtype=np.float64)
        labels = np.asarray(val_labels, dtype=np.float64)

        if len(scores) < 2 or len(np.unique(labels)) < 2:
            log.warning("Insufficient validation diversity to fit temperature. Using T=1.0.")
            self.temperature = 1.0
            self.is_fitted = True
            return self.temperature

        def _nll(temp: float) -> float:
            calibrated = temperature_scale(scores, temp)
            eps = 1e-12
            loss = -np.mean(labels * np.log(calibrated + eps) + (1.0 - labels) * np.log(1.0 - calibrated + eps))
            return loss

        res = minimize_scalar(_nll, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)
        self.is_fitted = True
        log.info("TemperatureScaler fitted: T=%.3f (NLL=%.4f)", self.temperature, res.fun)
        return self.temperature

    def calibrate(self, scores: np.ndarray | List[float]) -> np.ndarray:
        """Apply fitted temperature scaling."""
        return temperature_scale(scores, self.temperature)
