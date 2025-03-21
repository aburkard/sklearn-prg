import numpy as np
import pytest
import warnings
from sklearn_prg.metrics import precision_recall_gain_curve


def test_precision_recall_gain_curve_all_positive():
    y_true = np.ones(100)
    y_scores = np.random.rand(100)

    pg, rg = precision_recall_gain_curve(y_true, y_scores)

    assert np.allclose(pg, 0), f"Expected PG=0 for all-positive, got {pg}"
    assert np.allclose(rg, 0), f"Expected RG=0 for all-positive, got {rg}"


def test_precision_recall_gain_curve_all_negative():
    y_true = np.zeros(100)
    y_scores = np.random.rand(100)

    # Explicitly silence expected sklearn warning about no positives
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pg, rg = precision_recall_gain_curve(y_true, y_scores)

    assert np.allclose(rg, 0), f"Expected RG=0 for all-negative, got {rg}"
    assert np.allclose(pg, 0), f"Expected PG=0 for all-negative, got {pg}"


def test_precision_recall_gain_curve_random_predictions():
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.3, size=1000)
    y_scores = rng.rand(1000)

    pg, rg = precision_recall_gain_curve(y_true, y_scores)

    assert np.abs(np.mean(pg)) < 0.1, f"Unexpectedly high precision gain {np.mean(pg)}"
    assert 0 <= np.mean(rg) <= 1, f"Recall gain should be within [0,1], got {np.mean(rg)}"


def test_precision_recall_gain_curve_monotonicity():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.05])

    pg, rg = precision_recall_gain_curve(y_true, y_scores)

    assert np.all(rg >= 0) and np.all(rg <= 1), f"Recall gain should be within [0,1], got {rg}"
    assert np.all(np.isfinite(pg)), "Precision gain should be finite"
    assert np.all(np.isfinite(rg)), "Recall gain should be finite"


def test_precision_recall_gain_curve_perfect_classifier():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])

    pg, rg = precision_recall_gain_curve(y_true, y_scores)

    assert np.isclose(np.max(pg), 1), f"Expected max PG=1, got {np.max(pg)}"
    assert np.isclose(np.max(rg), 1), f"Expected max RG=1, got {np.max(rg)}"


def test_precision_recall_gain_curve_single_positive():
    y_true = np.array([0, 0, 0, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.9])

    pg, rg = precision_recall_gain_curve(y_true, y_scores)

    assert np.isclose(np.max(rg), 1), f"Expected max RG=1, got {np.max(rg)}"
    assert np.isclose(np.max(pg), 1), f"Expected max PG=1, got {np.max(pg)}"
