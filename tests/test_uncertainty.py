"""Unit tests for uncertainty estimation, split-conformal calibration, and OOD detection."""

import numpy as np
import pytest

from qcatalyst.uncertainty.calibrate import SplitConformalCalibrator
from qcatalyst.uncertainty.estimate import (
    EnsembleConfig,
    EnsemblePredictor,
    OODDetector,
    compute_aggregate_uncertainty,
    compute_structural_risk,
)


def test_split_conformal_calibrator_coverage():
    """Verify split conformal calibrator produces finite intervals and meets coverage target."""
    rng = np.random.RandomState(42)
    # Synthetic calibration residuals with known std
    n_cal = 100
    y_cal = rng.randn(n_cal) * 2.0
    y_pred_cal = y_cal + rng.randn(n_cal) * 0.5  # true error std = 0.5

    calibrator = SplitConformalCalibrator(alpha=0.10)
    calibrator.calibrate(y_cal, y_pred_cal)

    assert calibrator.is_calibrated is True
    assert calibrator.threshold_q is not None
    assert calibrator.threshold_q > 0.0

    # Test on new evaluation set
    n_test = 200
    y_test = rng.randn(n_test) * 2.0
    y_pred_test = y_test + rng.randn(n_test) * 0.5

    res = calibrator.predict_intervals(y_pred_test)
    assert res.calibration_status == "calibrated"
    assert len(res.lower_bounds) == n_test
    assert np.all(res.upper_bounds > res.lower_bounds)
    assert np.allclose(res.interval_widths, 2.0 * calibrator.threshold_q)

    # Evaluate empirical coverage
    cov, n_cov, n_tot = calibrator.evaluate_coverage(y_test, res.lower_bounds, res.upper_bounds)
    assert n_tot == n_test
    # 90% target coverage should achieve approximately 85-95% coverage on test set
    assert 0.82 <= cov <= 0.98


def test_conformal_insufficient_data_does_not_invent_intervals():
    """Calibration set with < 5 samples must refuse to calibrate and return NaN intervals."""
    y_cal = np.array([1.0, 2.0, 3.0])
    y_pred_cal = np.array([1.1, 1.9, 3.2])

    calibrator = SplitConformalCalibrator(alpha=0.10)
    calibrator.calibrate(y_cal, y_pred_cal)

    assert calibrator.is_calibrated is False
    assert calibrator.threshold_q is None

    res = calibrator.predict_intervals(np.array([1.5, 2.5]))
    assert res.calibration_status == "insufficient_calibration_data"
    assert np.all(np.isnan(res.lower_bounds))
    assert np.all(np.isnan(res.upper_bounds))


def test_ensemble_predictor_std():
    """Verify bootstrap ensemble produces finite predictions and non-negative disagreement."""
    rng = np.random.RandomState(42)
    X = rng.randn(30, 8).astype(np.float32)
    y = X[:, 0] * 2.0 + rng.randn(30) * 0.2

    cfg = EnsembleConfig(n_members=5, xgb_n_estimators=10, random_seed=42)
    ens = EnsemblePredictor(cfg)
    ens.fit(X, y)

    X_test = rng.randn(10, 8).astype(np.float32)
    mean_pred, std_pred, all_preds = ens.predict(X_test)

    assert mean_pred.shape == (10,)
    assert std_pred.shape == (10,)
    assert all_preds.shape == (5, 10)
    assert np.all(std_pred >= 0.0)
    assert np.all(np.isfinite(std_pred))


def test_ood_detector():
    """Verify OOD detector flags distant points and handles small samples safely."""
    rng = np.random.RandomState(42)
    # Training cluster centered at origin
    X_train = rng.randn(50, 16).astype(np.float32)

    detector = OODDetector()
    detector.fit(X_train)

    # In-distribution point
    X_in = rng.randn(5, 16).astype(np.float32) * 0.5
    scores_in, status_in = detector.score(X_in)
    assert np.all(np.isfinite(scores_in))
    assert all(s == "in_distribution" for s in status_in)

    # Out-of-distribution point (distant from training cluster)
    X_out = rng.randn(5, 16).astype(np.float32) + 20.0
    scores_out, status_out = detector.score(X_out)
    assert any(s == "ood_warning" for s in status_out)

    # Insufficient data (< 3 samples)
    detector_small = OODDetector()
    detector_small.fit(np.zeros((2, 16)))
    scores_small, status_small = detector_small.score(X_in)
    assert all(s == "insufficient_training_data" for s in status_small)
    assert np.all(np.isnan(scores_small))


def test_structural_risk_explicit_missingness():
    """Verify structural risk is computed near active site and NEVER imputed as zero when missing."""
    # Near Ser160 in catalytic triad
    risk, status = compute_structural_risk("160", "exact_1to1")
    assert status == "available"
    assert risk is not None
    assert risk > 0.9  # Direct triad residue

    # Far from triad
    risk_far, status_far = compute_structural_risk("50", "exact_1to1")
    assert status_far == "available"
    assert risk_far is not None
    assert risk_far < risk

    # Missing / unmapped structure positions
    risk_miss, status_miss = compute_structural_risk("", "unmapped")
    assert risk_miss is None  # MUST be None, not 0.0!
    assert status_miss == "missing_unmapped"

    risk_none, status_none = compute_structural_risk(None, "missing")
    assert risk_none is None
    assert status_none == "missing_unmapped"


def test_aggregate_uncertainty_handling():
    """Verify aggregate uncertainty combines available signals and flags missingness."""
    # All signals present
    u, status = compute_aggregate_uncertainty(u_ensemble=0.15, u_conformal=0.5, u_ood=0.8)
    assert status == "valid"
    assert u is not None
    assert 0.0 <= u <= 1.0

    # Missing conformal signal
    u_part, status_part = compute_aggregate_uncertainty(u_ensemble=0.15, u_conformal=None, u_ood=0.8)
    assert "incomplete" in status_part
    assert "conformal" in status_part
    assert u_part is not None  # Re-weighted on available terms

    # All signals missing
    u_none, status_none = compute_aggregate_uncertainty(None, None, None)
    assert u_none is None
    assert status_none == "all_signals_missing"



def test_conformal_small_calibration_set_refuses_nominal_90_percent_claim():
    """For alpha=.1, fewer than 9 calibration residuals cannot yield the finite split-conformal quantile."""
    y_cal = np.linspace(1.0, 2.0, 8)
    y_pred_cal = y_cal + np.linspace(-0.2, 0.2, 8)
    calibrator = SplitConformalCalibrator(alpha=0.10)
    calibrator.calibrate(y_cal, y_pred_cal)

    assert calibrator.is_calibrated is False
    assert calibrator.threshold_q is None
    result = calibrator.predict_intervals(np.array([1.5]))
    assert result.calibration_status == "insufficient_calibration_data"
    assert np.isnan(result.lower_bounds[0])
    assert np.isnan(result.upper_bounds[0])


def test_conformal_minimum_finite_quantile_sample_size():
    """At n=9, the 90% split-conformal threshold is finite and calibration can proceed."""
    y_cal = np.linspace(1.0, 2.0, 9)
    y_pred_cal = y_cal + np.linspace(-0.2, 0.2, 9)
    calibrator = SplitConformalCalibrator(alpha=0.10)
    calibrator.calibrate(y_cal, y_pred_cal)
    assert calibrator.is_calibrated is True
    assert calibrator.threshold_q is not None
    assert np.isfinite(calibrator.threshold_q)
