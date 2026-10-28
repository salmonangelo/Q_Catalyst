"""Split conformal calibration module for regression prediction intervals.

Implements inductive split-conformal regression with finite-sample coverage guarantees.
Calculates calibration non-conformity thresholds and measures empirical held-out coverage.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConformalIntervalResult:
    """Prediction intervals and calibration metadata for a batch of predictions."""

    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    interval_widths: np.ndarray
    threshold_q: float
    target_coverage: float
    calibration_size: int
    calibration_status: str  # 'calibrated', 'insufficient_calibration_data'


class SplitConformalCalibrator:
    """Split-conformal prediction interval generator for regression models.

    Computes non-conformity scores s_i = |y_i - y_hat_i| on an independent calibration split
    and derives the (1 - alpha) conformal quantile threshold with finite-sample coverage guarantee.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        """Initialize calibrator.

        Args:
            alpha: Desired error rate (default: 0.10 for 90% target coverage).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.target_coverage = 1.0 - alpha
        self.threshold_q: float | None = None
        self.n_cal: int = 0
        self.is_calibrated: bool = False

    def calibrate(self, y_cal: np.ndarray, y_pred_cal: np.ndarray) -> SplitConformalCalibrator:
        """Calibrate non-conformity scores using independent calibration data.

        Args:
            y_cal: Ground truth target values on calibration split.
            y_pred_cal: Model predictions on calibration split.

        Returns:
            Fitted calibrator instance.
        """
        y_true = np.asarray(y_cal, dtype=np.float64)
        y_pred = np.asarray(y_pred_cal, dtype=np.float64)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        self.n_cal = len(y_true)

        if self.n_cal < 5:
            # Insufficient samples to guarantee conformal coverage
            self.threshold_q = None
            self.is_calibrated = False
            return self

        # Non-conformity scores: absolute residuals
        scores = np.abs(y_true - y_pred)

        # Conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        quantile_level = math.ceil((self.n_cal + 1) * self.target_coverage) / self.n_cal
        quantile_level = min(1.0, max(0.0, quantile_level))

        self.threshold_q = float(np.quantile(scores, quantile_level, method="higher"))
        self.is_calibrated = True
        return self

    def predict_intervals(self, y_pred: np.ndarray) -> ConformalIntervalResult:
        """Construct conformal prediction intervals around point predictions.

        Args:
            y_pred: Array of model point predictions.

        Returns:
            ConformalIntervalResult containing lower/upper bounds and metadata.
        """
        preds = np.asarray(y_pred, dtype=np.float64)

        if not self.is_calibrated or self.threshold_q is None:
            # Cannot produce valid conformal intervals without calibration
            nan_arr = np.full_like(preds, np.nan)
            return ConformalIntervalResult(
                lower_bounds=nan_arr,
                upper_bounds=nan_arr,
                interval_widths=nan_arr,
                threshold_q=np.nan,
                target_coverage=self.target_coverage,
                calibration_size=self.n_cal,
                calibration_status="insufficient_calibration_data",
            )

        lower = preds - self.threshold_q
        upper = preds + self.threshold_q
        widths = upper - lower

        return ConformalIntervalResult(
            lower_bounds=lower,
            upper_bounds=upper,
            interval_widths=widths,
            threshold_q=self.threshold_q,
            target_coverage=self.target_coverage,
            calibration_size=self.n_cal,
            calibration_status="calibrated",
        )

    def evaluate_coverage(
        self,
        y_test: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> tuple[float, int, int]:
        """Compute empirical coverage percentage on held-out test points.

        Returns:
            (empirical_coverage, n_covered, total_valid_samples)
        """
        y_true = np.asarray(y_test, dtype=np.float64)
        low = np.asarray(lower_bounds, dtype=np.float64)
        up = np.asarray(upper_bounds, dtype=np.float64)

        mask = np.isfinite(y_true) & np.isfinite(low) & np.isfinite(up)
        if not np.any(mask):
            return 0.0, 0, 0

        covered = (y_true[mask] >= low[mask]) & (y_true[mask] <= up[mask])
        n_covered = int(np.sum(covered))
        total = int(len(covered))
        coverage = float(n_covered / total)

        return round(coverage, 4), n_covered, total
