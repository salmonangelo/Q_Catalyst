"""Uncertainty estimation module for Q-Catalyst.

Constructs multi-faceted uncertainty estimates for enzyme variant predictions:
- Epistemic model disagreement from a bootstrap-replicated XGBoost ensemble.
- Calibrated prediction intervals using split-conformal regression.
- Out-of-distribution (OOD) distance signals in representation space.
- A sequence-numbering-based catalytic-triad proximity heuristic (not 3D geometry) with explicit missingness handling.
- Scaled aggregate uncertainty score U(x) for candidate acquisition.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from qcatalyst.protein_ai.embeddings import ESM2Embedder, ESM2EmbedderConfig
from qcatalyst.uncertainty.calibrate import SplitConformalCalibrator


@dataclass
class EnsembleConfig:
    """Hyperparameters for bootstrap XGBoost ensemble."""

    n_members: int = 10  # Lightweight ensemble for fast, reproducible inference
    xgb_n_estimators: int = 80
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_seed: int = 42


class EnsemblePredictor:
    """Bootstrap-replicated ensemble of XGBoost regressors."""

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()
        self.scaler = StandardScaler()
        self.models: list[xgb.XGBRegressor] = []
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> EnsemblePredictor:
        """Fit ensemble members on bootstrap resamples of training data."""
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty.")

        n_samples = len(X)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.models.clear()
        rng = np.random.RandomState(self.config.random_seed)

        for i in range(self.config.n_members):
            # Bootstrap resample with replacement
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_b = X_scaled[boot_idx]
            y_b = y[boot_idx]

            member = xgb.XGBRegressor(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                random_state=self.config.random_seed + i * 17,
                n_jobs=2,  # Respect 2 CPU workers limit
            )
            member.fit(X_b, y_b)
            self.models.append(member)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute mean prediction, standard deviation disagreement, and full predictions matrix.

        Returns:
            (mean_prediction, std_disagreement, all_predictions_matrix)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before predict().")

        X_scaled = self.scaler.transform(X)
        preds = np.array([m.predict(X_scaled) for m in self.models], dtype=np.float64)  # (n_members, n_samples)

        mean_pred = np.mean(preds, axis=0)
        std_pred = np.std(preds, axis=0, ddof=1) if len(self.models) > 1 else np.zeros_like(mean_pred)

        return mean_pred, std_pred, preds


class OODDetector:
    """Out-of-distribution detector using distance in sequence embedding space."""

    def __init__(self) -> None:
        self.X_train_ref: np.ndarray | None = None
        self.reference_distance_95th: float = 1.0
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray) -> OODDetector:
        """Fit reference distribution statistics on training embeddings."""
        if len(X_train) < 3:
            self.is_fitted = False
            return self

        self.X_train_ref = np.asarray(X_train, dtype=np.float32)

        # Compute pairwise minimum distances within training set to calibrate scale
        train_dists = []
        for i in range(len(self.X_train_ref)):
            diffs = self.X_train_ref - self.X_train_ref[i]
            dists = np.linalg.norm(diffs, axis=1)
            # Min distance excluding self (idx i)
            dists[i] = np.inf
            train_dists.append(np.min(dists))

        self.reference_distance_95th = float(np.percentile(train_dists, 95)) if train_dists else 1.0
        if self.reference_distance_95th <= 1e-9:
            self.reference_distance_95th = 1.0

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Compute normalized distance to nearest training sample and status tags.

        Returns:
            (normalized_distances, status_tags)
        """
        X_arr = np.asarray(X, dtype=np.float32)

        if not self.is_fitted or self.X_train_ref is None:
            nan_arr = np.full(len(X_arr), np.nan)
            return nan_arr, ["insufficient_training_data"] * len(X_arr)

        normalized_scores = []
        statuses = []

        for vec in X_arr:
            diffs = self.X_train_ref - vec
            min_dist = float(np.min(np.linalg.norm(diffs, axis=1)))
            norm_dist = round(min_dist / self.reference_distance_95th, 4)
            normalized_scores.append(norm_dist)

            if norm_dist > 1.2:
                statuses.append("ood_warning")
            else:
                statuses.append("in_distribution")

        return np.array(normalized_scores, dtype=np.float64), statuses


def compute_structural_risk(
    structure_positions_str: str | None,
    structure_status: str,
) -> tuple[float | None, str]:
    """Compute a 1D residue-numbering heuristic from mapped 5XJH positions.

    Catalytic triad in 5XJH numbering: Ser160, Asp206, His237. The score decays
    with absolute difference in residue numbers. It is NOT Cartesian distance,
    contact analysis, solvent accessibility, or a 3D active-site gate. Role 2
    must perform geometry-based checks before chemistry allocation.
    If structural position is unmapped or missing, returns (None, 'missing_unmapped').
    NEVER imputes missing structural signal as zero.
    """
    if not structure_positions_str or structure_status in ("unmapped", "missing", "none"):
        return None, "missing_unmapped"

    triad_positions = {160, 206, 237}
    positions: list[int] = []

    for token in structure_positions_str.split(":"):
        token = token.strip()
        if token.isdigit():
            positions.append(int(token))

    if not positions:
        return None, "missing_unmapped"

    # Find minimum absolute difference in 5XJH residue numbering (not 3D distance).
    min_dist_to_triad = min(min(abs(p - t) for t in triad_positions) for p in positions)

    # Proximity score: 1.0 if directly at triad, decaying with distance
    # score in [0.0, 1.0]
    risk_score = float(np.exp(-min_dist_to_triad / 12.0))
    return round(risk_score, 4), "available"


def compute_aggregate_uncertainty(
    u_ensemble: float | None,
    u_conformal: float | None,
    u_ood: float | None,
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> tuple[float | None, str]:
    """Compute normalized composite uncertainty score U(x) across available signals.

    Missing components are documented explicitly in aggregate_status.
    """
    w_ens, w_conf, w_ood = weights
    available_terms: list[float] = []
    available_weights: list[float] = []
    missing_components: list[str] = []

    if u_ensemble is not None and np.isfinite(u_ensemble):
        # Scale typical std dev
        scaled_ens = min(1.0, max(0.0, u_ensemble / 0.5))
        available_terms.append(scaled_ens)
        available_weights.append(w_ens)
    else:
        missing_components.append("ensemble")

    if u_conformal is not None and np.isfinite(u_conformal):
        # Scale typical interval width
        scaled_conf = min(1.0, max(0.0, u_conformal / 1.0))
        available_terms.append(scaled_conf)
        available_weights.append(w_conf)
    else:
        missing_components.append("conformal")

    if u_ood is not None and np.isfinite(u_ood):
        scaled_ood = min(1.0, max(0.0, u_ood / 2.0))
        available_terms.append(scaled_ood)
        available_weights.append(w_ood)
    else:
        missing_components.append("ood")

    if not available_terms:
        return None, "all_signals_missing"

    total_weight = sum(available_weights)
    u_agg = sum(t * w for t, w in zip(available_terms, available_weights)) / total_weight

    status = "valid" if not missing_components else f"incomplete({','.join(missing_components)})"
    return round(float(u_agg), 4), status
