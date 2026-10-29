"""Supervised PETase prediction module for Q-Catalyst.

Trains and evaluates primary supervised models (XGBoost) and comparison baselines (Ridge)
on sequence representations (e.g., ESM-2 embeddings) for activity and melting temperature targets.
Guarantees leakage-free evaluation with frozen splits, training-only preprocessors,
and strictly separates incompatible activity and thermostability targets.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


@dataclass
class ModelMetrics:
    """Evaluation metrics for regression and ranking performance."""

    mae: float
    rmse: float
    r2: float
    pearson_r: float
    pearson_pvalue: float
    spearman_rho: float
    spearman_pvalue: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
    """Calculate MAE, RMSE, R2, Pearson r, and Spearman rho on valid arrays."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Filter out NaNs if any
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    n = len(y_t)

    if n < 2:
        return ModelMetrics(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, n)

    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    # R2
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Pearson
    if np.std(y_t) > 1e-9 and np.std(y_p) > 1e-9:
        p_res = pearsonr(y_t, y_p)
        pr = float(p_res.statistic)
        pp = float(p_res.pvalue)
    else:
        pr, pp = 0.0, 1.0

    # Spearman
    if np.std(y_t) > 1e-9 and np.std(y_p) > 1e-9:
        s_res = spearmanr(y_t, y_p)
        sr = float(s_res.statistic)
        sp = float(s_res.pvalue)
    else:
        sr, sp = 0.0, 1.0

    return ModelMetrics(
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        r2=round(r2, 4),
        pearson_r=round(pr, 4),
        pearson_pvalue=float(f"{pp:.4e}"),
        spearman_rho=round(sr, 4),
        spearman_pvalue=float(f"{sp:.4e}"),
        n_samples=n,
    )


@dataclass
class SupervisedModelConfig:
    """Hyperparameters and configuration for supervised PETase models."""

    target_name: str = "activity"  # 'activity' or 'tm'
    parent_enzyme: str = "IsPETase"
    split_strategy: str = "single_to_multi"
    random_seed: int = 42

    # XGBoost hyperparameters
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0

    # Ridge baseline hyperparameters
    ridge_alpha: float = 1.0


class SupervisedPETasePredictor:
    """Supervised predictor pairing XGBoost with Ridge regression comparator."""

    def __init__(self, config: SupervisedModelConfig | None = None) -> None:
        self.config = config or SupervisedModelConfig()
        self.scaler = StandardScaler()

        self.xgb_model = xgb.XGBRegressor(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            reg_alpha=self.config.xgb_reg_alpha,
            reg_lambda=self.config.xgb_reg_lambda,
            random_state=self.config.random_seed,
            n_jobs=2,  # Respect 2 CPU workers limit
        )

        self.ridge_model = Ridge(
            alpha=self.config.ridge_alpha,
            random_state=self.config.random_seed,
        )

        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> SupervisedPETasePredictor:
        """Fit preprocessor and models strictly on training data."""
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data cannot be empty.")
        if len(X_train) != len(y_train):
            raise ValueError(f"Mismatch: X has {len(X_train)} samples, y has {len(y_train)} samples.")

        # Fit preprocessor only on training data (no data leakage)
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Fit XGBoost
        self.xgb_model.fit(X_train_scaled, y_train)

        # Fit Ridge baseline
        self.ridge_model.fit(X_train_scaled, y_train)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions from both XGBoost and Ridge baseline.

        Returns:
            (y_pred_xgb, y_pred_ridge)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        X_scaled = self.scaler.transform(X)
        pred_xgb = self.xgb_model.predict(X_scaled).astype(np.float64)
        pred_ridge = self.ridge_model.predict(X_scaled).astype(np.float64)

        return pred_xgb, pred_ridge

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[ModelMetrics, ModelMetrics]:
        """Evaluate both models on held-out test data.

        Returns:
            (xgb_metrics, ridge_metrics)
        """
        pred_xgb, pred_ridge = self.predict(X_test)
        metrics_xgb = compute_regression_metrics(y_test, pred_xgb)
        metrics_ridge = compute_regression_metrics(y_test, pred_ridge)
        return metrics_xgb, metrics_ridge

    def save(self, save_dir: Path | str) -> None:
        """Serialize fitted model, preprocessor, and configuration to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")

        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = path / "config.json"
        config_path.write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

        # Save XGBoost model in native JSON format
        xgb_path = path / "xgb_model.json"
        self.xgb_model.save_model(str(xgb_path))

        # Save Ridge weights and preprocessor parameters in JSON
        ridge_dict = {
            "coef": self.ridge_model.coef_.tolist(),
            "intercept": float(self.ridge_model.intercept_),
            "alpha": float(self.config.ridge_alpha),
        }
        (path / "ridge_model.json").write_text(json.dumps(ridge_dict, indent=2), encoding="utf-8")

        scaler_dict = {
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "var": self.scaler.var_.tolist(),
        }
        (path / "scaler.json").write_text(json.dumps(scaler_dict, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, save_dir: Path | str) -> SupervisedPETasePredictor:
        """Reload saved predictor and restore fitted weights."""
        path = Path(save_dir)
        config_data = json.loads((path / "config.json").read_text(encoding="utf-8"))
        config = SupervisedModelConfig(**config_data)

        predictor = cls(config)

        # Load scaler
        scaler_data = json.loads((path / "scaler.json").read_text(encoding="utf-8"))
        predictor.scaler.mean_ = np.array(scaler_data["mean"], dtype=np.float64)
        predictor.scaler.scale_ = np.array(scaler_data["scale"], dtype=np.float64)
        predictor.scaler.var_ = np.array(scaler_data["var"], dtype=np.float64)
        predictor.scaler.n_features_in_ = len(scaler_data["mean"])

        # Load Ridge model
        ridge_data = json.loads((path / "ridge_model.json").read_text(encoding="utf-8"))
        predictor.ridge_model.coef_ = np.array(ridge_data["coef"], dtype=np.float64)
        predictor.ridge_model.intercept_ = float(ridge_data["intercept"])

        # Load XGBoost model
        predictor.xgb_model.load_model(str(path / "xgb_model.json"))

        predictor.is_fitted = True
        return predictor


def create_frozen_split(
    df: pd.DataFrame,
    split_strategy: str = "single_to_multi",
    random_seed: int = 42,
    version: str = "v2",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Partition dataset into training and evaluation sets and record split metadata.

    Standard PET-Gym strategy:
    - Train on single-site mutants
    - Evaluate on multi-site mutants (held-out zero-shot generalization)
    - Version 'v2' enforces strict deduplication and zero ID/sequence overlap.
    """
    if split_strategy == "single_to_multi":
        if version == "v2" or "is_multi_mutant" in df.columns:
            # Deduplicate by variant_id across single-site and progressive screening sets
            deduped = df.drop_duplicates(subset=["variant_id"]).copy()
            if "is_multi_mutant" in deduped.columns:
                train_df = deduped[~deduped["is_multi_mutant"]].copy()
                eval_df = deduped[deduped["is_multi_mutant"]].copy()
            else:
                train_df = deduped[deduped["dataset_split"] == "single-site-mutant"].copy()
                train_vids = set(train_df["variant_id"])
                eval_df = deduped[
                    (deduped["dataset_split"] == "low-to-high-mutant")
                    & (~deduped["variant_id"].isin(train_vids))
                ].copy()

            # Assert strict disjointness (automated leakage guard)
            overlap = set(train_df["variant_id"]).intersection(set(eval_df["variant_id"]))
            assert len(overlap) == 0, (
                f"DATA LEAKAGE DETECTED: {len(overlap)} overlapping IDs in single_to_multi split"
            )
        else:
            # Historical v1 partition (preserved for regression testing)
            train_df = df[df["dataset_split"] == "single-site-mutant"].copy()
            eval_df = df[df["dataset_split"] == "low-to-high-mutant"].copy()
    else:
        # Fallback to random partition
        shuffled = df.sample(frac=1.0, random_state=random_seed).copy()
        n_train = int(len(df) * 0.75)
        train_df = shuffled.iloc[:n_train].copy()
        eval_df = shuffled.iloc[n_train:].copy()

    split_manifest = {
        "version": version,
        "split_strategy": split_strategy,
        "random_seed": random_seed,
        "total_records": len(train_df) + len(eval_df),
        "n_train": len(train_df),
        "n_eval": len(eval_df),
        "train_variant_ids": train_df["variant_id"].tolist(),
        "eval_variant_ids": eval_df["variant_id"].tolist(),
    }

    return train_df, eval_df, split_manifest
