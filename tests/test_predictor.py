"""Unit tests for supervised PETase predictor, metrics, and persistence."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qcatalyst.protein_ai.predictor import (
    ModelMetrics,
    SupervisedModelConfig,
    SupervisedPETasePredictor,
    compute_regression_metrics,
    create_frozen_split,
)


def test_compute_regression_metrics_perfect_correlation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.r2 == 1.0
    assert pytest.approx(metrics.pearson_r, abs=1e-4) == 1.0
    assert pytest.approx(metrics.spearman_rho, abs=1e-4) == 1.0
    assert metrics.n_samples == 5


def test_compute_regression_metrics_constant_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([2.0, 2.0, 2.0, 2.0, 2.0])  # zero variance

    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics.mae > 0.0
    assert metrics.pearson_r == 0.0
    assert metrics.spearman_rho == 0.0


def test_create_frozen_split_single_to_multi():
    df = pd.DataFrame([
        {"variant_id": "v1", "dataset_split": "single-site-mutant"},
        {"variant_id": "v2", "dataset_split": "single-site-mutant"},
        {"variant_id": "v3", "dataset_split": "low-to-high-mutant"},
        {"variant_id": "v4", "dataset_split": "low-to-high-mutant"},
    ])

    train_df, eval_df, manifest = create_frozen_split(df, split_strategy="single_to_multi")

    assert len(train_df) == 2
    assert len(eval_df) == 2
    assert set(train_df["variant_id"]).isdisjoint(set(eval_df["variant_id"]))
    assert manifest["split_strategy"] == "single_to_multi"
    assert manifest["n_train"] == 2
    assert manifest["n_eval"] == 2


def test_predictor_fit_predict_and_evaluate():
    rng = np.random.RandomState(42)
    X_train = rng.randn(40, 32).astype(np.float32)
    # y with linear + non-linear signal
    y_train = X_train[:, 0] * 2.0 - X_train[:, 1] * 1.5 + rng.randn(40) * 0.1

    X_test = rng.randn(20, 32).astype(np.float32)
    y_test = X_test[:, 0] * 2.0 - X_test[:, 1] * 1.5 + rng.randn(20) * 0.1

    config = SupervisedModelConfig(
        target_name="activity",
        parent_enzyme="IsPETase",
        xgb_n_estimators=20,  # Fast test
        random_seed=42,
    )
    predictor = SupervisedPETasePredictor(config)
    predictor.fit(X_train, y_train)

    assert predictor.is_fitted is True

    # Predict
    pred_xgb, pred_ridge = predictor.predict(X_test)
    assert pred_xgb.shape == (20,)
    assert pred_ridge.shape == (20,)
    assert np.all(np.isfinite(pred_xgb))
    assert np.all(np.isfinite(pred_ridge))

    # Evaluate
    metrics_xgb, metrics_ridge = predictor.evaluate(X_test, y_test)
    assert isinstance(metrics_xgb, ModelMetrics)
    assert isinstance(metrics_ridge, ModelMetrics)
    assert metrics_xgb.rmse > 0.0
    assert metrics_ridge.rmse > 0.0
    # Both models should learn the signal
    assert metrics_xgb.pearson_r > 0.5
    assert metrics_ridge.pearson_r > 0.5


def test_predictor_save_and_reload():
    rng = np.random.RandomState(42)
    X_train = rng.randn(30, 16).astype(np.float32)
    y_train = X_train[:, 0] * 1.5 + rng.randn(30) * 0.1
    X_test = rng.randn(10, 16).astype(np.float32)

    config = SupervisedModelConfig(xgb_n_estimators=15, random_seed=42)
    predictor = SupervisedPETasePredictor(config)
    predictor.fit(X_train, y_train)

    orig_xgb_pred, orig_ridge_pred = predictor.predict(X_test)

    with tempfile.TemporaryDirectory() as tmp_dir:
        predictor.save(tmp_dir)

        # Reload
        reloaded = SupervisedPETasePredictor.load(tmp_dir)
        reloaded_xgb_pred, reloaded_ridge_pred = reloaded.predict(X_test)

        assert np.allclose(orig_xgb_pred, reloaded_xgb_pred, atol=1e-5)
        assert np.allclose(orig_ridge_pred, reloaded_ridge_pred, atol=1e-5)


def test_predict_before_fit_raises_error():
    predictor = SupervisedPETasePredictor()
    with pytest.raises(RuntimeError, match="must be fitted"):
        predictor.predict(np.zeros((5, 10)))


def test_mismatched_sample_counts_raise_error():
    predictor = SupervisedPETasePredictor()
    with pytest.raises(ValueError, match="Mismatch"):
        predictor.fit(np.zeros((10, 5)), np.zeros(8))
