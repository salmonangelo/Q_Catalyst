"""Unit and contract tests for Gate 0 Uncertainty Validation (Feature 7)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qcatalyst.uncertainty.gate import (
    Gate0Result,
    compute_spearman_bootstrap_ci,
    compute_worst_error_auroc,
    evaluate_gate0,
    generate_gate0_coverage_plot,
    save_gate0_artifacts,
)


def test_spearman_bootstrap_ci_positive_correlation():
    """Verify bootstrap CI strictly excludes zero when strong positive correlation exists."""
    rng = np.random.default_rng(42)
    # Strongly positively correlated signals
    error = np.linspace(0.1, 2.0, 100)
    uncertainty = error * 1.5 + rng.normal(0, 0.1, size=100)

    rho, p_val, ci_lower, ci_upper, ci_includes_zero = compute_spearman_bootstrap_ci(
        uncertainty, error, n_bootstrap=1000, seed=42
    )

    assert rho > 0.85
    assert p_val < 1e-10
    assert ci_lower > 0.80
    assert ci_upper > ci_lower
    assert ci_includes_zero is False


def test_spearman_bootstrap_ci_zero_correlation_includes_zero():
    """Verify bootstrap CI includes zero when signals are independent / uncorrelated."""
    rng = np.random.default_rng(42)
    error = rng.normal(1.0, 0.5, size=80)
    uncertainty = rng.normal(0.5, 0.2, size=80)  # independent noise

    rho, p_val, ci_lower, ci_upper, ci_includes_zero = compute_spearman_bootstrap_ci(
        uncertainty, error, n_bootstrap=1000, seed=42
    )

    assert ci_lower <= 0.0 <= ci_upper or ci_lower <= 0.0
    assert ci_includes_zero is True


def test_spearman_bootstrap_ci_edge_cases():
    """Verify constant signals or tiny sample sizes don't crash and report safe fallback."""
    # Constant x
    rho, p_val, ci_low, ci_high, zero_inc = compute_spearman_bootstrap_ci(
        np.ones(50), np.linspace(1, 10, 50)
    )
    assert zero_inc is True

    # Tiny sample size N < 4
    rho_small, _, _, _, zero_inc_small = compute_spearman_bootstrap_ci(
        np.array([1.0, 2.0]), np.array([2.0, 3.0])
    )
    assert zero_inc_small is True


def test_worst_error_auroc_valid_separation():
    """Verify AUROC correctly separates worst-quartile errors when predictive."""
    # Worst errors (top 25%) are indices 75..99
    abs_errors = np.linspace(0.1, 2.0, 100)
    # High uncertainty aligned with worst errors
    uncertainties = abs_errors.copy()

    auroc, status, threshold, count = compute_worst_error_auroc(abs_errors, uncertainties)

    assert status == "valid"
    assert auroc is not None
    assert auroc > 0.95
    assert count == 25
    assert threshold > 1.4


def test_worst_error_auroc_undefined_insufficient_samples():
    """Verify AUROC is explicitly marked undefined when samples are insufficient (< 8)."""
    abs_errors = np.array([0.1, 0.2, 0.5, 0.6])
    uncertainties = np.array([0.2, 0.3, 0.7, 0.8])

    auroc, status, _, _ = compute_worst_error_auroc(abs_errors, uncertainties, min_samples=8)

    assert auroc is None
    assert status == "undefined_insufficient_samples"


def test_worst_error_auroc_undefined_insufficient_classes():
    """Verify AUROC is explicitly marked undefined when all errors are identical (single class)."""
    # All identical errors
    abs_errors = np.full(50, 0.5)
    uncertainties = np.linspace(0.1, 0.9, 50)

    auroc, status, _, count = compute_worst_error_auroc(abs_errors, uncertainties)

    # In all-identical errors, threshold = 0.5, so all 50 items are >= threshold (pos_count = 50, unique classes = 1)
    assert auroc is None
    assert status == "undefined_insufficient_classes"


def test_gate0_verdict_passing_rule():
    """Test decision rule: Gate 0 PASSES when 95% CI lower bound is strictly > 0."""
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.uniform(0.5, 3.0, size=n)
    # Predictions with moderate error
    y_pred = y_true + rng.normal(0, 0.3, size=n)
    abs_err = np.abs(y_true - y_pred)
    # Uncertainty monotonically correlated with error
    u_score = abs_err * 0.8 + rng.normal(0, 0.05, size=n)

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred_mean": y_pred,
        "aggregate_uncertainty": u_score,
        "split_role": ["held_out_test"] * n,
    })

    result = evaluate_gate0(df, n_bootstrap=500, seed=42)

    assert result.gate_passed is True
    assert result.verdict == "PASS"
    assert result.recommended_action == "PROCEED_WITH_UNCERTAINTY_GATING"
    assert result.ci_lower > 0.0
    assert result.ci_includes_zero is False


def test_gate0_verdict_failing_rule_honest_reporting():
    """Test decision rule: Gate 0 FAILS honestly when 95% CI includes zero."""
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.uniform(0.5, 3.0, size=n)
    y_pred = y_true + rng.normal(0, 0.5, size=n)
    # Uncertainty completely independent of error (pure noise)
    u_noise = rng.uniform(0.1, 1.0, size=n)

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred_mean": y_pred,
        "aggregate_uncertainty": u_noise,
        "split_role": ["held_out_test"] * n,
    })

    result = evaluate_gate0(df, n_bootstrap=500, seed=42)

    assert result.gate_passed is False
    assert result.verdict == "FAIL"
    assert result.recommended_action == "FALLBACK_TO_DIVERSITY_SELECTION"
    assert result.ci_lower <= 0.0
    assert result.ci_includes_zero is True
    assert "Honest failure reported" in result.rationale


def test_gate0_artifacts_export_and_plot(tmp_path: Path):
    """Verify serialization of JSON verdict, summary CSV, report MD, and coverage PNG figure."""
    rng = np.random.default_rng(42)
    n = 60
    y_true = np.linspace(0.5, 2.5, n)
    y_pred = y_true + rng.normal(0, 0.2, n)
    u_agg = np.abs(y_true - y_pred) + 0.1

    test_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred_mean": y_pred,
        "conformal_lower": y_pred - 0.4,
        "conformal_upper": y_pred + 0.4,
        "aggregate_uncertainty": u_agg,
        "u_ensemble_std": rng.uniform(0.1, 0.3, n),
        "u_ood_distance": u_agg * 0.9,
        "u_conformal_width": np.full(n, 0.8),
        "split_role": ["held_out_test"] * n,
    })

    result = evaluate_gate0(test_df, n_bootstrap=300, seed=42)

    verdict_json = tmp_path / "gate0_verdict.json"
    summary_csv = tmp_path / "gate0_summary.csv"
    report_md = tmp_path / "gate0_validation_report.md"
    plot_png = tmp_path / "gate0_coverage_plot.png"

    # Save artifacts
    save_gate0_artifacts(result, verdict_json, summary_csv, report_md)
    generate_gate0_coverage_plot(test_df, plot_png, result=result)

    assert verdict_json.exists() and verdict_json.stat().st_size > 0
    assert summary_csv.exists() and summary_csv.stat().st_size > 0
    assert report_md.exists() and report_md.stat().st_size > 0
    assert plot_png.exists() and plot_png.stat().st_size > 0

    # Verify JSON load
    data = json.loads(verdict_json.read_text(encoding="utf-8"))
    assert data["verdict"] in ["PASS", "FAIL"]
    assert "components" in data
    assert "aggregate_uncertainty" in data["components"]


def test_gate0_real_dataset_integrity():
    """Verify Gate 0 execution on actual project held-out evaluation table."""
    uncertainty_path = Path("results/uncertainty/uncertainty.parquet")
    if not uncertainty_path.exists():
        pytest.skip("results/uncertainty/uncertainty.parquet not found.")

    df = pd.read_parquet(uncertainty_path)
    test_df = df[df["split_role"] == "held_out_test"]
    assert len(test_df) == 149

    result = evaluate_gate0(test_df, n_bootstrap=1000, seed=42)

    assert result.n_held_out == 149
    assert np.isfinite(result.spearman_rho)
    assert result.spearman_rho > 0.20  # ~0.2632
    assert result.ci_lower > 0.0       # CI lower bound > 0 -> PASS
    assert result.gate_passed is True
    assert result.verdict == "PASS"
    assert result.recommended_action == "PROCEED_WITH_UNCERTAINTY_GATING"
    assert result.auroc is not None
    assert result.auroc_status == "valid"
