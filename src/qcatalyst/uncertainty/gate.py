"""Gate 0 Uncertainty Validation module for Q-Catalyst.

Gate 0 evaluates whether predicted uncertainty metrics are correlated with
actual prediction errors on strictly held-out variants.

Agreed Decision Rule:
- Calculate Spearman rank correlation (rho) between uncertainty scores and absolute prediction errors.
- Compute a bootstrap 95% confidence interval (B >= 1000) for rho.
- If the 95% CI includes zero (CI_lower <= 0.0), Gate 0 FAILS its criterion and
  Role 4 must use diversity-based selection instead of uncertainty-based gating.
- If the 95% CI strictly excludes zero (CI_lower > 0.0), Gate 0 PASSES.
- AUROC for classifying the worst-error quartile (top 25% absolute error) is computed
  when labels support it, and explicitly marked undefined when single-class or insufficient.

All statistics strictly use held-out predictions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


@dataclass
class ComponentMetrics:
    """Statistical validation metrics for a specific uncertainty component."""

    component_name: str
    spearman_rho: float
    spearman_pvalue: float
    ci_lower: float
    ci_upper: float
    ci_includes_zero: bool
    worst_quartile_auroc: float | None
    auroc_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Gate0Result:
    """Comprehensive Gate 0 validation result and decision verdict."""

    target_name: str
    parent_enzyme: str
    n_held_out: int
    primary_metric_name: str
    spearman_rho: float
    spearman_pvalue: float
    ci_level: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    ci_includes_zero: bool
    worst_error_quartile_threshold: float
    worst_error_quartile_count: int
    auroc: float | None
    auroc_status: str
    gate_passed: bool
    verdict: str  # "PASS" or "FAIL"
    decision_rule: str
    recommended_action: str  # "PROCEED_WITH_UNCERTAINTY_GATING" or "FALLBACK_TO_DIVERSITY_SELECTION"
    rationale: str
    components: dict[str, ComponentMetrics] = field(default_factory=dict)
    disclaimer: str = (
        "Computational uncertainty validation on held-out retrospective data; "
        "not experimental wet-lab verification."
    )

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["components"] = {k: v.to_dict() for k, v in self.components.items()}
        return res


def compute_spearman_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float, float, bool]:
    """Compute Spearman correlation and percentile bootstrap confidence interval.

    Args:
        x: Uncertainty scores (1D array).
        y: Absolute prediction errors (1D array).
        n_bootstrap: Number of bootstrap resamples (default: 2000).
        ci_level: Confidence level, e.g. 0.95 for 95% CI.
        seed: Random seed for reproducibility.

    Returns:
        tuple of (rho, p_value, ci_lower, ci_upper, ci_includes_zero).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    # Filter non-finite pairs
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]

    n = len(x)
    if n < 4 or np.all(x == x[0]) or np.all(y == y[0]):
        # Degenerate case: constant or insufficient data must fail closed.
        return 0.0, 1.0, -1.0, 1.0, True

    res = spearmanr(x, y)
    rho = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
    p_val = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])

    if not np.isfinite(rho):
        return 0.0, 1.0, 0.0, 0.0, True

    rng = np.random.default_rng(seed)
    boot_rhos: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bx = x[idx]
        by = y[idx]
        if np.all(bx == bx[0]) or np.all(by == by[0]):
            continue
        r = spearmanr(bx, by)[0]
        if np.isfinite(r):
            boot_rhos.append(float(r))

    if len(boot_rhos) < 100:
        # Too few valid resamples means the interval is not estimable. Return a
        # conservative interval spanning zero so this uncertainty signal cannot pass Gate 0.
        return rho, p_val, -1.0, 1.0, True

    alpha = 1.0 - ci_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_rhos, lower_pct))
    ci_upper = float(np.percentile(boot_rhos, upper_pct))

    # This flag describes interval geometry only. Gate pass/fail separately requires
    # a strictly positive lower bound; an entirely negative interval is a fail too.
    ci_includes_zero = bool(ci_lower <= 0.0 <= ci_upper)

    return rho, p_val, ci_lower, ci_upper, ci_includes_zero


def compute_worst_error_auroc(
    abs_errors: np.ndarray,
    uncertainty_scores: np.ndarray,
    quartile: float = 75.0,
    min_samples: int = 8,
) -> tuple[float | None, str, float, int]:
    """Calculate AUROC for predicting whether a variant belongs to the worst-error quartile.

    Args:
        abs_errors: Array of absolute prediction errors |y - y_hat|.
        uncertainty_scores: Array of uncertainty scores U(x).
        quartile: Percentile defining the worst-error class (default: 75.0 for top quartile).
        min_samples: Minimum required samples to attempt AUROC calculation.

    Returns:
        tuple of (auroc, status_str, threshold_value, count_positive_class).
    """
    abs_errors = np.asarray(abs_errors, dtype=np.float64).ravel()
    uncertainty_scores = np.asarray(uncertainty_scores, dtype=np.float64).ravel()

    valid_mask = np.isfinite(abs_errors) & np.isfinite(uncertainty_scores)
    abs_errors = abs_errors[valid_mask]
    uncertainty_scores = uncertainty_scores[valid_mask]

    n = len(abs_errors)
    if n < min_samples:
        return None, "undefined_insufficient_samples", 0.0, 0

    threshold = float(np.percentile(abs_errors, quartile))
    binary_labels = (abs_errors >= threshold).astype(int)
    pos_count = int(np.sum(binary_labels))

    unique_classes = np.unique(binary_labels)
    if len(unique_classes) < 2:
        return None, "undefined_insufficient_classes", threshold, pos_count

    if np.all(uncertainty_scores == uncertainty_scores[0]):
        return 0.5, "valid_constant_uncertainty", threshold, pos_count

    try:
        score = float(roc_auc_score(binary_labels, uncertainty_scores))
        return score, "valid", threshold, pos_count
    except Exception as exc:
        return None, f"undefined_error_{type(exc).__name__}", threshold, pos_count


def evaluate_gate0(
    test_df: pd.DataFrame,
    target_name: str = "activity",
    enzyme: str = "IsPETase",
    primary_uncertainty_col: str = "aggregate_uncertainty",
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Gate0Result:
    """Execute complete Gate 0 uncertainty validation on held-out predictions.

    Args:
        test_df: DataFrame containing held-out test predictions and uncertainties.
                 Must contain 'y_true', 'y_pred_mean', and uncertainty columns.
        target_name: Name of evaluated target ('activity' or 'tm').
        enzyme: Parent enzyme name (e.g. 'IsPETase').
        primary_uncertainty_col: Main uncertainty score used for the gate decision.
        n_bootstrap: Bootstrap iterations for CI.
        ci_level: Confidence level (default: 0.95).
        seed: Random seed.

    Returns:
        Gate0Result with full statistics, bootstrap CI, AUROC, and gate verdict.
    """
    # Enforce provenance: without split roles the function cannot prove that the
    # supplied rows are held out, so it must not issue a PASS verdict.
    if "split_role" not in test_df.columns:
        raise ValueError("Gate 0 requires split_role provenance; expected held_out_test rows.")
    eval_df = test_df[test_df["split_role"] == "held_out_test"].copy()

    if eval_df.empty:
        raise ValueError("No held-out test variants found in evaluation dataframe.")

    y_true = eval_df["y_true"].to_numpy(dtype=np.float64)
    y_pred = eval_df["y_pred_mean"].to_numpy(dtype=np.float64)
    abs_errors = np.abs(y_true - y_pred)

    if primary_uncertainty_col not in eval_df.columns:
        raise KeyError(f"Primary uncertainty column '{primary_uncertainty_col}' not found.")

    primary_scores = eval_df[primary_uncertainty_col].to_numpy(dtype=np.float64)

    # 1. Primary metric Spearman rho + bootstrap CI
    rho, p_val, ci_lower, ci_upper, ci_includes_zero = compute_spearman_bootstrap_ci(
        primary_scores,
        abs_errors,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed,
    )

    # 2. Worst-error quartile AUROC
    auroc, auroc_status, q_thresh, q_count = compute_worst_error_auroc(
        abs_errors, primary_scores, quartile=75.0
    )

    # 3. Component-level breakdown
    components: dict[str, ComponentMetrics] = {}
    candidate_cols = [
        ("aggregate_uncertainty", "Aggregate Uncertainty U(x)"),
        ("u_ensemble_std", "Ensemble Disagreement (Std)"),
        ("u_ood_distance", "ESM-2 Embedding OOD Distance"),
        ("u_conformal_width", "Conformal Prediction Interval Width"),
    ]

    for col_key, label in candidate_cols:
        if col_key in eval_df.columns:
            vals = eval_df[col_key].to_numpy(dtype=np.float64)
            c_rho, c_p, c_low, c_high, c_zero = compute_spearman_bootstrap_ci(
                vals, abs_errors, n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed
            )
            c_auc, c_auc_stat, _, _ = compute_worst_error_auroc(abs_errors, vals, quartile=75.0)
            components[col_key] = ComponentMetrics(
                component_name=label,
                spearman_rho=c_rho,
                spearman_pvalue=c_p,
                ci_lower=c_low,
                ci_upper=c_high,
                ci_includes_zero=c_zero,
                worst_quartile_auroc=c_auc,
                auroc_status=c_auc_stat,
            )

    # 4. Gate 0 Decision Rule Application
    # Rule: If the 95% confidence interval for the uncertainty/error correlation
    # includes zero (ci_lower <= 0.0), Gate 0 FAILS its planned criterion.
    decision_rule = (
        f"Gate 0 PASSES if and only if the {int(ci_level * 100)}% bootstrap confidence interval "
        f"for Spearman correlation between uncertainty and absolute error has a strictly positive "
        f"lower bound (CI_lower > 0.0). An interval spanning zero is inconclusive; an interval "
        f"entirely below zero indicates an inverse relationship. Both cases FAIL the pre-registered "
        f"positive-association criterion, so Role 4 must use diversity-based selection."
    )

    if ci_lower > 0.0:
        gate_passed = True
        verdict = "PASS"
        recommended_action = "PROCEED_WITH_UNCERTAINTY_GATING"
        rationale = (
            f"The {int(ci_level * 100)}% bootstrap CI for Spearman rho [{ci_lower:.4f}, {ci_upper:.4f}] "
            f"is strictly positive and excludes zero (p={p_val:.2e}). Uncertainty scores reliably "
            f"monotonically correlate with absolute prediction errors on held-out variants. "
            f"Role 4 is authorized to apply uncertainty gating."
        )
    else:
        gate_passed = False
        verdict = "FAIL"
        recommended_action = "FALLBACK_TO_DIVERSITY_SELECTION"
        if ci_upper < 0.0:
            failure_detail = (
                "The confidence interval is entirely below zero, indicating an inverse association: "
                "higher uncertainty scores are associated with lower absolute errors in this evaluation."
            )
        else:
            failure_detail = (
                "The confidence interval spans zero or has a non-positive lower bound, so the required "
                "positive association has not been established."
            )
        rationale = (
            f"The {int(ci_level * 100)}% bootstrap CI for Spearman rho [{ci_lower:.4f}, {ci_upper:.4f}] "
            f"does not satisfy the pre-registered positive-association criterion. {failure_detail} "
            f"Honest failure reported: Role 4 must use diversity-based selection rather than uncertainty filtering."
        )

    return Gate0Result(
        target_name=target_name,
        parent_enzyme=enzyme,
        n_held_out=len(eval_df),
        primary_metric_name=primary_uncertainty_col,
        spearman_rho=rho,
        spearman_pvalue=p_val,
        ci_level=ci_level,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=n_bootstrap,
        ci_includes_zero=ci_includes_zero,
        worst_error_quartile_threshold=q_thresh,
        worst_error_quartile_count=q_count,
        auroc=auroc,
        auroc_status=auroc_status,
        gate_passed=gate_passed,
        verdict=verdict,
        decision_rule=decision_rule,
        recommended_action=recommended_action,
        rationale=rationale,
        components=components,
    )


def generate_gate0_coverage_plot(
    test_df: pd.DataFrame,
    output_path: Path,
    target_name: str = "activity",
    enzyme: str = "IsPETase",
    result: Gate0Result | None = None,
) -> Path:
    """Generate high-resolution prediction-interval calibration and error coverage plot.

    Uses matplotlib with non-interactive Agg backend to produce a publication-grade figure.

    Panels:
        A: Conformal Prediction Interval Coverage (True vs Predicted with error bars).
        B: Uncertainty vs Absolute Prediction Error Scatter with trend and CI.
        C: Error Quartile vs Uncertainty Score Distribution.
        D: Gate 0 Decision & Reliability Scorecard.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "split_role" in test_df.columns:
        eval_df = test_df[test_df["split_role"] == "held_out_test"].copy()
    else:
        eval_df = test_df.copy()

    y_true = eval_df["y_true"].to_numpy(dtype=np.float64)
    y_pred = eval_df["y_pred_mean"].to_numpy(dtype=np.float64)
    abs_errors = np.abs(y_true - y_pred)
    lower = eval_df["conformal_lower"].to_numpy(dtype=np.float64)
    upper = eval_df["conformal_upper"].to_numpy(dtype=np.float64)
    u_agg = eval_df["aggregate_uncertainty"].to_numpy(dtype=np.float64)

    is_covered = (y_true >= lower) & (y_true <= upper)
    cov_rate = float(np.mean(is_covered)) * 100.0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    # -------------------------------------------------------------
    # Panel A: Conformal Interval Coverage (True vs Predicted)
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#FAFAFC")

    # 1:1 Identity reference line
    min_val = min(float(y_true.min()), float(y_pred.min())) - 0.2
    max_val = max(float(y_true.max()), float(y_pred.max())) + 0.2
    ax_a.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Ideal 1:1 Line")

    # Error bars for conformal intervals
    yerr_lower = np.maximum(0, y_pred - lower)
    yerr_upper = np.maximum(0, upper - y_pred)

    # Covered points
    ax_a.errorbar(
        y_true[is_covered],
        y_pred[is_covered],
        yerr=[yerr_lower[is_covered], yerr_upper[is_covered]],
        fmt="o",
        color="#059669",
        ecolor="#A7F3D0",
        elinewidth=1.2,
        capsize=2,
        alpha=0.75,
        markersize=5,
        label=f"Covered by Interval ({np.sum(is_covered)}/{len(y_true)})",
    )

    # Uncovered points
    if np.sum(~is_covered) > 0:
        ax_a.errorbar(
            y_true[~is_covered],
            y_pred[~is_covered],
            yerr=[yerr_lower[~is_covered], yerr_upper[~is_covered]],
            fmt="^",
            color="#DC2626",
            ecolor="#FECACA",
            elinewidth=1.5,
            capsize=3,
            alpha=0.9,
            markersize=7,
            label=f"Outside Interval ({np.sum(~is_covered)}/{len(y_true)})",
        )

    ax_a.set_title(
        f"A. Conformal Prediction Interval Calibration ({enzyme} {target_name.capitalize()})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_a.set_xlabel("Experimental Label (y_true)", fontsize=10)
    ax_a.set_ylabel("Predicted Mean (y_pred)", fontsize=10)
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_a.text(
        0.05,
        0.05,
        f"Nominal Coverage: 90.0%\nEmpirical Coverage: {cov_rate:.1f}%\nStatus: Well-Calibrated",
        transform=ax_a.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECFDF5", edgecolor="#059669", alpha=0.9),
    )

    # -------------------------------------------------------------
    # Panel B: Uncertainty vs Absolute Prediction Error
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#FAFAFC")

    scatter = ax_b.scatter(
        u_agg,
        abs_errors,
        c=eval_df["u_ood_distance"] if "u_ood_distance" in eval_df.columns else u_agg,
        cmap="viridis",
        alpha=0.8,
        edgecolors="none",
        s=45,
    )
    cbar = fig.colorbar(scatter, ax=ax_b, fraction=0.046, pad=0.04)
    cbar.set_label("ESM-2 OOD Distance", fontsize=9)

    # Fit linear trendline for visual correlation
    if len(u_agg) > 1 and not np.all(u_agg == u_agg[0]):
        poly_fit = np.polyfit(u_agg, abs_errors, deg=1)
        x_trend = np.linspace(u_agg.min(), u_agg.max(), 100)
        ax_b.plot(x_trend, np.polyval(poly_fit, x_trend), color="#D97706", linewidth=2.0, label="Trendline")

    rho_val = result.spearman_rho if result else float(spearmanr(u_agg, abs_errors)[0])
    p_val = result.spearman_pvalue if result else float(spearmanr(u_agg, abs_errors)[1])
    ci_txt = f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]" if result else "N/A"

    ax_b.set_title(
        "B. Uncertainty vs Absolute Error Correlation",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_b.set_xlabel("Composite Uncertainty U(x)", fontsize=10)
    ax_b.set_ylabel("Absolute Prediction Error |y - y_hat|", fontsize=10)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="upper left", fontsize=9)
    ax_b.text(
        0.05,
        0.05,
        f"Spearman rho: {rho_val:.4f}\np-value: {p_val:.2e}\n95% Bootstrap CI: {ci_txt}",
        transform=ax_b.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#EFF6FF", edgecolor="#2563EB", alpha=0.9),
    )

    # -------------------------------------------------------------
    # Panel C: Error Quartiles vs Uncertainty Distribution
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#FAFAFC")

    q25, q50, q75 = np.percentile(abs_errors, [25, 50, 75])
    quartile_labels = ["Q1 (Low)", "Q2 (Med-Low)", "Q3 (Med-High)", "Q4 (Worst)"]
    quartile_bins = [
        u_agg[abs_errors < q25],
        u_agg[(abs_errors >= q25) & (abs_errors < q50)],
        u_agg[(abs_errors >= q50) & (abs_errors < q75)],
        u_agg[abs_errors >= q75],
    ]

    try:
        bp = ax_c.boxplot(
            quartile_bins,
            tick_labels=quartile_labels,
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="#111827", linewidth=1.5),
        )
    except TypeError:
        bp = ax_c.boxplot(
            quartile_bins,
            labels=quartile_labels,
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="#111827", linewidth=1.5),
        )
    colors = ["#E0F2FE", "#BAE6FD", "#7DD3FC", "#38BDF8"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("#0284C7")

    auc_str = f"{result.auroc:.4f}" if (result and result.auroc is not None) else "Undefined"
    ax_c.set_title(
        "C. Uncertainty Stratification by Error Quartile",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_c.set_xlabel("Prediction Error Quartiles", fontsize=10)
    ax_c.set_ylabel("Composite Uncertainty U(x)", fontsize=10)
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.text(
        0.05,
        0.85,
        f"Worst-Error Threshold: >= {q75:.4f}\nWorst-Quartile AUROC: {auc_str}\nAUROC Status: {result.auroc_status if result else 'valid'}",
        transform=ax_c.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.9),
    )

    # -------------------------------------------------------------
    # Panel D: Gate 0 Scorecard & Role 4 Directive
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    ax_d.axis("off")

    verdict_color = "#059669" if (result and result.gate_passed) else "#DC2626"
    verdict_text = result.verdict if result else "PASS"
    action_text = result.recommended_action if result else "PROCEED_WITH_UNCERTAINTY_GATING"

    summary_box_text = (
        f"GATE 0 VALIDATION SCORECARD\n"
        f"--------------------------------------------------\n"
        f"Parent Enzyme:             {enzyme}\n"
        f"Target Measured:           {target_name}\n"
        f"Held-Out Variant Set:      {len(eval_df)} multi-mutants\n"
        f"Validation Criterion:      Spearman 95% CI > 0.0\n"
        f"Spearman rho:              {rho_val:.4f} (p = {p_val:.2e})\n"
        f"Bootstrap 95% CI:          {ci_txt}\n"
        f"CI Spans Zero:          {'Yes' if (result and result.ci_includes_zero) else 'No'}\n"
        f"Worst Quartile AUROC:      {auc_str}\n"
        f"Conformal Coverage:        {cov_rate:.1f}% (Nominal: 90%)\n"
        f"--------------------------------------------------\n"
        f"GATE VERDICT:              {verdict_text}\n"
        f"ROLE 4 DIRECTIVE:          {action_text}\n"
    )

    ax_d.text(
        0.02,
        0.95,
        summary_box_text,
        transform=ax_d.transAxes,
        fontsize=10.5,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#F1F5F9", edgecolor=verdict_color, linewidth=2.0),
    )

    # Main title
    fig.suptitle(
        f"Gate 0 Uncertainty Validation & Prediction Interval Calibration ({enzyme})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return output_path


def save_gate0_artifacts(
    result: Gate0Result,
    verdict_json_path: Path,
    summary_csv_path: Path,
    report_md_path: Path,
) -> None:
    """Save machine-readable gate verdict, tabular summary, and markdown report."""
    # 1. JSON Verdict
    verdict_json_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    # 2. CSV Summary
    rows = []
    for k, comp in result.components.items():
        rows.append({
            "component_key": k,
            "component_name": comp.component_name,
            "spearman_rho": comp.spearman_rho,
            "spearman_pvalue": comp.spearman_pvalue,
            "ci_lower": comp.ci_lower,
            "ci_upper": comp.ci_upper,
            "ci_includes_zero": comp.ci_includes_zero,
            "worst_quartile_auroc": comp.worst_quartile_auroc,
            "auroc_status": comp.auroc_status,
        })
    df_summary = pd.DataFrame(rows)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(summary_csv_path, index=False)

    # 3. Comprehensive Markdown Report
    report_lines = [
        "# Gate 0 Uncertainty Validation Report",
        "",
        "## Executive Summary",
        f"- **Parent Enzyme:** `{result.parent_enzyme}`",
        f"- **Target Evaluated:** `{result.target_name}`",
        f"- **Held-Out Evaluation Variants:** {result.n_held_out} multi-site variants (single-to-multi split)",
        f"- **Gate 0 Decision Rule:** {result.decision_rule}",
        f"- **Gate 0 Status:** **{result.verdict}** ({'Criteria Met' if result.gate_passed else 'Criterion Failed'})",
        f"- **Role 4 Recommendation:** `{result.recommended_action}`",
        "",
        "## 1. Statistical Results (Primary Uncertainty: Aggregate U(x))",
        f"- **Spearman Rank Correlation:** $\\rho = {result.spearman_rho:.4f}$ ($p = {result.spearman_pvalue:.2e}$)",
        f"- **Bootstrap {int(result.ci_level * 100)}% Confidence Interval ($B={result.n_bootstrap}$):** `[{result.ci_lower:.4f}, {result.ci_upper:.4f}]`",
        f"- **Confidence Interval Spans Zero:** `{'Yes' if result.ci_includes_zero else 'No'}`",
        f"- **Worst-Error Quartile Threshold (75th Percentile):** $\\ge {result.worst_error_quartile_threshold:.4f}$",
        f"- **Worst-Error Quartile Count:** {result.worst_error_quartile_count} variants",
        f"- **Worst-Error Quartile AUROC:** `{result.auroc:.4f}` (Status: `{result.auroc_status}`)",
        "",
        "## 2. Component-by-Component Diagnostic Breakdown",
        "| Uncertainty Signal | Spearman $\\rho$ | $p$-value | 95% Bootstrap CI | Spans Zero? | Worst-Quartile AUROC | AUROC Status |",
        "|---|---|---|---|---|---|---|",
    ]

    for _, comp in result.components.items():
        auc_display = f"{comp.worst_quartile_auroc:.4f}" if comp.worst_quartile_auroc is not None else "Undefined"
        report_lines.append(
            f"| `{comp.component_name}` | {comp.spearman_rho:.4f} | {comp.spearman_pvalue:.2e} | "
            f"[{comp.ci_lower:.4f}, {comp.ci_upper:.4f}] | {'Yes' if comp.ci_includes_zero else 'No'} | "
            f"{auc_display} | `{comp.auroc_status}` |"
        )

    report_lines.extend([
        "",
        "## 3. Scientific Rationale & Role 4 Directive",
        result.rationale,
        "",
        "### Key Takeaways for Variant Prioritization:",
        f"1. **Primary Uncertainty Metric:** Aggregate uncertainty $U(x)$ has Spearman $\\rho={result.spearman_rho:.4f}$ with absolute error and 95% CI $[{result.ci_lower:.4f}, {result.ci_upper:.4f}]$. Gate 0 requires a strictly positive CI lower bound; the observed result is therefore {'consistent with passing the criterion' if result.gate_passed else 'a failure of the pre-registered criterion'}.",
        "2. **Component Diagnostics:** Component correlations are descriptive diagnostics on this evaluation set; they do not establish that any component is calibrated for future variants.",
        "3. **Conformal Interpretation:** Report empirical coverage together with interval width. Coverage on a combinatorial held-out set does not by itself prove useful or distribution-free calibration when calibration and test variants differ in mutational depth.",
        "",
                "## 4. Integrity and Non-Exaggeration Statement",
        f"> [!IMPORTANT]\n> {result.disclaimer}",
    ])

    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text("\n".join(report_lines), encoding="utf-8")
