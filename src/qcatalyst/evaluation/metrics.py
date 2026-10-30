"""Evaluation metrics module for Q-Catalyst.

Provides ranking, enrichment, and correlation metrics with bootstrap confidence intervals
for benchmarking variant prioritisation algorithms (Arms A0–A7).

Supported Metrics:
- Precision@K (K = 5, 10, 20)
- Enrichment Factor@K (EF@K)
- Normalized Discounted Cumulative Gain@K (NDCG@K)
- Spearman Rank Correlation (rho)
- Kendall Tau Concordance Correlation (tau)
- Bootstrap Percentile Confidence Intervals (95% CI)
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from scipy.stats import kendalltau, spearmanr


def precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 10,
    threshold: float = 1.0,
) -> float:
    """Calculate Precision@K: fraction of top-K variants satisfying y_true > threshold.

    Args:
        y_true: Experimental target values (1D array).
        y_score: Model prioritisation scores (higher score = prioritized first).
        k: Cutoff rank for top candidates.
        threshold: Minimum experimental value to qualify as a 'good variant' (default: 1.0).

    Returns:
        Precision in range [0.0, 1.0].
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_t = y_true[mask]
    y_s = y_score[mask]

    n = len(y_t)
    if n == 0 or k <= 0:
        return 0.0

    k_eff = min(k, n)
    # Sort descending by score
    top_indices = np.argsort(y_s)[::-1][:k_eff]
    hits = np.sum(y_t[top_indices] > threshold)
    return float(hits / k_eff)


def enrichment_factor_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 10,
    threshold: float = 1.0,
) -> float:
    """Calculate Enrichment Factor (EF@K): fold-improvement over random screening.

    EF@K = Precision@K / Base_Rate
    where Base_Rate = fraction of hits across the entire evaluation pool.

    Args:
        y_true: Experimental target values.
        y_score: Model prioritisation scores.
        k: Cutoff rank.
        threshold: Hit definition threshold.

    Returns:
        Enrichment factor (>= 0.0). If base rate is 0, returns 0.0.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_t = y_true[mask]

    if len(y_t) == 0:
        return 0.0

    base_rate = float(np.mean(y_t > threshold))
    if base_rate <= 0.0:
        return 0.0

    p_k = precision_at_k(y_true, y_score, k=k, threshold=threshold)
    return float(p_k / base_rate)


def dcg_at_k(relevance_scores: np.ndarray, k: int) -> float:
    """Calculate Discounted Cumulative Gain at rank K."""
    relevance_scores = np.asarray(relevance_scores, dtype=np.float64)
    k_eff = min(k, len(relevance_scores))
    if k_eff <= 0:
        return 0.0

    discounts = np.log2(np.arange(2, k_eff + 2))
    return float(np.sum(relevance_scores[:k_eff] / discounts))


def ndcg_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 10,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K).

    Uses true continuous experimental value as relevance (clipped to non-negative).

    Args:
        y_true: True experimental values (relevance scores).
        y_score: Model prioritisation scores.
        k: Cutoff rank.

    Returns:
        NDCG@K in range [0.0, 1.0].
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_t = y_true[mask]
    y_s = y_score[mask]

    n = len(y_t)
    if n == 0 or k <= 0:
        return 0.0

    k_eff = min(k, n)
    # Relevance cannot be negative for standard NDCG
    rel = np.maximum(0.0, y_t)

    # Actual ranking by model score
    sorted_order = np.argsort(y_s)[::-1]
    actual_rel = rel[sorted_order]
    dcg = dcg_at_k(actual_rel, k_eff)

    # Ideal ranking by true relevance
    ideal_order = np.argsort(rel)[::-1]
    ideal_rel = rel[ideal_order]
    idcg = dcg_at_k(ideal_rel, k_eff)

    if idcg <= 0.0:
        return 0.0

    return float(min(1.0, max(0.0, dcg / idcg)))


def spearman_rank_correlation(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, float]:
    """Calculate Spearman rank correlation rho and two-sided p-value."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_t = y_true[mask]
    y_s = y_score[mask]

    if len(y_t) < 3 or np.all(y_t == y_t[0]) or np.all(y_s == y_s[0]):
        return 0.0, 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = spearmanr(y_s, y_t)
        rho = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
        p_val = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])

    if not np.isfinite(rho):
        return 0.0, 1.0
    return rho, p_val


def kendall_tau_correlation(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, float]:
    """Calculate Kendall rank concordance tau and two-sided p-value."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_t = y_true[mask]
    y_s = y_score[mask]

    if len(y_t) < 3 or np.all(y_t == y_t[0]) or np.all(y_s == y_s[0]):
        return 0.0, 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = kendalltau(y_s, y_t)
        tau = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
        p_val = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])

    if not np.isfinite(tau):
        return 0.0, 1.0
    return tau, p_val


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute point estimate and percentile bootstrap confidence interval for any ranking metric.

    Args:
        y_true: True experimental values.
        y_score: Model prioritisation scores.
        metric_func: Callable accepting (y_true_sub, y_score_sub) and returning a scalar float.
        n_bootstrap: Number of bootstrap iterations.
        ci_level: Confidence level (default: 0.95).
        seed: Random seed.

    Returns:
        tuple of (point_estimate, ci_lower, ci_upper).
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_t = y_true[mask]
    y_s = y_score[mask]

    n = len(y_t)
    if n < 4:
        pt = float(metric_func(y_t, y_s))
        return pt, pt, pt

    point_estimate = float(metric_func(y_t, y_s))

    rng = np.random.default_rng(seed)
    boot_vals: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        sample_val = metric_func(y_t[idx], y_s[idx])
        if np.isfinite(sample_val):
            boot_vals.append(float(sample_val))

    if len(boot_vals) < 50:
        return point_estimate, point_estimate, point_estimate

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(boot_vals, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_vals, 100.0 * (1.0 - alpha / 2.0)))

    return point_estimate, ci_lower, ci_upper


def evaluate_all_ranking_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 1.0,
    k_list: list[int] | None = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Evaluate full suite of prioritisation metrics with bootstrap confidence intervals.

    Returns a dictionary mapping metric names to {'value': ..., 'ci_lower': ..., 'ci_upper': ...}.
    """
    if k_list is None:
        k_list = [5, 10, 20]

    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    results: dict[str, dict[str, float]] = {}

    # 1. Precision@K
    for k in k_list:
        pt, low, high = compute_bootstrap_ci(
            y_true,
            y_score,
            metric_func=lambda yt, ys, _k=k: precision_at_k(yt, ys, k=_k, threshold=threshold),
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )
        results[f"precision@{k}"] = {"value": pt, "ci_lower": low, "ci_upper": high}

    # 2. Enrichment Factor@K
    for k in k_list:
        pt, low, high = compute_bootstrap_ci(
            y_true,
            y_score,
            metric_func=lambda yt, ys, _k=k: enrichment_factor_at_k(yt, ys, k=_k, threshold=threshold),
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )
        results[f"enrichment_factor@{k}"] = {"value": pt, "ci_lower": low, "ci_upper": high}

    # 3. NDCG@K
    for k in k_list:
        pt, low, high = compute_bootstrap_ci(
            y_true,
            y_score,
            metric_func=lambda yt, ys, _k=k: ndcg_at_k(yt, ys, k=_k),
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )
        results[f"ndcg@{k}"] = {"value": pt, "ci_lower": low, "ci_upper": high}

    # 4. Spearman Rho
    rho_pt, rho_low, rho_high = compute_bootstrap_ci(
        y_true,
        y_score,
        metric_func=lambda yt, ys: spearman_rank_correlation(yt, ys)[0],
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed,
    )
    results["spearman_rho"] = {"value": rho_pt, "ci_lower": rho_low, "ci_upper": rho_high}

    # 5. Kendall Tau
    tau_pt, tau_low, tau_high = compute_bootstrap_ci(
        y_true,
        y_score,
        metric_func=lambda yt, ys: kendall_tau_correlation(yt, ys)[0],
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed,
    )
    results["kendall_tau"] = {"value": tau_pt, "ci_lower": tau_low, "ci_upper": tau_high}

    return results
