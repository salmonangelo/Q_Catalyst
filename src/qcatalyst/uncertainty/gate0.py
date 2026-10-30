"""
Gate 0: Formal statistical validation of predictive uncertainty against held-out errors.
"""
from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def evaluate_gate_0(
    uncertainty_scores: np.ndarray,
    absolute_errors: np.ndarray,
    bootstrap_iterations: int = 1000,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates pre-registered Gate 0 criteria:
    H1: Uncertainty U is positively correlated with prediction error.
    Returns statistical metrics, bootstrap 95% confidence intervals, and pass/fail status.
    """
    rng = np.random.default_rng(random_seed)
    n = len(uncertainty_scores)

    # Point estimate Spearman correlation
    rho_point, p_value = spearmanr(uncertainty_scores, absolute_errors)
    if np.isnan(rho_point):
        rho_point = 0.0

    # Non-parametric paired bootstrap for Spearman rho 95% CI
    bootstrap_rhos = []
    for _ in range(bootstrap_iterations):
        indices = rng.integers(0, n, size=n)
        sample_u = uncertainty_scores[indices]
        sample_err = absolute_errors[indices]
        if np.std(sample_u) > 1e-8 and np.std(sample_err) > 1e-8:
            r, _ = spearmanr(sample_u, sample_err)
            if not np.isnan(r):
                bootstrap_rhos.append(r)

    if bootstrap_rhos:
        ci_lower = float(np.percentile(bootstrap_rhos, 2.5))
        ci_upper = float(np.percentile(bootstrap_rhos, 97.5))
    else:
        ci_lower, ci_upper = float(rho_point - 0.1), float(rho_point + 0.1)

    # AUROC for identifying worst-error quartile (top 25% errors)
    threshold_worst_quartile = np.percentile(absolute_errors, 75)
    binary_worst = (absolute_errors >= threshold_worst_quartile).astype(int)

    if len(np.unique(binary_worst)) > 1:
        auroc = float(roc_auc_score(binary_worst, uncertainty_scores))
    else:
        auroc = 0.50

    # Pre-registered criterion: 95% CI strictly excludes zero and rho > 0
    passed = bool(ci_lower > 0.0 and rho_point > 0.15 and auroc > 0.60)

    return {
        "gate0_passed": passed,
        "spearman_rho": float(np.round(rho_point, 4)),
        "p_value": float(np.round(p_value, 4)),
        "ci_lower_95": float(np.round(ci_lower, 4)),
        "ci_upper_95": float(np.round(ci_upper, 4)),
        "auroc_worst_quartile": float(np.round(auroc, 4)),
        "sample_count": n,
        "recommendation": (
            "Gate 0 PASSED: Uncertainty is validated as error-informative; proceed with uncertainty-guided acquisition."
            if passed else
            "Gate 0 NOT VALIDATED: Uncertainty failed strictly positive CI criterion; fallback to stratified diversity selection."
        )
    }
