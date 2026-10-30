"""Evaluation, metrics, and benchmarking package for Q-Catalyst."""

from qcatalyst.evaluation.benchmarks import (
    BenchmarkRecord,
    load_quantum_results,
    run_benchmark_suite,
    save_benchmark_artifacts,
    score_arm_a0_random,
    score_arm_a1_esm1v,
    score_arm_a2_xgboost_structural,
)
from qcatalyst.evaluation.metrics import (
    compute_bootstrap_ci,
    dcg_at_k,
    enrichment_factor_at_k,
    evaluate_all_ranking_metrics,
    kendall_tau_correlation,
    ndcg_at_k,
    precision_at_k,
    spearman_rank_correlation,
)

__all__ = [
    # Metrics
    "precision_at_k",
    "enrichment_factor_at_k",
    "dcg_at_k",
    "ndcg_at_k",
    "spearman_rank_correlation",
    "kendall_tau_correlation",
    "compute_bootstrap_ci",
    "evaluate_all_ranking_metrics",
    # Benchmarking
    "BenchmarkRecord",
    "load_quantum_results",
    "score_arm_a0_random",
    "score_arm_a1_esm1v",
    "score_arm_a2_xgboost_structural",
    "run_benchmark_suite",
    "save_benchmark_artifacts",
]
