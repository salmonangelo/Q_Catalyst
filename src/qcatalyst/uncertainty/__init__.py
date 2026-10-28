"""Uncertainty estimation, calibration, and Gate 0 validation module for Q-Catalyst."""

from qcatalyst.uncertainty.calibrate import (
    ConformalIntervalResult,
    SplitConformalCalibrator,
)
from qcatalyst.uncertainty.estimate import (
    EnsembleConfig,
    EnsemblePredictor,
    OODDetector,
    compute_aggregate_uncertainty,
    compute_structural_risk,
)
from qcatalyst.uncertainty.gate import (
    ComponentMetrics,
    Gate0Result,
    compute_spearman_bootstrap_ci,
    compute_worst_error_auroc,
    evaluate_gate0,
    generate_gate0_coverage_plot,
    save_gate0_artifacts,
)

__all__ = [
    # Conformal Calibration
    "ConformalIntervalResult",
    "SplitConformalCalibrator",
    # Ensemble & OOD Estimation
    "EnsembleConfig",
    "EnsemblePredictor",
    "OODDetector",
    "compute_structural_risk",
    "compute_aggregate_uncertainty",
    # Gate 0 Validation
    "ComponentMetrics",
    "Gate0Result",
    "compute_spearman_bootstrap_ci",
    "compute_worst_error_auroc",
    "evaluate_gate0",
    "generate_gate0_coverage_plot",
    "save_gate0_artifacts",
]
