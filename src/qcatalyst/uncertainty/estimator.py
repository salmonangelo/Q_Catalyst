"""
Uncertainty quantification module combining ensemble disagreement,
conformal prediction intervals, out-of-distribution signals, and structural risk.
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class UncertaintyEstimator:
    """
    Computes rigorous calibrated uncertainty metrics for protein AI predictions.
    """
    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.conformal_q: float = 0.45 # Calibrated non-conformity quantile
        self.wt_feature_vector: np.ndarray = np.zeros(10)

    def calibrate_conformal(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calibrates residual-based conformal prediction interval on held-out calibration set.
        """
        residuals = np.abs(y_true - y_pred)
        n = len(residuals)
        if n > 0:
            p = np.clip(np.ceil((n + 1) * self.confidence_level) / n, 0.0, 1.0)
            self.conformal_q = float(np.quantile(residuals, p, method="higher"))
        else:
            self.conformal_q = 0.50

    def compute_conformal_intervals(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns lower bounds, upper bounds, and interval widths for 90% confidence level.
        """
        lower = np.clip(y_pred - self.conformal_q, 0.0, None)
        upper = y_pred + self.conformal_q
        width = upper - lower
        return lower, upper, width

    def compute_ood_distance(self, X: np.ndarray) -> np.ndarray:
        """
        Normalized Euclidean distance from the wild-type reference feature vector.
        """
        diffs = X - self.wt_feature_vector
        dists = np.linalg.norm(diffs, axis=1)
        max_d = np.max(dists) if np.max(dists) > 0 else 1.0
        return dists / max_d

    def compute_structural_risk(self, min_distances: np.ndarray, mutation_counts: np.ndarray) -> np.ndarray:
        """
        Risk is higher for mutations directly inside catalytic active site (Ser160/His237/Asp206)
        due to non-linear electrostatic sensitivity, or for high mutational loads.
        """
        # Close to triad (< 6 A) increases catalytic disruption risk
        proximity_risk = np.exp(-min_distances / 5.0)
        load_risk = np.clip(mutation_counts / 10.0, 0.0, 1.0)
        risk = 0.6 * proximity_risk + 0.4 * load_risk
        return np.clip(risk, 0.0, 1.0)

    def compute_composite_uncertainty(
        self,
        ensemble_std: np.ndarray,
        interval_width: np.ndarray,
        ood_dist: np.ndarray,
        struct_risk: np.ndarray,
        weights: Tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20)
    ) -> np.ndarray:
        """
        Fuses component signals into composite uncertainty score U in [0, 1].
        """
        # Normalize ensemble std
        std_norm = ensemble_std / (np.max(ensemble_std) if np.max(ensemble_std) > 0 else 1.0)
        # Normalize interval width
        width_norm = interval_width / (np.max(interval_width) if np.max(interval_width) > 0 else 1.0)

        U = (
            weights[0] * std_norm +
            weights[1] * width_norm +
            weights[2] * ood_dist +
            weights[3] * struct_risk
        )
        return np.clip(U, 0.0, 1.0)
