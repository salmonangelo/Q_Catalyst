"""
Supervised ensemble predictor for activity and thermal stability of PETase variants.
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from src.qcatalyst.protein_ai.features import build_feature_matrix


class PETaseEnsemblePredictor:
    """
    Ensemble regressor combining Gradient Boosted Trees (XGBoost) and L2 Regularized Linear models (Ridge)
    trained across distinct random seeds to estimate prediction mean and epistemic disagreement.
    """
    def __init__(self, seeds: List[int] = [42, 123, 456, 789, 999]):
        self.seeds = seeds
        self.xgb_models: List[XGBRegressor] = []
        self.ridge_models: List[Ridge] = []
        self.tm_models: List[Ridge] = []
        self.feature_names = [
            "delta_charge", "delta_hydrophobicity", "delta_volume",
            "delta_polarity", "mutation_count", "min_distance_triad",
            "esm_score", "temp_scaled", "ph_scaled", "is_crystallized"
        ]

    def fit(self, X: np.ndarray, y_activity: np.ndarray, y_delta_tm: np.ndarray):
        """Fit the ensemble on training feature matrix and ground truth targets."""
        self.xgb_models.clear()
        self.ridge_models.clear()
        self.tm_models.clear()

        for s in self.seeds:
            # XGBoost activity model
            xgb = XGBRegressor(
                n_estimators=60,
                max_depth=3,
                learning_rate=0.08,
                random_state=s,
                n_jobs=1
            )
            xgb.fit(X, y_activity)
            self.xgb_models.append(xgb)

            # Ridge activity model
            ridge = Ridge(alpha=1.5, random_state=s)
            ridge.fit(X, y_activity)
            self.ridge_models.append(ridge)

            # Ridge delta_tm model
            ridge_tm = Ridge(alpha=2.0, random_state=s)
            ridge_tm.fit(X, y_delta_tm)
            self.tm_models.append(ridge_tm)

    def predict_activity(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            mean_pred: Ensemble average predicted relative activity
            std_pred: Ensemble standard deviation (epistemic disagreement)
            all_preds: Array of predictions across all ensemble members
        """
        all_preds = []
        for m in self.xgb_models:
            all_preds.append(m.predict(X))
        for m in self.ridge_models:
            all_preds.append(m.predict(X))

        all_preds_arr = np.array(all_preds) # Shape (n_models, n_samples)
        # Bounded to non-negative relative activity
        all_preds_arr = np.clip(all_preds_arr, 0.0, 10.0)
        mean_pred = np.mean(all_preds_arr, axis=0)
        std_pred = np.std(all_preds_arr, axis=0)

        return mean_pred, std_pred, all_preds_arr

    def predict_delta_tm(self, X: np.ndarray) -> np.ndarray:
        """Predicts thermal stability shift delta_Tm in °C."""
        preds = [m.predict(X) for m in self.tm_models]
        return np.mean(preds, axis=0)
