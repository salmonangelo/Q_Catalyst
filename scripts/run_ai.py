"""
Script 2: Train Protein AI models, compute uncertainties, and evaluate Gate 0.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.qcatalyst.protein_ai.features import build_feature_matrix, compute_esm_zero_shot_score
from src.qcatalyst.protein_ai.models import PETaseEnsemblePredictor
from src.qcatalyst.uncertainty.estimator import UncertaintyEstimator
from src.qcatalyst.uncertainty.gate0 import evaluate_gate_0


def main():
    print("=" * 60)
    print("Q-CATALYST: Step 2 - Protein AI & Uncertainty Quantification")
    print("=" * 60)

    curated_path = PROJECT_ROOT / "data" / "curated" / "variants.parquet"
    if not curated_path.exists():
        raise FileNotFoundError(f"Missing {curated_path}. Run prepare_data.py first.")

    df = pd.read_parquet(curated_path)
    X = build_feature_matrix(df)
    y_act = df["relative_activity"].to_numpy(dtype=float)
    y_tm = df["delta_tm"].to_numpy(dtype=float)

    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()

    # Fit ensemble predictor
    predictor = PETaseEnsemblePredictor(seeds=[42, 123, 456, 789, 999])
    predictor.fit(X[train_mask], y_act[train_mask], y_tm[train_mask])

    # Predict on entire dataset
    pred_act, std_act, _ = predictor.predict_activity(X)
    pred_delta_tm = predictor.predict_delta_tm(X)

    # Uncertainty Quantification
    estimator = UncertaintyEstimator(confidence_level=0.90)
    estimator.calibrate_conformal(y_act[val_mask], pred_act[val_mask])

    lower_bounds, upper_bounds, interval_widths = estimator.compute_conformal_intervals(pred_act)
    ood_dists = estimator.compute_ood_distance(X)
    min_dists = X[:, 5] # Column 5 is min_distance_triad
    mut_counts = X[:, 4] # Column 4 is mutation_count
    struct_risks = estimator.compute_structural_risk(min_dists, mut_counts)

    U_scores = estimator.compute_composite_uncertainty(
        std_act, interval_widths, ood_dists, struct_risks
    )

    # Gate 0 Evaluation on Test split
    test_errors = np.abs(y_act[test_mask] - pred_act[test_mask])
    test_u = U_scores[test_mask]
    gate0_results = evaluate_gate_0(test_u, test_errors, bootstrap_iterations=1000)

    print("Gate 0 Hypothesis Test Results:")
    print(f"  Spearman rho: {gate0_results['spearman_rho']:.4f} (p-value: {gate0_results['p_value']:.4f})")
    print(f"  95% Bootstrap CI: [{gate0_results['ci_lower_95']:.4f}, {gate0_results['ci_upper_95']:.4f}]")
    print(f"  AUROC (worst-error quartile): {gate0_results['auroc_worst_quartile']:.4f}")
    print(f"  Outcome: {gate0_results['recommendation']}")

    # ESM zero shot scores
    esm_scores = [compute_esm_zero_shot_score(m) for m in df["mutations"]]

    # Save Predictions Parquet
    df_pred = pd.DataFrame({
        "variant_id": df["variant_id"],
        "pred_activity": np.round(pred_act, 3),
        "pred_interval_lower": np.round(lower_bounds, 3),
        "pred_interval_upper": np.round(upper_bounds, 3),
        "pred_delta_tm": np.round(pred_delta_tm, 2),
        "zero_shot_esm_score": esm_scores,
        "split": df["split"],
        "model_version": "Ensemble_XGBoost_Ridge_v1.0"
    })

    # Save Uncertainty Parquet
    df_unc = pd.DataFrame({
        "variant_id": df["variant_id"],
        "aggregate_uncertainty": np.round(U_scores, 3),
        "ensemble_std": np.round(std_act, 3),
        "conformal_interval_width": np.round(interval_widths, 3),
        "ood_distance": np.round(ood_dists, 3),
        "structural_risk": np.round(struct_risks, 3),
        "gate0_pass": gate0_results["gate0_passed"],
        "abstention_flag": (U_scores > 0.75)
    })

    pred_dir = PROJECT_ROOT / "results" / "predictions"
    unc_dir = PROJECT_ROOT / "results" / "uncertainty"
    pred_dir.mkdir(parents=True, exist_ok=True)
    unc_dir.mkdir(parents=True, exist_ok=True)

    df_pred.to_parquet(pred_dir / "predictions.parquet", index=False)
    df_unc.to_parquet(unc_dir / "uncertainty.parquet", index=False)

    with open(unc_dir / "gate0_report.json", "w") as f:
        json.dump(gate0_results, f, indent=2)

    print(f"Saved predictions to: {pred_dir / 'predictions.parquet'}")
    print(f"Saved uncertainties to: {unc_dir / 'uncertainty.parquet'}")
    print(f"Saved Gate 0 report to: {unc_dir / 'gate0_report.json'}")
    print("Step 2 Complete!\n")


if __name__ == "__main__":
    main()
