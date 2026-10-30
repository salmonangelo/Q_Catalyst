"""
Script 3: Apply structural proximity filter, acquisition function, and budget gate.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.qcatalyst.protein_ai.features import build_feature_matrix
from src.qcatalyst.acquisition.selector import select_candidates_for_chemistry


def main():
    print("=" * 60)
    print("Q-CATALYST: Step 3 - Candidate Acquisition & Budget Gate")
    print("=" * 60)

    curated_path = PROJECT_ROOT / "data" / "curated" / "variants.parquet"
    pred_path = PROJECT_ROOT / "results" / "predictions" / "predictions.parquet"
    unc_path = PROJECT_ROOT / "results" / "uncertainty" / "uncertainty.parquet"
    gate0_path = PROJECT_ROOT / "results" / "uncertainty" / "gate0_report.json"

    df_variants = pd.read_parquet(curated_path)
    df_pred = pd.read_parquet(pred_path)
    df_unc = pd.read_parquet(unc_path)

    with open(gate0_path, "r") as f:
        gate0 = json.load(f)

    X = build_feature_matrix(df_variants)
    pred_act = df_pred["pred_activity"].to_numpy()
    u_scores = df_unc["aggregate_uncertainty"].to_numpy()

    df_filtered, chem_jobs = select_candidates_for_chemistry(
        df_variants=df_variants,
        pred_activity=pred_act,
        uncertainty_scores=u_scores,
        feature_matrix=X,
        gate0_passed=gate0.get("gate0_passed", True),
        max_jobs=12,
        radius_angstrom=8.0
    )

    n_eligible = int(df_filtered["chem_eligible"].sum())
    print(f"Structural filter (R <= 8.0 Å): {n_eligible}/{len(df_variants)} variants eligible for chemistry.")
    print(f"Acquisition gate selected {len(chem_jobs)} jobs within budget constraint (Max: 12).")

    chem_dir = PROJECT_ROOT / "results" / "chemistry"
    chem_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = chem_dir / "chem_jobs.json"

    with open(jobs_path, "w") as f:
        json.dump(chem_jobs, f, indent=2)

    print(f"Saved chemistry queue to: {jobs_path}")
    print("Step 3 Complete!\n")


if __name__ == "__main__":
    main()
