"""
Candidate acquisition and budget-gated selection module.
"""
from typing import Dict, List, Any, Optional
import json
import numpy as np
import pandas as pd
from src.qcatalyst.structure.proximity import is_eligible_for_chemistry


def compute_diversity_score(features: np.ndarray) -> np.ndarray:
    """
    Computes distance to nearest neighbor in feature space to reward mutational diversity.
    """
    n = features.shape[0]
    div_scores = np.zeros(n)
    for i in range(n):
        diffs = features - features[i]
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = np.inf
        div_scores[i] = np.min(dists)
    max_d = np.max(div_scores) if np.max(div_scores) > 0 else 1.0
    return div_scores / max_d


def select_candidates_for_chemistry(
    df_variants: pd.DataFrame,
    pred_activity: np.ndarray,
    uncertainty_scores: np.ndarray,
    feature_matrix: np.ndarray,
    gate0_passed: bool = True,
    max_jobs: int = 12,
    weights: Dict[str, float] = None,
    radius_angstrom: float = 8.0
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Applies structural proximity filter and acquisition policy:
    a(x) = lambda*U + eta*P + mu*D - nu*C
    Returns filtered, ranked candidates and chem_jobs.json structure.
    """
    if weights is None:
        weights = {"lambda": 0.4, "eta": 0.3, "mu": 0.2, "nu": 0.1}

    n = len(df_variants)
    # 1. Structural proximity filter
    eligible_mask = []
    min_dists = []
    for _, row in df_variants.iterrows():
        elig, d = is_eligible_for_chemistry(row["mutations"], radius_angstrom=radius_angstrom)
        eligible_mask.append(elig)
        min_dists.append(d)

    eligible_mask = np.array(eligible_mask, dtype=bool)
    min_dists = np.array(min_dists, dtype=float)

    # 2. Diversity scores
    div_scores = compute_diversity_score(feature_matrix)

    # 3. Estimated compute cost (cost is higher for multi-mutants requiring larger cluster relaxation)
    mutation_counts = df_variants["mutation_count"].to_numpy(dtype=float)
    comp_cost = 0.5 + 0.5 * np.clip(mutation_counts / 5.0, 0.0, 1.0) # Scaled in [0.5, 1.0]

    # 4. Standardized component terms
    P_norm = pred_activity / (np.max(pred_activity) if np.max(pred_activity) > 0 else 1.0)
    U_norm = uncertainty_scores / (np.max(uncertainty_scores) if np.max(uncertainty_scores) > 0 else 1.0)
    D_norm = div_scores
    C_norm = comp_cost

    # 5. Acquisition score computation
    if gate0_passed:
        acq_score = (
            weights["lambda"] * U_norm +
            weights["eta"] * P_norm +
            weights["mu"] * D_norm -
            weights["nu"] * C_norm
        )
        selection_mode = "UNCERTAINTY_GUIDED_ACQUISITION"
    else:
        # Pre-registered Fallback: Stratified diversity selection
        acq_score = 0.6 * D_norm + 0.4 * P_norm - 0.1 * C_norm
        selection_mode = "STRATIFIED_DIVERSITY_FALLBACK"

    # Always ensure WT is included as baseline control
    wt_idx = df_variants[df_variants["mutations"] == "WT"].index
    if len(wt_idx) > 0:
        acq_score[wt_idx[0]] = 999.0 # Force WT inclusion

    df_out = df_variants.copy()
    df_out["min_distance_triad_angstrom"] = min_dists
    df_out["chem_eligible"] = eligible_mask
    df_out["acquisition_score"] = np.round(acq_score, 4)
    df_out["selection_mode"] = selection_mode

    # Filter to eligible variants and sort descending by acquisition score
    eligible_df = df_out[df_out["chem_eligible"]].sort_values(by="acquisition_score", ascending=False)
    selected_subset = eligible_df.head(max_jobs).copy()

    # Create chem_jobs JSON records
    chem_jobs = []
    for rank, (_, row) in enumerate(selected_subset.iterrows(), start=1):
        job = {
            "job_id": f"CHEM-JOB-{rank:03d}",
            "variant_id": row["variant_id"],
            "mutations": row["mutations"],
            "min_dist_to_triad_angstrom": float(row["min_distance_triad_angstrom"]),
            "predicted_activity": float(pred_activity[row.name]),
            "uncertainty_u": float(uncertainty_scores[row.name]),
            "acquisition_score": float(row["acquisition_score"]),
            "selection_rank": rank,
            "status": "SCHEDULED",
            "cluster_tier": "Tier_S_ActiveSite"
        }
        chem_jobs.append(job)

    return df_out, chem_jobs
