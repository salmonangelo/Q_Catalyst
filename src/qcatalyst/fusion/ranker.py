"""
Evidence fusion and candidate re-ranking engine.
"""
from typing import Dict, List, Any
import numpy as np
import pandas as pd


def fuse_evidence_and_rank(
    df_variants: pd.DataFrame,
    df_predictions: pd.DataFrame,
    df_uncertainty: pd.DataFrame,
    df_chem: pd.DataFrame = None,
    df_quantum: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Fuses protein AI scores, uncertainty penalties, and quantum/classical chemistry
    descriptors to produce final prioritised ranking with confidence tiers.
    """
    df = df_variants.copy()

    # Join predictions
    pred_map = df_predictions.set_index("variant_id")
    df["pred_activity"] = df["variant_id"].map(pred_map["pred_activity"])
    df["pred_interval_lower"] = df["variant_id"].map(pred_map["pred_interval_lower"])
    df["pred_interval_upper"] = df["variant_id"].map(pred_map["pred_interval_upper"])
    df["pred_delta_tm"] = df["variant_id"].map(pred_map["pred_delta_tm"])
    df["zero_shot_esm_score"] = df["variant_id"].map(pred_map["zero_shot_esm_score"])

    # Join uncertainty
    unc_map = df_uncertainty.set_index("variant_id")
    df["aggregate_uncertainty"] = df["variant_id"].map(unc_map["aggregate_uncertainty"])
    df["conformal_interval_width"] = df["variant_id"].map(unc_map["conformal_interval_width"])
    df["structural_risk"] = df["variant_id"].map(unc_map["structural_risk"])

    # Baseline AI rank (descending by predicted activity)
    df["ai_rank"] = df["pred_activity"].rank(ascending=False, method="min").astype(int)

    # Chemistry support join
    chem_dict = {}
    if df_chem is not None and not df_chem.empty:
        for _, row in df_chem.iterrows():
            chem_dict[row["variant_id"]] = row.to_dict()

    # Quantum support join
    quantum_dict = {}
    if df_quantum is not None and not df_quantum.empty:
        # Take noiseless run
        if "noise_level" in df_quantum.columns:
            noiseless = df_quantum[df_quantum["noise_level"] == 0.0]
        else:
            noiseless = df_quantum
        for _, row in noiseless.iterrows():
            quantum_dict[row["variant_id"]] = row.to_dict()

    chem_supported = []
    quantum_supported = []
    ts_stabilizations = []
    d_attacks = []
    fused_scores = []
    confidence_tiers = []
    reasons = []
    limitations = []

    for _, row in df.iterrows():
        vid = row["variant_id"]
        muts = row["mutations"]
        pred_act = row["pred_activity"]
        u = row["aggregate_uncertainty"]

        has_chem = vid in chem_dict
        has_quant = vid in quantum_dict
        chem_supported.append(has_chem)
        quantum_supported.append(has_quant)

        # Baseline composite AI utility
        base_score = float(pred_act - 0.45 * u)

        if has_chem:
            c_info = chem_dict[vid]
            ts_stab = c_info["ts_stabilization_kcal_mol"]
            d_att = c_info["d_attack_angstrom"]
            ts_stabilizations.append(ts_stab)
            d_attacks.append(d_att)

            # Favorable TS stabilization (negative kcal/mol) provides mechanistic boost
            chem_modifier = -0.15 * ts_stab
            # Penalty if attack distance is disrupted
            if d_att > 3.2:
                chem_modifier -= 1.5

            fused_score = base_score + chem_modifier
        else:
            ts_stabilizations.append(np.nan)
            d_attacks.append(np.nan)
            fused_score = base_score

        fused_scores.append(float(np.round(fused_score, 3)))

        # Assign Confidence Tier and Provenance
        if muts in ["S160A", "H237A"]:
            tier = "Tier 3: Catalytic Dead / Detrimental"
            reason = "Ablation of essential catalytic triad residue completely destroys activity."
            lim = "Severe negative control; excluded from wet-lab candidate pipeline."
        elif has_chem and ts_stab < -1.0 and pred_act >= 1.5:
            tier = "Tier 1: High Confidence Mechanism-Backed"
            reason = (
                f"Favorable active-site geometry (d_attack = {d_att:.2f} Å) and "
                f"TS stabilization ({ts_stab:.1f} kcal/mol) corroborating high AI activity ({pred_act:.2f}x WT)."
            )
            lim = "Cluster QM neglects full-protein conformational entropy and macroscopic solvent relaxation."
        elif pred_act >= 1.3 and u < 0.55:
            tier = "Tier 2: AI-Promising (Awaiting Wet-Lab)"
            reason = f"Strong sequence-level fitness score ({pred_act:.2f}x WT) with calibrated moderate uncertainty."
            lim = "No active-site chemistry calculation; relies on empirical protein AI priors."
        else:
            tier = "Tier 3: Elevated Risk / Low Yield"
            reason = f"High uncertainty ({u:.2f}) or marginal predicted improvement over wild type."
            lim = "Insufficient mechanistic evidence to warrant wet-lab synthesis."

        confidence_tiers.append(tier)
        reasons.append(reason)
        limitations.append(lim)

    df["chem_supported"] = chem_supported
    df["quantum_supported"] = quantum_supported
    df["ts_stabilization_kcal_mol"] = ts_stabilizations
    df["d_attack_angstrom"] = d_attacks
    df["final_composite_score"] = fused_scores
    df["confidence_tier"] = confidence_tiers
    df["recommendation_reason"] = reasons
    df["limitations"] = limitations

    # Final prioritised rank
    df["final_rank"] = df["final_composite_score"].rank(ascending=False, method="min").astype(int)
    df["rank_delta"] = df["ai_rank"] - df["final_rank"]

    return df.sort_values(by="final_rank").reset_index(drop=True)
