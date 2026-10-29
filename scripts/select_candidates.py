"""Build Role 1 candidate-gate artifact from real, held-out pipeline outputs.

This script joins independent artifacts by stable variant_id; it never treats missing
model/uncertainty/ESM-1v values as zero. Gate 0 is a global policy switch. Candidate
eligibility here is a data-quality check, not a claim of experimental activity or a
replacement for Role 2's 3-D catalytic-site proximity gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qcatalyst.acquisition.gate import (
    annotate_candidate_gate,
    load_acquisition_config,
    resolve_gate0_policy,
)
from qcatalyst.protein_ai.embeddings import compute_sequence_hash


def _load_cached_embedding(index_row: pd.Series, index_root: Path) -> np.ndarray:
    """Load a real ESM-2 embedding from its indexed cache path, with portable fallback."""
    raw_path = index_row.get("cache_path")
    cache_path = Path(str(raw_path)) if isinstance(raw_path, str) and raw_path else Path()
    if not raw_path or not isinstance(raw_path, str) or not str(raw_path).strip():
        cache_path = Path("__missing_embedding_cache__")
    if not cache_path.exists():
        model_id = str(index_row.get("model_id", "facebook/esm2_t12_35M_UR50D"))
        pooling = str(index_row.get("pooling_strategy", "mean"))
        seq_hash = str(index_row.get("sequence_hash", ""))
        model_dir = index_root / f"real_{model_id.replace('/', '_').replace(chr(92), '_')}"
        cache_path = model_dir / f"{seq_hash}_{pooling}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Real ESM-2 embedding cache missing for variant {index_row.get('variant_id')}: {cache_path}"
        )
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            vector = np.asarray(data["embedding"], dtype=np.float32)
            source = str(data["feature_source"].item()) if "feature_source" in data else "unknown"
    except Exception as exc:
        raise ValueError(f"Cannot read ESM-2 embedding cache {cache_path}: {exc}") from exc
    if source != "real_esm2_35M":
        raise ValueError(f"Embedding cache is not genuine ESM-2 output: {cache_path} ({source})")
    if vector.shape != (480,) or not np.isfinite(vector).all():
        raise ValueError(f"Embedding vector must be finite with 480 dimensions: {cache_path}")
    return vector


def _percentile_score(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """Map a vector to stable [0, 1] percentile scores; tied values remain tied."""
    series = pd.Series(np.asarray(values, dtype=np.float64))
    ranks = series.rank(method="average", pct=True).to_numpy(dtype=np.float64)
    return ranks if higher_is_better else 1.0 - ranks


def apply_acquisition_ranking(
    candidates: pd.DataFrame,
    embedding_index: pd.DataFrame,
    train_ids: set[str],
    index_root: Path,
    policy: dict[str, Any],
    uncertainty_enabled: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute the config-defined acquisition score without fabricated missing-cost values.

    Diversity is cosine distance to the nearest single-site training embedding, converted
    to a candidate-pool percentile. Predicted activity and (when available) chemistry-cost
    estimates are also converted to percentiles. Missing cost estimates cause the configured
    cost weight to be removed and the remaining available weights to be renormalized.
    """
    out = candidates.copy()
    if not train_ids:
        raise ValueError("Acquisition ranking requires non-empty single-site training IDs.")
    if not {"variant_id", "cache_path", "sequence_hash"}.issubset(embedding_index.columns):
        raise ValueError("Embedding index lacks cache_path, sequence_hash, or variant_id.")
    idx = embedding_index.copy()
    idx["variant_id"] = idx["variant_id"].astype(str)
    idx = idx.set_index("variant_id", drop=False)
    missing_train = sorted(train_ids - set(idx.index))
    missing_candidates = sorted(set(out["variant_id"].astype(str)) - set(idx.index))
    if missing_train or missing_candidates:
        raise ValueError(
            f"Embedding index does not cover ranking pool: missing_train={missing_train[:5]}, "
            f"missing_candidates={missing_candidates[:5]}"
        )

    train_vectors = np.vstack([
        _load_cached_embedding(idx.loc[variant_id], index_root)
        for variant_id in sorted(train_ids)
    ]).astype(np.float64)
    candidate_vectors = np.vstack([
        _load_cached_embedding(idx.loc[variant_id], index_root)
        for variant_id in out["variant_id"].astype(str)
    ]).astype(np.float64)
    train_norms = np.linalg.norm(train_vectors, axis=1, keepdims=True)
    candidate_norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
    if np.any(train_norms == 0.0) or np.any(candidate_norms == 0.0):
        raise ValueError("ESM-2 embeddings must have non-zero norm to compute cosine diversity.")
    train_unit = train_vectors / train_norms
    candidate_unit = candidate_vectors / candidate_norms
    nearest_similarity = np.max(candidate_unit @ train_unit.T, axis=1)
    raw_diversity = np.clip(1.0 - nearest_similarity, 0.0, 2.0)
    diversity_score = _percentile_score(raw_diversity, higher_is_better=True)
    out["nearest_train_cosine_distance"] = raw_diversity
    out["diversity_percentile_score"] = diversity_score

    activity_col = "y_pred_mean" if "y_pred_mean" in out.columns else "y_pred_xgb"
    if activity_col not in out.columns:
        raise ValueError("Candidates lack y_pred_mean/y_pred_xgb for acquisition activity score.")
    activity_values = out[activity_col].to_numpy(dtype=np.float64)
    if not np.isfinite(activity_values).all():
        raise ValueError(f"{activity_col} contains non-finite values; cannot rank candidates.")
    activity_score = _percentile_score(activity_values, higher_is_better=True)
    out["activity_score_source"] = activity_col
    out["predicted_activity_percentile_score"] = activity_score

    if uncertainty_enabled:
        uncertainty_cfg = policy["uncertainty_guided"]
        raw_uncertainty = out["aggregate_uncertainty"].to_numpy(dtype=np.float64)
        if not np.isfinite(raw_uncertainty).all():
            raise ValueError("Gate 0 passed but aggregate_uncertainty contains missing/non-finite values.")
        uncertainty_score = _percentile_score(raw_uncertainty, higher_is_better=True)
        out["uncertainty_value_percentile_score"] = uncertainty_score
        weights = {
            "diversity": float(uncertainty_cfg["sequence_embedding_diversity_weight"]),
            "activity": float(uncertainty_cfg["predicted_activity_weight"]),
            "uncertainty": float(uncertainty_cfg["uncertainty_weight"]),
            "cost": float(uncertainty_cfg["estimated_compute_cost_weight"]),
        }
        mode = "uncertainty_guided_allowed"
    else:
        fallback_cfg = policy["diversity_fallback"]
        out["uncertainty_value_percentile_score"] = np.nan
        weights = {
            "diversity": float(fallback_cfg["sequence_embedding_diversity_weight"]),
            "activity": float(fallback_cfg["predicted_activity_weight"]),
            "uncertainty": 0.0,
            "cost": float(fallback_cfg["estimated_compute_cost_weight"]),
        }
        mode = "diversity_fallback"

    # Only use explicit computed cost fields; do not infer expensive chemistry from mutation count.
    cost_col = next((
        col for col in (
            "compute_cost_cpu_seconds", "estimated_compute_cost_seconds", "estimated_chemistry_cost"
        ) if col in out.columns
    ), None)
    cost_score: np.ndarray | None = None
    if cost_col is not None:
        cost_values = out[cost_col].to_numpy(dtype=np.float64)
        if np.isfinite(cost_values).all():
            cost_score = _percentile_score(cost_values, higher_is_better=False)
            out["compute_cost_score_source"] = cost_col
    if cost_score is None:
        # Support explicit Tier S/Tier M estimates where no runtime numeric is available.
        tier_col = next((col for col in ("compute_cost_tier", "chemistry_tier", "cluster_tier") if col in out.columns), None)
        if tier_col is not None:
            tiers = out[tier_col].astype(str).str.upper()
            mapped = tiers.map({"TIER S": 1.0, "S": 1.0, "TIER M": 0.0, "M": 0.0})
            if mapped.notna().all():
                cost_score = mapped.to_numpy(dtype=np.float64)
                out["compute_cost_score_source"] = tier_col
    used_cost_source: str | None = None
    if cost_score is None:
        out["compute_cost_score_source"] = "unavailable_not_imputed"
        out["compute_cost_percentile_score"] = np.nan
        weights["cost"] = 0.0
    else:
        out["compute_cost_percentile_score"] = cost_score
        used_cost_source = (
            str(cost_col)
            if cost_col is not None and np.isfinite(out[cost_col].to_numpy(dtype=np.float64)).all()
            else str(tier_col)
        )

    if uncertainty_enabled:
        # If any required uncertainty signal was unavailable, the code above fails closed.
        pass
    else:
        out["uncertainty_value_percentile_score"] = np.nan

    score_components = {
        "diversity": diversity_score,
        "activity": activity_score,
    }
    active_weights = {
        key: weight for key, weight in weights.items()
        if weight > 0.0 and key in score_components | ({"uncertainty": uncertainty_score} if uncertainty_enabled else {})
    }
    if cost_score is not None and weights.get("cost", 0.0) > 0.0:
        score_components["cost"] = cost_score
    if uncertainty_enabled:
        score_components["uncertainty"] = uncertainty_score

    if not active_weights:
        raise ValueError("Acquisition policy has no available positive-weight score components.")
    weight_sum = sum(active_weights.values())
    normalized_weights = {key: weight / weight_sum for key, weight in active_weights.items()}
    acquisition_score = np.zeros(len(out), dtype=np.float64)
    for component, weight in normalized_weights.items():
        acquisition_score += weight * np.asarray(score_components[component], dtype=np.float64)
    eligible_mask = out["role1_candidate_eligible"].astype(bool).to_numpy()
    acquisition_score[~eligible_mask] = np.nan
    out["acquisition_score"] = acquisition_score
    out["acquisition_rank"] = pd.Series(acquisition_score, index=out.index).rank(
        ascending=False, method="min", na_option="keep"
    ).astype("Int64")
    out = out.sort_values("acquisition_rank", ascending=True, na_position="last").reset_index(drop=True)
    summary = {
        "acquisition_mode": mode,
        "cost_estimate_source": used_cost_source or "unavailable_not_imputed",
        "configured_weights": weights,
        "effective_weights_after_missing_signal_renormalization": normalized_weights,
        "n_ranked_candidates": int(np.isfinite(acquisition_score).sum()),
        "n_ineligible_candidates": int((~eligible_mask).sum()),
        "diversity_definition": "1 - maximum cosine similarity to the single-site training embedding pool; percentile normalized",
        "activity_definition": f"percentile of {activity_col}; higher is better",
        "cost_definition": "inverse percentile of measured/runtime tier estimate when present; otherwise omitted and remaining weights renormalized",
        "uncertainty_definition": "uncertainty percentile contributes only when Gate 0 passes; disabled after Gate 0 failure",
        "science_note": "Ranking is computational prioritisation only; no wet-lab outcome is implied.",
    }
    return out, summary


def build_role1_candidate_gate(
    variants_path: Path,
    predictions_path: Path,
    uncertainty_path: Path,
    esm1v_scores_path: Path,
    embedding_index_path: Path,
    gate0_verdict_path: Path,
    split_manifest_path: Path,
    output_path: Path,
    decision_path: Path,
    enzyme: str = "IsPETase",
    target_name: str = "activity",
) -> dict[str, Any]:
    """Join v2 held-out predictions and return a machine-readable policy summary."""
    policy = load_acquisition_config()
    candidate_rules = policy["candidate_gate"]
    required_files = [
        variants_path, predictions_path, uncertainty_path, esm1v_scores_path,
        embedding_index_path, gate0_verdict_path, split_manifest_path,
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Role 1 candidate gate inputs missing: {missing}")

    variants = pd.read_parquet(variants_path)
    predictions = pd.read_parquet(predictions_path)
    uncertainty = pd.read_parquet(uncertainty_path)
    esm1v = pd.read_parquet(esm1v_scores_path)
    embedding_index = pd.read_parquet(embedding_index_path)
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))

    train_ids = set(map(str, manifest.get("train_variant_ids", [])))
    eval_ids = set(map(str, manifest.get("eval_variant_ids", [])))
    if not train_ids or not eval_ids:
        raise ValueError("Corrected split manifest must contain non-empty train/eval ID lists.")
    id_overlap = train_ids.intersection(eval_ids)
    if id_overlap:
        raise ValueError(f"Split manifest contains train/eval ID overlap: {sorted(id_overlap)[:5]}")

    variants = variants[
        (variants["parent_enzyme"].astype(str).str.lower() == enzyme.lower())
        & (~variants["is_duplicate"].fillna(False).astype(bool))
    ].drop_duplicates(subset=["variant_id"], keep="first").copy()
    variants["variant_id"] = variants["variant_id"].astype(str)

    pred = predictions[
        (predictions["parent_enzyme"].astype(str).str.lower() == enzyme.lower())
        & (predictions["target_name"].astype(str).str.lower() == target_name.lower())
        & (predictions["split_role"] == "held_out_test")
    ].copy()
    unc = uncertainty[
        (uncertainty["parent_enzyme"].astype(str).str.lower() == enzyme.lower())
        & (uncertainty["target_name"].astype(str).str.lower() == target_name.lower())
        & (uncertainty["split_role"] == "held_out_test")
    ].copy()
    scores = esm1v.copy()

    for name, frame in [("predictions", pred), ("uncertainty", unc), ("ESM-1v scores", scores)]:
        if "variant_id" not in frame.columns:
            raise ValueError(f"{name} artifact is missing variant_id.")
        frame["variant_id"] = frame["variant_id"].astype(str)
    required_embedding_columns = {
        "variant_id", "model_id", "feature_source", "dimensions", "is_finite",
        "status", "sequence_hash", "cache_path", "pooling_strategy",
    }
    missing_embedding_columns = required_embedding_columns.difference(embedding_index.columns)
    if missing_embedding_columns:
        raise ValueError(
            f"ESM-2 embedding index lacks provenance columns: {sorted(missing_embedding_columns)}"
        )
    embedding_index = embedding_index.copy()
    embedding_index["variant_id"] = embedding_index["variant_id"].astype(str)
    if embedding_index["variant_id"].duplicated().any():
        raise ValueError("ESM-2 embedding index contains duplicate variant IDs.")
    if not (
        embedding_index["feature_source"].astype(str) == "real_esm2_35M"
    ).all():
        raise ValueError("ESM-2 embedding index contains mock or unexpected embedding provenance.")
    if not (
        embedding_index["model_id"].astype(str) == str(candidate_rules["esm2_embedding_model_id"])
    ).all():
        raise ValueError("ESM-2 embedding index contains an unexpected model identifier.")
    if not (
        embedding_index["dimensions"].astype(int)
        == int(candidate_rules["esm2_embedding_dimensions"])
    ).all():
        raise ValueError("ESM-2 embedding index has dimensions other than the configured 480.")
    if not embedding_index["is_finite"].astype(bool).all():
        raise ValueError("ESM-2 embedding index includes non-finite embeddings.")
    if pred["variant_id"].duplicated().any():
        raise ValueError("Supervised prediction artifact has duplicate held-out variant IDs.")
    if unc["variant_id"].duplicated().any():
        raise ValueError("Uncertainty artifact has duplicate held-out variant IDs.")
    if scores["variant_id"].duplicated().any():
        scores = scores.drop_duplicates(subset=["variant_id"], keep="last")
    required_score_columns = {"score", "run_status", "feature_source", "model_name", "scoring_method"}
    missing_score_columns = required_score_columns.difference(scores.columns)
    if missing_score_columns:
        raise ValueError(
            f"ESM-1v score artifact lacks provenance columns: {sorted(missing_score_columns)}"
        )
    scores = scores[
        (scores["run_status"].astype(str).str.lower() == "success")
        & (scores["feature_source"].astype(str) == "esm1v_zero_shot_llr")
        & (scores["model_name"].astype(str) == str(candidate_rules["esm1v_model_id"]))
        & (scores["scoring_method"].astype(str) == str(candidate_rules["esm1v_scoring_method"]))
    ].copy()
    if scores.empty:
        raise ValueError("No successful ESM-1v scores match the configured real model/scoring method.")

    eval_id_set = set(unc["variant_id"])
    if eval_id_set != eval_ids:
        missing_ids = sorted(eval_ids - eval_id_set)
        extra_ids = sorted(eval_id_set - eval_ids)
        raise ValueError(
            "Held-out uncertainty IDs do not match frozen v2 manifest. "
            f"Missing={missing_ids[:5]}, unexpected={extra_ids[:5]}"
        )
    if eval_id_set.intersection(train_ids):
        raise ValueError("Leakage check failed: held-out artifact contains a training variant ID.")
    missing_embeddings = sorted(eval_ids - set(embedding_index["variant_id"]))
    if missing_embeddings:
        raise ValueError(f"Held-out variants missing real ESM-2 embeddings: {missing_embeddings[:5]}")

    candidate = variants[variants["variant_id"].isin(eval_ids)].copy()
    if set(candidate["variant_id"]) != eval_ids:
        missing_ids = sorted(eval_ids - set(candidate["variant_id"]))
        raise ValueError(f"Curated candidate rows missing split-manifest IDs: {missing_ids[:5]}")
    candidate = candidate.merge(
        pred[["variant_id", "y_true", "y_pred_xgb", "y_pred_ridge", "prediction_status"]],
        on="variant_id", how="left", validate="one_to_one", suffixes=("", "_pred"),
    )
    candidate = candidate.merge(
        unc.drop(columns=[col for col in ["y_true"] if col in unc.columns]),
        on="variant_id", how="left", validate="one_to_one",
    )
    candidate = candidate.merge(
        scores[["variant_id", "score", "run_status"]].rename(
            columns={"score": "esm1v_delta_llr", "run_status": "esm1v_run_status"}
        ),
        on="variant_id", how="left", validate="one_to_one",
    )
    embedding_cols = [
        "variant_id", "sequence_hash", "dimensions", "feature_source", "status", "is_finite"
    ]
    candidate = candidate.merge(
        embedding_index[embedding_cols].rename(columns={
            "sequence_hash": "esm2_sequence_hash",
            "dimensions": "esm2_embedding_dimensions",
            "feature_source": "esm2_feature_source",
            "status": "esm2_embedding_status",
            "is_finite": "esm2_embedding_finite",
        }),
        on="variant_id", how="left", validate="one_to_one",
    )

    decision = resolve_gate0_policy(gate0_verdict_path)
    if candidate["y_pred_xgb"].isna().any():
        raise ValueError("At least one held-out candidate has no supervised XGBoost prediction.")
    if decision.uncertainty_gating_enabled and candidate["aggregate_uncertainty"].isna().any():
        raise ValueError(
            "Gate 0 permits uncertainty gating but at least one candidate lacks aggregate uncertainty."
        )
    if candidate["esm1v_delta_llr"].isna().any():
        bad = candidate.loc[candidate["esm1v_delta_llr"].isna(), "variant_id"].astype(str).tolist()
        raise ValueError(f"At least one held-out candidate lacks a real ESM-1v score: {bad[:5]}")
    if candidate["mutant_sequence"].isna().any():
        raise ValueError("At least one candidate lacks mutant_sequence.")
    candidate["calculated_mutant_sequence_hash"] = candidate["mutant_sequence"].map(compute_sequence_hash)
    sequence_hash_mismatch = candidate[
        candidate["calculated_mutant_sequence_hash"] != candidate["esm2_sequence_hash"]
    ]
    if not sequence_hash_mismatch.empty:
        bad = sequence_hash_mismatch["variant_id"].astype(str).tolist()
        raise ValueError(
            "ESM-2 embedding sequence hash does not match the curated mutant sequence for IDs: "
            f"{bad[:5]}"
        )
    candidate["esm2_sequence_match"] = True

    output = annotate_candidate_gate(candidate, decision)
    output, acquisition_summary = apply_acquisition_ranking(
        candidates=output,
        embedding_index=embedding_index,
        train_ids=train_ids,
        index_root=embedding_index_path.parent,
        policy=policy,
        uncertainty_enabled=decision.uncertainty_gating_enabled,
    )
    output["pipeline_split_version"] = str(manifest.get("version", "unknown"))
    output["candidate_pool_role"] = "held_out_multi_mutant"
    output["esm1v_model_id"] = str(candidate_rules["esm1v_model_id"])
    output["esm2_embedding_model_id"] = str(candidate_rules["esm2_embedding_model_id"])
    output["structural_risk_basis"] = "sequence_numbering_heuristic_not_3d"
    output["role2_3d_geometry_check_required"] = True
    output["target_name"] = target_name
    output["parent_enzyme"] = enzyme

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    summary: dict[str, Any] = {
        "pipeline_component": "role1_candidate_gate",
        "parent_enzyme": enzyme,
        "target_name": target_name,
        "split_version": manifest.get("version"),
        "n_train": len(train_ids),
        "n_held_out_candidates": len(output),
        "train_eval_id_overlap": len(train_ids.intersection(set(output["variant_id"]))),
        "n_eligible_for_acquisition": int(output["role1_candidate_eligible"].sum()),
        "n_ineligible_for_acquisition": int((~output["role1_candidate_eligible"]).sum()),
        "n_chemistry_mapping_eligible": int(output["chemistry_mapping_eligible"].sum()),
        "role2_3d_geometry_check_required": True,
        "acquisition_policy_version": policy.get("schema_version"),
        "diversity_fallback_policy": policy.get("diversity_fallback", {}),
        "acquisition_ranking": acquisition_summary,
        "gate_policy": decision.to_dict(),
        "input_artifacts": {
            "variants": str(variants_path),
            "predictions": str(predictions_path),
            "uncertainty": str(uncertainty_path),
            "esm1v_scores": str(esm1v_scores_path),
            "embedding_index": str(embedding_index_path),
            "gate0_verdict": str(gate0_verdict_path),
            "split_manifest": str(split_manifest_path),
        },
        "output_artifact": str(output_path),
        "disclaimer": (
            "Computational candidate prioritisation from retrospective labels; "
            "not experimentally verified improvement in PET degradation."
        ),
    }
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Q-Catalyst Role 1 candidate-gate artifact.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/curated/variants.parquet")
    parser.add_argument("--predictions", type=Path, default=REPO_ROOT / "results/predictions/predictions.parquet")
    parser.add_argument("--uncertainty", type=Path, default=REPO_ROOT / "results/uncertainty/uncertainty.parquet")
    parser.add_argument("--esm1v-scores", type=Path, default=REPO_ROOT / "results/predictions/esm1v_scores.parquet")
    parser.add_argument("--embedding-index", type=Path, default=REPO_ROOT / "results/embeddings/embedding_index.parquet")
    parser.add_argument("--gate0-verdict", type=Path, default=REPO_ROOT / "reports/tables/gate0_verdict.json")
    parser.add_argument("--split-manifest", type=Path, default=REPO_ROOT / "data/splits/ispetase_activity_single_to_multi_v2.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/rankings/role1_candidate_gate.parquet")
    parser.add_argument("--decision-output", type=Path, default=REPO_ROOT / "results/rankings/role1_candidate_gate.json")
    parser.add_argument("--enzyme", default="IsPETase")
    parser.add_argument("--target", choices=["activity", "tm"], default="activity")
    args = parser.parse_args()

    summary = build_role1_candidate_gate(
        variants_path=args.input,
        predictions_path=args.predictions,
        uncertainty_path=args.uncertainty,
        esm1v_scores_path=args.esm1v_scores,
        embedding_index_path=args.embedding_index,
        gate0_verdict_path=args.gate0_verdict,
        split_manifest_path=args.split_manifest,
        output_path=args.output,
        decision_path=args.decision_output,
        enzyme=args.enzyme,
        target_name=args.target,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
