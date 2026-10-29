"""Fail-safe bridge from Gate 0 validation to candidate eligibility metadata.

This module does not claim that a candidate is experimentally validated. It turns
the pre-registered Gate 0 result into an explicit downstream policy decision.
When the Gate 0 artifact is missing, malformed, or inconsistent, it fails closed
to diversity-based selection rather than silently enabling uncertainty filtering.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class GatePolicyDecision:
    gate0_verdict: str
    gate_passed: bool
    acquisition_mode: str
    uncertainty_gating_enabled: bool
    ci_lower: float | None
    ci_upper: float | None
    policy_valid: bool
    policy_errors: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy_errors"] = list(self.policy_errors)
        return payload


def load_acquisition_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the versioned project acquisition policy from configs/acquisition.yaml."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[3] / "configs" / "acquisition.yaml"
    config_path = Path(config_path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot load acquisition policy at {config_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported or malformed acquisition policy at {config_path}")
    for required_section in ("gate0", "candidate_gate", "diversity_fallback", "scientific_limits"):
        if not isinstance(payload.get(required_section), dict):
            raise ValueError(f"Acquisition policy section {required_section!r} is missing or malformed")
    return payload


def resolve_gate0_policy(
    source: str | Path | Mapping[str, Any],
    config_path: str | Path | None = None,
) -> GatePolicyDecision:
    """Read a Gate 0 artifact and apply the versioned policy; invalid data fails closed."""
    errors: list[str] = []
    try:
        policy = load_acquisition_config(config_path)
    except Exception as exc:
        policy = {}
        errors.append(f"acquisition_policy_unreadable:{type(exc).__name__}")
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        path = Path(source)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload = {}
            errors.append(f"gate_artifact_unreadable:{type(exc).__name__}")

    verdict = str(payload.get("verdict", "UNKNOWN")).upper()
    action = str(payload.get("recommended_action", "UNKNOWN")).upper()
    gate_passed_raw = payload.get("gate_passed", False)
    gate_passed = gate_passed_raw is True
    policy_gate0 = policy.get("gate0", {})
    pass_threshold = float(policy_gate0.get("pass_if_ci_lower_strictly_greater_than", 0.0))
    minimum_bootstrap = int(policy_gate0.get("min_bootstrap_resamples", 1000))
    required_ci_level = float(policy_gate0.get("confidence_level", 0.95))
    pass_action = str(policy_gate0.get("on_pass_action", "PROCEED_WITH_UNCERTAINTY_GATING")).upper()
    fail_action = str(policy_gate0.get("on_fail_action", "FALLBACK_TO_DIVERSITY_SELECTION")).upper()

    try:
        ci_lower = float(payload["ci_lower"])
        ci_upper = float(payload["ci_upper"])
        ci_level = float(payload["ci_level"])
        n_bootstrap = int(payload["n_bootstrap"])
        if not np.isfinite(ci_lower) or not np.isfinite(ci_upper) or ci_lower > ci_upper:
            raise ValueError("invalid finite interval")
    except Exception:
        ci_lower = ci_upper = None
        errors.append("gate_confidence_interval_or_method_metadata_missing_or_invalid")
        ci_level = 0.0
        n_bootstrap = 0

    if verdict not in {"PASS", "FAIL"}:
        errors.append("gate_verdict_missing_or_invalid")
    if gate_passed != (verdict == "PASS"):
        errors.append("gate_boolean_and_verdict_disagree")
    if ci_lower is not None and ci_upper is not None:
        if verdict == "PASS" and ci_lower <= pass_threshold:
            errors.append("PASS_requires_strictly_positive_ci_lower_bound")
        if verdict == "FAIL" and ci_lower > pass_threshold:
            errors.append("FAIL_inconsistent_with_positive_ci_lower_bound")
    if ci_level + 1e-12 < required_ci_level:
        errors.append("gate_confidence_level_below_policy_minimum")
    if n_bootstrap < minimum_bootstrap:
        errors.append("gate_bootstrap_count_below_policy_minimum")
    expected_action = pass_action if verdict == "PASS" else fail_action
    if action != expected_action:
        errors.append("gate_recommended_action_inconsistent_with_policy")

    policy_valid = not errors
    uncertainty_enabled = (
        policy_valid and gate_passed and ci_lower is not None and ci_lower > pass_threshold
    )
    mode = "uncertainty_guided_allowed" if uncertainty_enabled else "diversity_fallback"
    if uncertainty_enabled:
        rationale = (
            "Gate 0 artifact is internally consistent and its confidence-interval lower bound is "
            "strictly positive; downstream code may use uncertainty according to the frozen policy."
        )
    elif errors:
        rationale = (
            "Gate 0 artifact is missing, malformed, or internally inconsistent. Fail closed: "
            "do not use uncertainty as a gating signal; use diversity-based selection."
        )
    elif ci_upper is not None and ci_upper < 0.0:
        rationale = (
            "Gate 0 shows an inverse uncertainty/error association. Do not filter candidates using "
            "uncertainty; use diversity-based selection."
        )
    else:
        rationale = (
            "Gate 0 did not establish a strictly positive uncertainty/error association. Do not use "
            "uncertainty as a gating signal; use diversity-based selection."
        )

    return GatePolicyDecision(
        gate0_verdict=verdict if policy_valid else "FAIL_CLOSED",
        gate_passed=uncertainty_enabled,
        acquisition_mode=mode,
        uncertainty_gating_enabled=uncertainty_enabled,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        policy_valid=policy_valid,
        policy_errors=tuple(errors),
        rationale=rationale,
    )


def annotate_candidate_gate(
    candidates: pd.DataFrame,
    decision: GatePolicyDecision,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Annotate candidate rows with global Gate 0 policy and explicit eligibility reasons.

    Eligibility here is a data/pipeline check only. Spatial catalytic-site gating and
    chemistry-resource allocation remain downstream chemistry/integration decisions.
    """
    policy = load_acquisition_config(config_path)
    rules = policy["candidate_gate"]
    if "variant_id" not in candidates.columns:
        raise ValueError("Candidate table requires variant_id.")
    if candidates["variant_id"].isna().any():
        raise ValueError("Candidate table contains missing variant_id values.")
    if candidates["variant_id"].duplicated().any():
        raise ValueError("Candidate table contains duplicate variant_id rows.")

    out = candidates.copy()
    reasons: list[str] = []
    eligible: list[bool] = []
    chemistry_eligible: list[bool] = []
    for _, row in out.iterrows():
        row_reasons: list[str] = []
        raw_multi = row.get("is_multi_mutant", None)
        if raw_multi is None or pd.isna(raw_multi):
            raw_count = row.get("num_mutations", 0)
            n_mutations = int(raw_count) if pd.notna(raw_count) else 0
            is_multi = n_mutations > 1
        else:
            is_multi = bool(raw_multi)
        raw_duplicate = row.get("is_duplicate", False)
        is_duplicate = bool(raw_duplicate) if pd.notna(raw_duplicate) else False
        sequence = row.get("mutant_sequence")
        raw_num_mutations = row.get("num_mutations", 0)
        try:
            num_mutations = int(raw_num_mutations) if pd.notna(raw_num_mutations) else 0
        except (TypeError, ValueError):
            num_mutations = 0
        minimum_mutations = int(rules.get("minimum_mutations", 2))
        if bool(rules.get("require_single_to_multi_heldout_split", True)) and (
            not is_multi or num_mutations < minimum_mutations
        ):
            row_reasons.append("not_multi_mutant")
        if bool(rules.get("exclude_duplicate_records", True)) and is_duplicate:
            row_reasons.append("duplicate_sequence_record")
        if bool(rules.get("require_nonempty_mutant_sequence", True)) and (
            not isinstance(sequence, str) or not sequence.strip()
        ):
            row_reasons.append("missing_mutant_sequence")
        pred_value = row.get("y_pred_xgb", row.get("y_pred_mean", np.nan))
        if bool(rules.get("require_finite_supervised_activity_prediction", True)):
            try:
                if not np.isfinite(float(pred_value)):
                    row_reasons.append("missing_or_nonfinite_activity_prediction")
            except (TypeError, ValueError):
                row_reasons.append("missing_or_nonfinite_activity_prediction")
        if bool(rules.get("require_finite_real_esm1v_score", True)):
            try:
                if not np.isfinite(float(row.get("esm1v_delta_llr", np.nan))):
                    row_reasons.append("missing_or_nonfinite_real_esm1v_score")
            except (TypeError, ValueError):
                row_reasons.append("missing_or_nonfinite_real_esm1v_score")
        if bool(rules.get("require_finite_composite_uncertainty", True)):
            try:
                if not np.isfinite(float(row.get("aggregate_uncertainty", np.nan))):
                    row_reasons.append("missing_or_nonfinite_composite_uncertainty")
            except (TypeError, ValueError):
                row_reasons.append("missing_or_nonfinite_composite_uncertainty")
        eligible.append(not row_reasons)
        reasons.append("eligible" if not row_reasons else ";".join(row_reasons))

        map_status = str(row.get("structure_mapping_status", "")).lower()
        mapped = map_status in {"exact_1to1", "mapped", "available"}
        chemistry_eligible.append(
            not row_reasons and (
                mapped if bool(rules.get("require_structure_mapping_for_chemistry", True)) else True
            )
        )

    out["role1_candidate_eligible"] = eligible
    out["role1_candidate_gate_reason"] = reasons
    out["global_gate0_verdict"] = decision.gate0_verdict
    out["acquisition_mode"] = decision.acquisition_mode
    out["uncertainty_gating_enabled"] = decision.uncertainty_gating_enabled
    out["uncertainty_allowed_for_ranking"] = decision.uncertainty_gating_enabled
    out["chemistry_mapping_eligible"] = chemistry_eligible
    out["role1_gate_policy_valid"] = decision.policy_valid
    out["role1_gate_rationale"] = decision.rationale

    # This is an integrated Role 1 handoff status, not a final chemistry/quantum PASS.
    integrated_status: list[str] = []
    for role1_ok, mapping_ok in zip(eligible, chemistry_eligible):
        if not role1_ok:
            integrated_status.append("REJECT_ROLE1_INPUT_RULES")
        elif not mapping_ok:
            integrated_status.append("BLOCK_CHEMISTRY_PENDING_STRUCTURE_MAPPING")
        elif decision.uncertainty_gating_enabled:
            integrated_status.append("ELIGIBLE_UNCERTAINTY_MODE_PENDING_ROLE2_3D_GATE")
        else:
            integrated_status.append("ELIGIBLE_DIVERSITY_FALLBACK_PENDING_ROLE2_3D_GATE")
    out["combined_gate_status"] = integrated_status
    out["role2_3d_geometry_gate_status"] = "NOT_RUN_PENDING_ROLE2"
    out["requires_role2_3d_geometry_check"] = True
    fallback = policy["diversity_fallback"]
    out["fallback_weight_sequence_diversity"] = float(fallback["sequence_embedding_diversity_weight"])
    out["fallback_weight_predicted_activity"] = float(fallback["predicted_activity_weight"])
    out["fallback_weight_compute_cost"] = float(fallback["estimated_compute_cost_weight"])
    out["fallback_weight_uncertainty"] = float(fallback["uncertainty_weight"])
    return out
