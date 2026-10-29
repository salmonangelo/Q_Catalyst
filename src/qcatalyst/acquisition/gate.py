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


def resolve_gate0_policy(source: str | Path | Mapping[str, Any]) -> GatePolicyDecision:
    """Read and validate a Gate 0 artifact; uncertainty gating is enabled only on a valid PASS."""
    errors: list[str] = []
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
    try:
        ci_lower = float(payload["ci_lower"])
        ci_upper = float(payload["ci_upper"])
        if not np.isfinite(ci_lower) or not np.isfinite(ci_upper) or ci_lower > ci_upper:
            raise ValueError("invalid finite interval")
    except Exception:
        ci_lower = ci_upper = None
        errors.append("gate_confidence_interval_missing_or_invalid")

    if verdict not in {"PASS", "FAIL"}:
        errors.append("gate_verdict_missing_or_invalid")
    if gate_passed != (verdict == "PASS"):
        errors.append("gate_boolean_and_verdict_disagree")
    if ci_lower is not None and ci_upper is not None:
        if verdict == "PASS" and ci_lower <= 0.0:
            errors.append("PASS_requires_strictly_positive_ci_lower_bound")
        if verdict == "FAIL" and ci_lower > 0.0:
            errors.append("FAIL_inconsistent_with_positive_ci_lower_bound")
    expected_action = (
        "PROCEED_WITH_UNCERTAINTY_GATING" if verdict == "PASS"
        else "FALLBACK_TO_DIVERSITY_SELECTION"
    )
    if action != expected_action:
        errors.append("gate_recommended_action_inconsistent_with_verdict")

    policy_valid = not errors
    uncertainty_enabled = policy_valid and gate_passed and ci_lower is not None and ci_lower > 0.0
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


def annotate_candidate_gate(candidates: pd.DataFrame, decision: GatePolicyDecision) -> pd.DataFrame:
    """Annotate candidate rows with global Gate 0 policy and explicit eligibility reasons.

    Eligibility here is a data/pipeline check only. Spatial catalytic-site gating and
    chemistry-resource allocation remain downstream chemistry/integration decisions.
    """
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
        if not is_multi:
            row_reasons.append("not_multi_mutant")
        if is_duplicate:
            row_reasons.append("duplicate_sequence_record")
        if not isinstance(sequence, str) or not sequence.strip():
            row_reasons.append("missing_mutant_sequence")
        pred_value = row.get("y_pred_xgb", row.get("y_pred_mean", np.nan))
        try:
            if not np.isfinite(float(pred_value)):
                row_reasons.append("missing_or_nonfinite_activity_prediction")
        except (TypeError, ValueError):
            row_reasons.append("missing_or_nonfinite_activity_prediction")
        eligible.append(not row_reasons)
        reasons.append("eligible" if not row_reasons else ";".join(row_reasons))

        map_status = str(row.get("structure_mapping_status", "")).lower()
        chemistry_eligible.append(
            not row_reasons and map_status in {"exact_1to1", "mapped", "available"}
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
    return out
