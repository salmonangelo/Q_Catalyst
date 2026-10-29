"""Unit tests for the fail-safe Role 1 Gate 0 policy bridge."""
import json

import numpy as np
import pandas as pd

from qcatalyst.acquisition.gate import annotate_candidate_gate, resolve_gate0_policy


def _payload(verdict="PASS", lower=0.12, upper=0.55):
    passed = verdict == "PASS"
    return {
        "verdict": verdict,
        "gate_passed": passed,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": 0.95,
        "n_bootstrap": 2000,
        "recommended_action": (
            "PROCEED_WITH_UNCERTAINTY_GATING" if passed
            else "FALLBACK_TO_DIVERSITY_SELECTION"
        ),
    }


def test_gate0_valid_positive_ci_enables_uncertainty_policy():
    result = resolve_gate0_policy(_payload())
    assert result.policy_valid is True
    assert result.uncertainty_gating_enabled is True
    assert result.acquisition_mode == "uncertainty_guided_allowed"


def test_gate0_negative_ci_fails_to_diversity_even_when_it_does_not_span_zero():
    result = resolve_gate0_policy(_payload("FAIL", lower=-0.60, upper=-0.13))
    assert result.policy_valid is True
    assert result.ci_lower == -0.60
    assert result.ci_upper == -0.13
    assert result.uncertainty_gating_enabled is False
    assert result.acquisition_mode == "diversity_fallback"
    assert "inverse association" in result.rationale


def test_gate0_inconsistent_artifact_fails_closed():
    payload = _payload("PASS", lower=-0.20, upper=0.30)
    result = resolve_gate0_policy(payload)
    assert result.policy_valid is False
    assert result.gate0_verdict == "FAIL_CLOSED"
    assert result.uncertainty_gating_enabled is False
    assert result.acquisition_mode == "diversity_fallback"
    assert result.policy_errors


def test_gate0_missing_file_fails_closed(tmp_path):
    result = resolve_gate0_policy(tmp_path / "missing.json")
    assert result.policy_valid is False
    assert result.acquisition_mode == "diversity_fallback"


def test_candidate_gate_marks_invalid_rows_and_preserves_policy():
    candidates = pd.DataFrame([
        {
            "variant_id": "good_multi",
            "is_multi_mutant": True,
            "num_mutations": 2,
            "esm1v_delta_llr": -1.2,
            "aggregate_uncertainty": 0.4,
            "is_duplicate": False,
            "mutant_sequence": "ACDE",
            "y_pred_xgb": 1.2,
            "structure_mapping_status": "exact_1to1",
        },
        {
            "variant_id": "single",
            "is_multi_mutant": False,
            "num_mutations": 1,
            "esm1v_delta_llr": -0.4,
            "aggregate_uncertainty": 0.2,
            "is_duplicate": False,
            "mutant_sequence": "ACDF",
            "y_pred_xgb": 1.0,
            "structure_mapping_status": "exact_1to1",
        },
        {
            "variant_id": "missing_prediction",
            "is_multi_mutant": True,
            "num_mutations": 2,
            "esm1v_delta_llr": -0.9,
            "aggregate_uncertainty": 0.6,
            "is_duplicate": False,
            "mutant_sequence": "ACDG",
            "y_pred_xgb": np.nan,
            "structure_mapping_status": "unmapped",
        },
    ])
    decision = resolve_gate0_policy(_payload("FAIL", lower=-0.6, upper=-0.1))
    output = annotate_candidate_gate(candidates, decision)
    assert output["role1_candidate_eligible"].tolist() == [True, False, False]
    assert output.loc[0, "chemistry_mapping_eligible"] is True or bool(output.loc[0, "chemistry_mapping_eligible"])
    assert not output["uncertainty_gating_enabled"].any()
    assert (output["acquisition_mode"] == "diversity_fallback").all()
    assert output.loc[0, "combined_gate_status"] == "ELIGIBLE_DIVERSITY_FALLBACK_PENDING_ROLE2_3D_GATE"
    assert output.loc[1, "combined_gate_status"] == "REJECT_ROLE1_INPUT_RULES"
    assert output.loc[2, "combined_gate_status"] == "REJECT_ROLE1_INPUT_RULES"
    assert (output["role2_3d_geometry_gate_status"] == "NOT_RUN_PENDING_ROLE2").all()
    assert "not_multi_mutant" in output.loc[1, "role1_candidate_gate_reason"]
    assert "missing_or_nonfinite_activity_prediction" in output.loc[2, "role1_candidate_gate_reason"]


def test_candidate_gate_rejects_duplicate_ids():
    candidates = pd.DataFrame([
        {"variant_id": "same", "is_multi_mutant": True, "mutant_sequence": "ACDE", "y_pred_xgb": 1.0},
        {"variant_id": "same", "is_multi_mutant": True, "mutant_sequence": "ACDF", "y_pred_xgb": 1.1},
    ])
    decision = resolve_gate0_policy(_payload())
    try:
        annotate_candidate_gate(candidates, decision)
    except ValueError as exc:
        assert "duplicate variant_id" in str(exc)
    else:
        raise AssertionError("Duplicate candidate IDs must be rejected.")
