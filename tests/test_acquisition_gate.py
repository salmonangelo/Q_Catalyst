"""Unit tests for the fail-safe Role 1 Gate 0 policy bridge."""
import json

import numpy as np
import pandas as pd

from qcatalyst.acquisition.gate import annotate_candidate_gate, load_acquisition_config, resolve_gate0_policy
from scripts.select_candidates import apply_acquisition_ranking


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
    assert "inverse uncertainty/error association" in result.rationale
    assert "inverse" in result.rationale and "association" in result.rationale


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



def test_diversity_fallback_does_not_require_uncertainty_signal():
    """When Gate 0 fails, absent uncertainty must not exclude an otherwise valid candidate."""
    candidates = pd.DataFrame([{
        "variant_id": "multi_without_uncertainty",
        "is_multi_mutant": True,
        "num_mutations": 2,
        "is_duplicate": False,
        "mutant_sequence": "ACDE",
        "y_pred_xgb": 1.2,
        "esm1v_delta_llr": -0.5,
        "aggregate_uncertainty": np.nan,
        "structure_mapping_status": "exact_1to1",
    }])
    decision = resolve_gate0_policy(_payload("FAIL", lower=-0.6, upper=-0.1))
    output = annotate_candidate_gate(candidates, decision)
    assert bool(output.loc[0, "role1_candidate_eligible"]) is True
    assert bool(output.loc[0, "uncertainty_gating_enabled"]) is False
    assert output.loc[0, "combined_gate_status"] == "ELIGIBLE_DIVERSITY_FALLBACK_PENDING_ROLE2_3D_GATE"



def test_diversity_fallback_computes_real_embedding_acquisition_ranking(tmp_path):
    """Fallback score uses real 480-d embedding vectors and configured weights, with no fake cost."""
    import numpy as np

    train_ids = {"train_a", "train_b"}
    candidate_ids = ["candidate_novel", "candidate_similar", "candidate_middle"]
    vectors = {
        "train_a": np.array([1.0, 0.0] + [0.0] * 478, dtype=np.float32),
        "train_b": np.array([0.0, 1.0] + [0.0] * 478, dtype=np.float32),
        "candidate_novel": np.array([0.0, 0.0, 1.0] + [0.0] * 477, dtype=np.float32),
        "candidate_similar": np.array([1.0, 0.0] + [0.0] * 478, dtype=np.float32),
        "candidate_middle": np.array([0.0, 0.8, 0.6] + [0.0] * 477, dtype=np.float32),
    }
    rows = []
    for variant_id, vector in vectors.items():
        cache_path = tmp_path / f"{variant_id}.npz"
        np.savez_compressed(
            cache_path,
            embedding=vector,
            feature_source=np.asarray("real_esm2_35M"),
        )
        rows.append({
            "variant_id": variant_id,
            "model_id": "facebook/esm2_t12_35M_UR50D",
            "sequence_hash": variant_id,
            "cache_path": str(cache_path),
            "pooling_strategy": "mean",
            "dimensions": 480,
            "feature_source": "real_esm2_35M",
            "is_finite": True,
        })
    candidates = pd.DataFrame([
        {"variant_id": "candidate_novel", "y_pred_mean": 0.5, "role1_candidate_eligible": True,
         "aggregate_uncertainty": np.nan},
        {"variant_id": "candidate_similar", "y_pred_mean": 0.8, "role1_candidate_eligible": True,
         "aggregate_uncertainty": np.nan},
        {"variant_id": "candidate_middle", "y_pred_mean": 0.7, "role1_candidate_eligible": True,
         "aggregate_uncertainty": np.nan},
    ])
    output, summary = apply_acquisition_ranking(
        candidates=candidates,
        embedding_index=pd.DataFrame(rows),
        train_ids=train_ids,
        index_root=tmp_path,
        policy=load_acquisition_config(),
        uncertainty_enabled=False,
    )
    assert summary["acquisition_mode"] == "diversity_fallback"
    assert summary["n_ranked_candidates"] == 3
    assert summary["cost_estimate_source"] == "unavailable_not_imputed"
    assert summary["effective_weights_after_missing_signal_renormalization"] == {
        "diversity": 0.5 / 0.85,
        "activity": 0.35 / 0.85,
    }
    assert output["acquisition_rank"].tolist() == [1, 2, 3]
    assert output.iloc[0]["variant_id"] == "candidate_novel"
    assert output["uncertainty_value_percentile_score"].isna().all()
