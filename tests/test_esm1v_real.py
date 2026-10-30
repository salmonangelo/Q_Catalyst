"""Integration tests for real ESM-1v 650M pretrained model inference.

These tests execute against the locally cached pretrained checkpoint
(facebook/esm1v_t33_650M_UR90S_1) in evaluation mode with gradients disabled.
Routine unit tests in tests/test_protein_ai.py remain 100% offline using mock=True.
"""

from pathlib import Path
import numpy as np
import pytest

from qcatalyst.protein_ai.esm_scoring import (
    ESMScorer,
    ESMScorerConfig,
    ESMScoreResult,
)


@pytest.fixture(scope="module")
def real_esm1v_scorer():
    """Module-scoped scorer loading real ESM-1v from local HF cache."""
    config = ESMScorerConfig(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        model_version="v1",
        scoring_strategy="masked_marginal",
        device="cpu",
        mock=False,
    )
    return ESMScorer(config)


def test_real_esm1v_model_loading_and_attributes(real_esm1v_scorer):
    """Verify real ESM-1v model loads in eval mode with no gradients."""
    assert real_esm1v_scorer._model is not None
    assert real_esm1v_scorer._tokenizer is not None
    assert not real_esm1v_scorer._model.training, "Model must be in strict eval mode"
    for param in real_esm1v_scorer._model.parameters():
        assert not param.requires_grad, "Gradients must be disabled on all parameters"


def test_real_esm1v_wt_and_single_mutant(real_esm1v_scorer):
    """Verify real masked-marginal scoring on WT and N37P mutant."""
    # Real IsPETase WT sequence fragment containing position 37
    # Full WT length 290 aa
    variants_path = Path("data/curated/variants.parquet")
    import pandas as pd
    df = pd.read_parquet(variants_path)
    wt_seq = df[df["parent_enzyme"] == "IsPETase"]["wildtype_sequence"].iloc[0]

    # WT score must be 0.0
    wt_res = real_esm1v_scorer.score_variant(wt_seq, "WT", parent_enzyme="IsPETase")
    assert wt_res.score == 0.0
    assert wt_res.run_status == "success"
    assert wt_res.num_mutations == 0

    # N37P score
    n37p_res1 = real_esm1v_scorer.score_variant(wt_seq, "N37P", parent_enzyme="IsPETase")
    assert np.isfinite(n37p_res1.score)
    assert n37p_res1.run_status == "success"
    assert n37p_res1.num_mutations == 1
    assert n37p_res1.mutations_canonical == "N37P"
    assert n37p_res1.feature_source == "esm1v_zero_shot_llr"

    # Determinism
    n37p_res2 = real_esm1v_scorer.score_variant(wt_seq, "N37P", parent_enzyme="IsPETase")
    assert n37p_res1.score == n37p_res2.score, "Real ESM-1v scoring must be strictly deterministic"


def test_real_esm1v_multimutant_additivity(real_esm1v_scorer):
    """Verify multi-mutant score sums individual masked marginal LLRs."""
    variants_path = Path("data/curated/variants.parquet")
    import pandas as pd
    df = pd.read_parquet(variants_path)
    wt_seq = df[df["parent_enzyme"] == "IsPETase"]["wildtype_sequence"].iloc[0]

    multi_res = real_esm1v_scorer.score_variant(wt_seq, "N37P:S290P", parent_enzyme="IsPETase")
    n37p_res = real_esm1v_scorer.score_variant(wt_seq, "N37P", parent_enzyme="IsPETase")
    s290p_res = real_esm1v_scorer.score_variant(wt_seq, "S290P", parent_enzyme="IsPETase")

    expected_sum = round(n37p_res.score + s290p_res.score, 4)
    assert multi_res.score == expected_sum
    assert multi_res.is_multi_mutant is True
    assert multi_res.num_mutations == 2

