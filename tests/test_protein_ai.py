"""Unit tests for ESM-1v mutation-effect scoring and protein AI module."""

import hashlib
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from qcatalyst.protein_ai.esm_scoring import (
    ESMScoreResult,
    ESMScorer,
    ESMScorerConfig,
    save_esm_predictions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CURATED_VARIANTS = REPO_ROOT / "data" / "curated" / "variants.parquet"


def test_esm_module_imports_without_downloads():
    """Verify ESM module imports cleanly without downloading model weights."""
    config = ESMScorerConfig(mock=True)
    scorer = ESMScorer(config)
    assert scorer is not None
    assert scorer.config.mock is True


def test_mock_scoring_single_mutation():
    """Verify single-site mutation scoring produces expected fields and finite score."""
    scorer = ESMScorer(ESMScorerConfig(mock=True))
    wt_seq = "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARG"  # 35 aa

    result = scorer.score_variant(
        wt_seq=wt_seq,
        mutant_str="N2P",
        parent_enzyme="IsPETase",
        variant_id="IsPETase_N2P",
    )

    assert isinstance(result, ESMScoreResult)
    assert result.variant_id == "IsPETase_N2P"
    assert result.parent_enzyme == "IsPETase"
    assert result.mutant_raw == "N2P"
    assert result.mutations_canonical == "N2P"
    assert result.num_mutations == 1
    assert not result.is_multi_mutant
    assert isinstance(result.score, float)
    assert result.scoring_method == "masked_marginal"
    assert "esm1v" in result.model_name
    assert result.run_status == "mocked"
    assert len(result.sequence_hash) == 64
    assert result.is_prediction is True
    assert "not experimentally validated activity" in result.disclaimer.lower()


def test_mock_scoring_multi_mutation():
    """Verify multi-mutation sums log-likelihood ratios over all sites."""
    scorer = ESMScorer(ESMScorerConfig(mock=True))
    wt_seq = "MNFPRASRLMQA"

    # Score individual single mutants
    res1 = scorer.score_variant(wt_seq, "M1A")
    res2 = scorer.score_variant(wt_seq, "N2G")

    # Score combination
    res_multi = scorer.score_variant(wt_seq, "N2G:M1A")

    assert res_multi.num_mutations == 2
    assert res_multi.is_multi_mutant is True
    assert res_multi.mutations_canonical == "M1A:N2G"
    # In masked-marginal sum, multi score equals sum of individual mock LLRs
    assert pytest.approx(res_multi.score, abs=1e-4) == res1.score + res2.score


def test_wildtype_scoring_returns_zero():
    """Wild-type sequence by definition has 0.0 log-likelihood difference."""
    scorer = ESMScorer(ESMScorerConfig(mock=True))
    wt_seq = "MNFPRASRLMQA"

    for wt_token in ["WT", "wildtype", "none", "", None]:
        res = scorer.score_variant(wt_seq, wt_token, parent_enzyme="IsPETase")
        assert res.score == 0.0
        assert res.num_mutations == 0
        assert not res.is_multi_mutant
        assert res.mutations_canonical == "WT"


def test_invalid_mutation_syntax_raises_error():
    """Invalid syntax should raise ValueError."""
    scorer = ESMScorer(ESMScorerConfig(mock=True))
    wt_seq = "MNFP"

    with pytest.raises(ValueError):
        scorer.score_variant(wt_seq, "INVALID")

    with pytest.raises(ValueError):
        scorer.score_variant(wt_seq, "N-5P")


def test_out_of_bounds_mutation_raises_error():
    """Mutations beyond sequence length must raise ValueError."""
    scorer = ESMScorer(ESMScorerConfig(mock=True))
    wt_seq = "MNFP"  # Length 4

    with pytest.raises(ValueError, match="out of sequence bounds"):
        scorer.score_variant(wt_seq, "P10A")


def test_residue_mismatch_raises_error():
    """Mutation claiming wrong wild-type residue must raise ValueError."""
    scorer = ESMScorer(ESMScorerConfig(mock=True))
    wt_seq = "MNFP"  # Pos 1 is M, not A

    with pytest.raises(ValueError, match="Residue mismatch"):
        scorer.score_variant(wt_seq, "A1G")


def test_empty_sequence_raises_error():
    """Empty or non-string sequence must raise ValueError."""
    scorer = ESMScorer(ESMScorerConfig(mock=True))

    with pytest.raises(ValueError, match="non-empty string"):
        scorer.score_variant("", "N1A")


def test_missing_cache_dir_raises_without_downloading():
    """Attempting real model load from non-existent cache directory must raise FileNotFoundError."""
    config = ESMScorerConfig(
        mock=False,
        cache_dir="/non/existent/path/to/esm1v",
    )
    with pytest.raises(FileNotFoundError, match="Per project rules"):
        ESMScorer(config)


def test_scoring_dataframe_from_variants_table():
    """Verify batch scoring on a real subset from data/curated/variants.parquet."""
    assert CURATED_VARIANTS.exists(), "curated variants.parquet must exist"
    df = pd.read_parquet(CURATED_VARIANTS)

    scorer = ESMScorer(ESMScorerConfig(mock=True))
    scored_df = scorer.score_variants_dataframe(df, max_rows=10)

    assert len(scored_df) == 10
    required_cols = [
        "variant_id",
        "parent_enzyme",
        "mutant_raw",
        "mutations_canonical",
        "num_mutations",
        "is_multi_mutant",
        "score",
        "scoring_method",
        "model_name",
        "model_version",
        "sequence_hash",
        "run_status",
        "is_prediction",
        "disclaimer",
    ]
    for col in required_cols:
        assert col in scored_df.columns, f"Missing output prediction column: {col}"

    assert (scored_df["is_prediction"] == True).all()
    assert (scored_df["scoring_method"] == "masked_marginal").all()
    assert (scored_df["run_status"] == "mocked").all()
    assert scored_df["score"].notnull().all()


def test_save_and_reload_predictions():
    """Verify predictions table saves to Parquet and reloads identically."""
    scorer = ESMScorer(ESMScorerConfig(mock=True))
    res = scorer.score_variant("MNFPRASRLMQA", "M1A", variant_id="test_v1")
    df = pd.DataFrame([asdict(res)])

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "test_esm_scores.parquet"
        reloaded = save_esm_predictions(df, out_path)

        assert len(reloaded) == 1
        assert reloaded.iloc[0]["variant_id"] == "test_v1"
        assert reloaded.iloc[0]["score"] == res.score
