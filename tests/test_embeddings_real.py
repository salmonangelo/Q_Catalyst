"""Integration tests for real ESM-2 35M pretrained model inference.

These tests execute against the locally cached pretrained checkpoint
(facebook/esm2_t12_35M_UR50D) in evaluation mode with gradients disabled.
Routine unit tests in tests/test_embeddings.py remain 100% offline using mock=True.
"""

from pathlib import Path
import numpy as np
import pytest

from qcatalyst.protein_ai.embeddings import (
    ESM2Embedder,
    ESM2EmbedderConfig,
)


@pytest.fixture(scope="module")
def real_embedder():
    """Module-scoped embedder loading real ESM-2 35M from local HF cache."""
    config = ESM2EmbedderConfig(
        model_id="facebook/esm2_t12_35M_UR50D",
        model_version="v1",
        pooling_strategy="mean",
        cache_dir=Path("results/embeddings"),
        device="cpu",
        mock=False,
    )
    return ESM2Embedder(config)


def test_real_esm2_model_loading_and_attributes(real_embedder):
    """Verify real ESM-2 35M model loads in eval mode with no gradients."""
    assert real_embedder._model is not None
    assert real_embedder._tokenizer is not None
    assert not real_embedder._model.training, "Model must be in strict eval mode"
    for param in real_embedder._model.parameters():
        assert not param.requires_grad, "Gradients must be disabled on all parameters"


def test_real_esm2_wt_and_mutant_inference(real_embedder):
    """Verify real 480-dimensional embeddings on IsPETase WT and N37P mutant."""
    wt_seq = "MNFPRASRLMQAAVLGGLMAVSAAATAQTN"  # Valid test fragment
    mut_seq = "MNFPRASRLMQAAVLGGLMAVSAAATAQTP"  # 1 aa substitution at end

    # WT extraction
    wt_vec, wt_entry = real_embedder.extract_embedding(wt_seq, variant_id="test_wt")
    assert wt_vec.shape == (480,)
    assert wt_vec.dtype == np.float32
    assert np.all(np.isfinite(wt_vec)), "Embeddings must contain strictly finite values"
    assert wt_entry.dimensions == 480
    assert wt_entry.feature_source == "real_esm2_35M"
    assert wt_entry.device == "cpu"
    assert wt_entry.status in ("computed", "cached")
    assert "HuggingFace Hub" in wt_entry.checkpoint_provenance

    # Mutant extraction
    mut_vec, mut_entry = real_embedder.extract_embedding(mut_seq, variant_id="test_mut")
    assert mut_vec.shape == (480,)
    assert np.all(np.isfinite(mut_vec))
    assert mut_entry.feature_source == "real_esm2_35M"

    # Vectors must be subtly distinct (learned representation shift)
    assert not np.allclose(wt_vec, mut_vec)
    cos_sim = float(np.dot(wt_vec, mut_vec) / (np.linalg.norm(wt_vec) * np.linalg.norm(mut_vec)))
    assert 0.95 < cos_sim < 1.0, f"Expected high cosine similarity for 1-aa mutation, got {cos_sim}"


def test_real_esm2_cache_reuse(real_embedder):
    """Verify second extraction reuses cached .npz with identical values."""
    seq = "MNFPRASRLMQAAVLGGLMAVSAAATAQTN"
    vec1, entry1 = real_embedder.extract_embedding(seq, variant_id="cache_test_1")
    vec2, entry2 = real_embedder.extract_embedding(seq, variant_id="cache_test_2")

    assert entry2.status == "cached"
    assert np.array_equal(vec1, vec2)
    assert entry2.feature_source == "real_esm2_35M"


def test_real_esm2_rejects_invalid_sequence(real_embedder):
    """Verify real embedder rejects invalid characters without mock fallback."""
    with pytest.raises(ValueError, match="invalid non-amino-acid"):
        real_embedder.extract_embedding("MNFP123INVALID")

