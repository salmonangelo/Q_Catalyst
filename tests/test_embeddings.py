"""Unit tests for ESM-2 sequence embedding extraction and caching."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qcatalyst.protein_ai.embeddings import (
    ESM2Embedder,
    ESM2EmbedderConfig,
    compute_sequence_hash,
    save_embedding_index,
    validate_protein_sequence,
)


def test_sequence_hash_deterministic():
    seq1 = "MNFPRASRLMQA"
    seq2 = "mnfprasrlmqa "  # lowercase with whitespace
    seq3 = "MNFPRASRLMQC"  # 1 aa diff

    hash1 = compute_sequence_hash(seq1)
    hash2 = compute_sequence_hash(seq2)
    hash3 = compute_sequence_hash(seq3)

    assert hash1 == hash2, "Whitespace and case differences must yield identical hash"
    assert hash1 != hash3, "Different sequences must produce different hashes"
    assert len(hash1) == 64


def test_validate_protein_sequence():
    # Valid
    validate_protein_sequence("MNFPRASRLMQA")

    # Invalid characters
    with pytest.raises(ValueError, match="invalid non-amino-acid"):
        validate_protein_sequence("MNFP123")

    with pytest.raises(ValueError, match="invalid non-amino-acid"):
        validate_protein_sequence("MNFP-X")

    # Empty
    with pytest.raises(ValueError, match="non-empty string"):
        validate_protein_sequence("")


def test_embedding_dimensions_and_finite():
    """Verify extracted embeddings have expected dimensions and strictly finite numbers."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test 35M default (480-d)
        config_35m = ESM2EmbedderConfig(
            model_id="facebook/esm2_t12_35M_UR50D",
            cache_dir=tmp_dir,
            mock=True,
        )
        embedder_35m = ESM2Embedder(config_35m)
        seq = "MNFPRASRLMQA"
        vec_35m, entry_35m = embedder_35m.extract_embedding(seq, variant_id="v1")

        assert vec_35m.shape == (480,)
        assert entry_35m.dimensions == 480
        assert np.all(np.isfinite(vec_35m)), "Embedding must contain strictly finite values"
        assert entry_35m.is_finite is True
        assert entry_35m.status == "mocked"

        # Test 8M model (320-d)
        config_8m = ESM2EmbedderConfig(
            model_id="facebook/esm2_t6_8M_UR50D",
            cache_dir=tmp_dir,
            mock=True,
        )
        embedder_8m = ESM2Embedder(config_8m)
        vec_8m, entry_8m = embedder_8m.extract_embedding(seq, variant_id="v2")

        assert vec_8m.shape == (320,)
        assert entry_8m.dimensions == 320
        assert np.all(np.isfinite(vec_8m))


def test_pooling_strategies_produce_distinct_representations():
    """Verify mean vs bos pooling yield different valid representation vectors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        seq = "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARG"

        cfg_mean = ESM2EmbedderConfig(pooling_strategy="mean", cache_dir=tmp_dir, mock=True)
        vec_mean, _ = ESM2Embedder(cfg_mean).extract_embedding(seq)

        cfg_bos = ESM2EmbedderConfig(pooling_strategy="bos", cache_dir=tmp_dir, mock=True)
        vec_bos, _ = ESM2Embedder(cfg_bos).extract_embedding(seq)

        assert vec_mean.shape == vec_bos.shape
        # Vectors must not be identical across different pooling mechanisms
        assert not np.allclose(vec_mean, vec_bos)


def test_cache_reuse():
    """Verify extracting the same sequence twice reuses the cached .npz file without recomputation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = ESM2EmbedderConfig(cache_dir=tmp_dir, mock=True)
        embedder = ESM2Embedder(config)
        seq = "MNFPRASRLMQA"

        # First extraction: computes and writes to cache
        vec1, entry1 = embedder.extract_embedding(seq, variant_id="v_first")
        assert entry1.status == "mocked"
        cache_file = Path(entry1.cache_path)
        assert cache_file.exists()
        mod_time1 = cache_file.stat().st_mtime

        # Second extraction: reads from cache
        vec2, entry2 = embedder.extract_embedding(seq, variant_id="v_second")
        assert entry2.status == "cached"
        assert np.allclose(vec1, vec2)
        # Verify file was read, not rewritten
        assert cache_file.stat().st_mtime == mod_time1


def test_batch_extraction_and_index():
    """Verify batch extraction builds correct matrix shape and valid Parquet index table."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = ESM2EmbedderConfig(cache_dir=tmp_dir, mock=True)
        embedder = ESM2Embedder(config)

        df = pd.DataFrame([
            {"variant_id": "v1", "mutant_sequence": "MNFPRASRLMQA"},
            {"variant_id": "v2", "mutant_sequence": "MNFPRASRLMQC"},
            {"variant_id": "v3", "mutant_sequence": "MNFPRASRLMQD"},
        ])

        matrix, index_df = embedder.extract_batch(df)

        assert matrix.shape == (3, 480)
        assert len(index_df) == 3
        assert list(index_df["variant_id"]) == ["v1", "v2", "v3"]
        assert (index_df["dimensions"] == 480).all()
        assert (index_df["is_finite"] == True).all()

        # Save index and verify roundtrip
        idx_path = Path(tmp_dir) / "embedding_index.parquet"
        reloaded = save_embedding_index(index_df, idx_path)
        assert len(reloaded) == 3
        assert (reloaded["dimensions"] == 480).all()
