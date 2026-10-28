"""ESM-2 sequence embedding extraction module for Q-Catalyst.

Provides sequence representation extraction with:
- Evaluation-mode inference with gradients disabled (torch.no_grad()).
- Mean, BOS, and per-residue pooling strategies.
- Hash-based caching (.npz) by sequence SHA256 and model ID.
- Tabular embedding index tracking dimensions, cache paths, and status.
- Calibrated deterministic offline mock embedder for safe home execution.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Mapping of known ESM-2 model identifiers to hidden layer dimensions
ESM2_MODEL_DIMS: dict[str, int] = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "facebook/esm2_t36_3B_UR50D": 2560,
    "esm2_35M": 480,
    "esm2_8M": 320,
}

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def compute_sequence_hash(sequence: str) -> str:
    """Compute normalized SHA256 hash of an amino-acid sequence."""
    clean_seq = sequence.strip().upper()
    return hashlib.sha256(clean_seq.encode("utf-8")).hexdigest()


def validate_protein_sequence(sequence: str) -> None:
    """Ensure sequence is non-empty, composed of standard amino acids, and string type."""
    if not sequence or not isinstance(sequence, str):
        raise ValueError("Protein sequence must be a non-empty string.")

    clean_seq = sequence.strip().upper()
    invalid_chars = set(clean_seq) - STANDARD_AMINO_ACIDS
    if invalid_chars:
        raise ValueError(f"Sequence contains invalid non-amino-acid characters: {invalid_chars}")


@dataclass
class ESM2EmbedderConfig:
    """Configuration for ESM-2 sequence embedding extraction."""

    model_id: str = "facebook/esm2_t12_35M_UR50D"
    model_version: str = "v1"
    pooling_strategy: str = "mean"  # 'mean', 'bos', 'cls'
    cache_dir: str | Path = "results/embeddings"
    weights_cache_dir: str | Path | None = None
    device: str = "cpu"
    mock: bool = True  # Safe home testing default
    batch_size: int = 4
    expected_dim: int | None = None

    def __post_init__(self) -> None:
        if self.expected_dim is None:
            self.expected_dim = ESM2_MODEL_DIMS.get(self.model_id, 480)
        self.cache_dir = Path(self.cache_dir)


@dataclass(frozen=True)
class EmbeddingIndexEntry:
    """Record indexing a cached sequence embedding."""

    variant_id: str
    sequence_hash: str
    model_id: str
    model_version: str
    pooling_strategy: str
    dimensions: int
    cache_path: str
    status: str  # 'computed', 'cached', 'mocked'
    is_finite: bool


class ESM2Embedder:
    """ESM-2 embedding extractor with local array caching and gradient-free inference."""

    def __init__(self, config: ESM2EmbedderConfig | None = None) -> None:
        self.config = config or ESM2EmbedderConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tokenizer = None

        if not self.config.mock:
            self._load_real_model()

    def _load_real_model(self) -> None:
        """Load pretrained ESM-2 model and tokenizer in strict evaluation mode with no gradients."""
        weights_dir = (
            Path(self.config.weights_cache_dir)
            if self.config.weights_cache_dir
            else None
        )

        try:
            from transformers import AutoModel, AutoTokenizer

            if weights_dir and weights_dir.exists():
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(weights_dir), local_files_only=True
                )
                self._model = AutoModel.from_pretrained(
                    str(weights_dir), local_files_only=True
                )
            else:
                # Load from HuggingFace cache or local path
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
                self._model = AutoModel.from_pretrained(self.config.model_id)

            self._model.to(self.config.device)
            self._model.eval()  # Strictly evaluation mode
            for param in self._model.parameters():
                param.requires_grad = False  # Explicitly disable gradients
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ESM-2 model '{self.config.model_id}': {exc}. "
                "For local testing without model downloads, initialize with mock=True."
            ) from exc

    def _get_cache_path(self, seq_hash: str) -> Path:
        """Construct safe path for cached embedding file."""
        model_safe = self.config.model_id.replace("/", "_").replace("\\", "_")
        sub_dir = self.config.cache_dir / model_safe
        sub_dir.mkdir(parents=True, exist_ok=True)
        return sub_dir / f"{seq_hash}_{self.config.pooling_strategy}.npz"

    def _mock_embedding(self, sequence: str, seq_hash: str) -> np.ndarray:
        """Generate deterministic, finite, normalized embedding vector without weights."""
        dim = self.config.expected_dim or 480
        seed = int(seq_hash[:8], 16) % (2**32)
        rng = np.random.RandomState(seed)

        # Base representation based on amino-acid composition
        composition_seed = sum(ord(c) for c in sequence)
        rng_comp = np.random.RandomState(composition_seed % (2**32))
        base_vec = rng_comp.randn(dim).astype(np.float32)

        if self.config.pooling_strategy == "mean":
            # Compositional mean vector with length modulation
            length_factor = np.sin(len(sequence) / 100.0)
            vec = base_vec + length_factor * rng.randn(dim).astype(np.float32) * 0.1
        elif self.config.pooling_strategy in ("bos", "cls"):
            # BOS representation influenced primarily by N-terminal region
            n_term_val = sum(ord(c) for c in sequence[:10])
            rng_bos = np.random.RandomState(n_term_val % (2**32))
            vec = rng_bos.randn(dim).astype(np.float32)
        else:
            vec = base_vec

        # Normalize to unit length (finite values guaranteed)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.astype(np.float32)

    def extract_embedding(
        self,
        sequence: str,
        variant_id: str = "",
        force_recompute: bool = False,
    ) -> tuple[np.ndarray, EmbeddingIndexEntry]:
        """Extract a pooled embedding vector for a single sequence with cache check.

        Args:
            sequence: Validated protein amino-acid sequence.
            variant_id: Optional identifier for index tracking.
            force_recompute: If True, bypasses cache read.

        Returns:
            (embedding_array, index_entry)

        Raises:
            ValueError: If sequence is invalid.
        """
        validate_protein_sequence(sequence)
        seq_hash = compute_sequence_hash(sequence)
        cache_path = self._get_cache_path(seq_hash)
        dim = self.config.expected_dim or 480

        # 1. Check cache lookup
        if not force_recompute and cache_path.exists():
            try:
                data = np.load(cache_path)
                cached_vec = data["embedding"].astype(np.float32)
                if cached_vec.shape == (dim,) and np.all(np.isfinite(cached_vec)):
                    entry = EmbeddingIndexEntry(
                        variant_id=variant_id,
                        sequence_hash=seq_hash,
                        model_id=self.config.model_id,
                        model_version=self.config.model_version,
                        pooling_strategy=self.config.pooling_strategy,
                        dimensions=dim,
                        cache_path=str(cache_path.as_posix()),
                        status="cached",
                        is_finite=True,
                    )
                    return cached_vec, entry
            except Exception:
                # If cache corrupted, recompute
                pass

        # 2. Compute embedding
        if self.config.mock:
            vec = self._mock_embedding(sequence, seq_hash)
            status = "mocked"
        else:
            vec = self._compute_torch_embedding(sequence)
            status = "computed"

        is_finite = bool(np.all(np.isfinite(vec)))
        if not is_finite:
            raise ValueError(f"Computed embedding contains non-finite values (NaN/Inf) for variant {variant_id}")

        # 3. Save to cache (.npz)
        np.savez_compressed(
            cache_path,
            embedding=vec,
            sequence_hash=seq_hash,
            variant_id=variant_id,
            model_id=self.config.model_id,
            pooling=self.config.pooling_strategy,
        )

        entry = EmbeddingIndexEntry(
            variant_id=variant_id,
            sequence_hash=seq_hash,
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            pooling_strategy=self.config.pooling_strategy,
            dimensions=dim,
            cache_path=str(cache_path.as_posix()),
            status=status,
            is_finite=is_finite,
        )

        return vec, entry

    def _compute_torch_embedding(self, sequence: str) -> np.ndarray:
        """Run PyTorch forward pass in evaluation mode with gradients disabled."""
        import torch

        inputs = self._tokenizer(sequence, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            last_hidden_state = outputs.last_hidden_state[0]  # Shape: (seq_len + 2, hidden_dim)

        if self.config.pooling_strategy == "mean":
            # Exclude special tokens: BOS (idx 0) and EOS (idx -1)
            residue_embeddings = last_hidden_state[1:-1]
            pooled = residue_embeddings.mean(dim=0)
        elif self.config.pooling_strategy in ("bos", "cls"):
            pooled = last_hidden_state[0]
        else:
            pooled = last_hidden_state[1:-1].mean(dim=0)

        return pooled.cpu().numpy().astype(np.float32)

    def extract_batch(
        self,
        variants_df: pd.DataFrame,
        sequence_column: str = "mutant_sequence",
        id_column: str = "variant_id",
        max_rows: int | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Extract embeddings for a DataFrame of variants and build embedding index.

        Args:
            variants_df: DataFrame with sequence and id columns.
            sequence_column: Column containing protein sequences.
            id_column: Column containing variant identifiers.
            max_rows: Optional row count limit.

        Returns:
            (embeddings_matrix, index_dataframe)
        """
        subset = variants_df.head(max_rows) if max_rows is not None else variants_df
        embeddings: list[np.ndarray] = []
        index_entries: list[dict[str, Any]] = []

        for _, row in subset.iterrows():
            vid = str(row[id_column])
            seq = str(row[sequence_column])
            vec, entry = self.extract_embedding(seq, variant_id=vid)
            embeddings.append(vec)
            index_entries.append(asdict(entry))

        matrix = np.vstack(embeddings) if embeddings else np.empty((0, self.config.expected_dim or 480))
        index_df = pd.DataFrame(index_entries)
        return matrix, index_df


def save_embedding_index(index_df: pd.DataFrame, index_path: Path) -> pd.DataFrame:
    """Save embedding index table to Parquet and verify roundtrip."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_parquet(index_path, engine="pyarrow", index=False)
    return pd.read_parquet(index_path, engine="pyarrow")
