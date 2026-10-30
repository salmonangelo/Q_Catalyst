"""Protein AI module for Q-Catalyst: ESM scoring, embeddings, and predictors."""

from qcatalyst.protein_ai.embeddings import (
    ESM2Embedder,
    ESM2EmbedderConfig,
    EmbeddingIndexEntry,
    compute_sequence_hash,
    save_embedding_index,
    validate_protein_sequence,
)
from qcatalyst.protein_ai.esm_scoring import (
    ESMScoreResult,
    ESMScorer,
    ESMScorerConfig,
    ModelWeightsNotAvailableError,
    save_esm_predictions,
)
from qcatalyst.protein_ai.predictor import (
    ModelMetrics,
    SupervisedModelConfig,
    SupervisedPETasePredictor,
    compute_regression_metrics,
    create_frozen_split,
)

__all__ = [
    # ESM-1v Scoring
    "ESMScorer",
    "ESMScorerConfig",
    "ESMScoreResult",
    "ModelWeightsNotAvailableError",
    "save_esm_predictions",
    # ESM-2 Embeddings
    "ESM2Embedder",
    "ESM2EmbedderConfig",
    "EmbeddingIndexEntry",
    "compute_sequence_hash",
    "validate_protein_sequence",
    "save_embedding_index",
    # Supervised Predictors
    "ModelMetrics",
    "SupervisedModelConfig",
    "SupervisedPETasePredictor",
    "compute_regression_metrics",
    "create_frozen_split",
]
