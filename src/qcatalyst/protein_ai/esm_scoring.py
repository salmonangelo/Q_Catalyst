"""ESM-1v mutation-effect scoring module for Q-Catalyst.

Provides modular zero-shot mutation effect prediction using masked-marginal
log-likelihood ratio (LLR) scoring from pretrained protein language models.
Supports offline-cached weights and a deterministic mocked fallback for CI/home testing
without requiring large checkpoint downloads.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from qcatalyst.data.mutation import (
    canonicalize_mutations,
    parse_mutations,
    validate_mutations,
)

# Standard BLOSUM62 log-odds substitution matrix (used by MockESMModel as a realistic offline proxy)
BLOSUM62_MATRIX: dict[tuple[str, str], int] = {
    ("A", "A"): 4, ("A", "R"): -1, ("A", "N"): -2, ("A", "D"): -2, ("A", "C"): 0, ("A", "Q"): -1,
    ("A", "E"): -1, ("A", "G"): 0, ("A", "H"): -2, ("A", "I"): -1, ("A", "L"): -1, ("A", "K"): -1,
    ("A", "M"): -1, ("A", "F"): -2, ("A", "P"): -1, ("A", "S"): 1, ("A", "T"): 0, ("A", "W"): -3,
    ("A", "Y"): -2, ("A", "V"): 0,
    ("R", "R"): 5, ("R", "N"): 0, ("R", "D"): -2, ("R", "C"): -3, ("R", "Q"): 1, ("R", "E"): 0,
    ("R", "G"): -2, ("R", "H"): 0, ("R", "I"): -3, ("R", "L"): -2, ("R", "K"): 2, ("R", "M"): -1,
    ("R", "F"): -3, ("R", "P"): -2, ("R", "S"): -1, ("R", "T"): -1, ("R", "W"): -3, ("R", "Y"): -2,
    ("R", "V"): -3,
    ("N", "N"): 6, ("N", "D"): 1, ("N", "C"): -3, ("N", "Q"): 0, ("N", "E"): 0, ("N", "G"): 0,
    ("N", "H"): 1, ("N", "I"): -3, ("N", "L"): -3, ("N", "K"): 0, ("N", "M"): -2, ("N", "F"): -3,
    ("N", "P"): -2, ("N", "S"): 1, ("N", "T"): 0, ("N", "W"): -4, ("N", "Y"): -2, ("N", "V"): -3,
    ("D", "D"): 6, ("D", "C"): -3, ("D", "Q"): 0, ("D", "E"): 2, ("D", "G"): -1, ("D", "H"): -1,
    ("D", "I"): -3, ("D", "L"): -4, ("D", "K"): -1, ("D", "M"): -3, ("D", "F"): -3, ("D", "P"): -1,
    ("D", "S"): 0, ("D", "T"): -1, ("D", "W"): -4, ("D", "Y"): -3, ("D", "V"): -3,
    ("C", "C"): 9, ("C", "Q"): -3, ("C", "E"): -4, ("C", "G"): -3, ("C", "H"): -3, ("C", "I"): -1,
    ("C", "L"): -1, ("C", "K"): -3, ("C", "M"): -1, ("C", "F"): -2, ("C", "P"): -3, ("C", "S"): -1,
    ("C", "T"): -1, ("C", "W"): -2, ("C", "Y"): -2, ("C", "V"): -1,
    ("Q", "Q"): 5, ("Q", "E"): 2, ("Q", "G"): -2, ("Q", "H"): 0, ("Q", "I"): -3, ("Q", "L"): -2,
    ("Q", "K"): 1, ("Q", "M"): 0, ("Q", "F"): -3, ("Q", "P"): -1, ("Q", "S"): 0, ("Q", "T"): -1,
    ("Q", "W"): -2, ("Q", "Y"): -1, ("Q", "V"): -2,
    ("E", "E"): 5, ("E", "G"): -2, ("E", "H"): 0, ("E", "I"): -3, ("E", "L"): -3, ("E", "K"): 1,
    ("E", "M"): -2, ("E", "F"): -3, ("E", "P"): -1, ("E", "S"): 0, ("E", "T"): -1, ("E", "W"): -3,
    ("E", "Y"): -2, ("E", "V"): -2,
    ("G", "G"): 6, ("G", "H"): -2, ("G", "I"): -4, ("G", "L"): -4, ("G", "K"): -2, ("G", "M"): -3,
    ("G", "F"): -3, ("G", "P"): -2, ("G", "S"): 0, ("G", "T"): -2, ("G", "W"): -2, ("G", "Y"): -3,
    ("G", "V"): -3,
    ("H", "H"): 8, ("H", "I"): -3, ("H", "L"): -3, ("H", "K"): -1, ("H", "M"): -2, ("H", "F"): -1,
    ("H", "P"): -2, ("H", "S"): -1, ("H", "T"): -2, ("H", "W"): -2, ("H", "Y"): 2, ("H", "V"): -3,
    ("I", "I"): 4, ("I", "L"): 2, ("I", "K"): -3, ("I", "M"): 1, ("I", "F"): 0, ("I", "P"): -3,
    ("I", "S"): -2, ("I", "T"): -1, ("I", "W"): -3, ("I", "Y"): -1, ("I", "V"): 3,
    ("L", "L"): 4, ("L", "K"): -2, ("L", "M"): 2, ("L", "F"): 0, ("L", "P"): -3, ("L", "S"): -2,
    ("L", "T"): -1, ("L", "W"): -2, ("L", "Y"): -1, ("L", "V"): 1,
    ("K", "K"): 5, ("K", "M"): -1, ("K", "F"): -3, ("K", "P"): -1, ("K", "S"): 0, ("K", "T"): -1,
    ("K", "W"): -3, ("K", "Y"): -2, ("K", "V"): -2,
    ("M", "M"): 5, ("M", "F"): 0, ("M", "P"): -2, ("M", "S"): -1, ("M", "T"): -1, ("M", "W"): -1,
    ("M", "Y"): -1, ("M", "V"): 1,
    ("F", "F"): 6, ("F", "P"): -4, ("F", "S"): -2, ("F", "T"): -2, ("F", "W"): 1, ("F", "Y"): 3,
    ("F", "V"): -1,
    ("P", "P"): 7, ("P", "S"): -1, ("P", "T"): -1, ("P", "W"): -4, ("P", "Y"): -3, ("P", "V"): -2,
    ("S", "S"): 4, ("S", "T"): 1, ("S", "W"): -3, ("S", "Y"): -2, ("S", "V"): -2,
    ("T", "T"): 5, ("T", "W"): -2, ("T", "Y"): -2, ("T", "V"): 0,
    ("W", "W"): 11, ("W", "Y"): 2, ("W", "V"): -3,
    ("Y", "Y"): 7, ("Y", "V"): -1,
    ("V", "V"): 4,
}


def get_blosum62_score(aa1: str, aa2: str) -> int:
    """Symmetric lookup in BLOSUM62 matrix."""
    return BLOSUM62_MATRIX.get((aa1, aa2), BLOSUM62_MATRIX.get((aa2, aa1), -4))


@dataclass(frozen=True)
class ESMScoreResult:
    """Standardized output record for an ESM-1v mutation score prediction."""

    variant_id: str
    parent_enzyme: str
    mutant_raw: str
    mutations_canonical: str
    num_mutations: int
    is_multi_mutant: bool
    score: float
    scoring_method: str
    model_name: str
    model_version: str
    sequence_hash: str
    run_status: str
    is_prediction: bool = True
    disclaimer: str = (
        "Computational zero-shot likelihood ratio; not experimentally validated activity."
    )


@dataclass
class ESMScorerConfig:
    """Configuration for ESM scoring model and inference environment."""

    model_name: str = "facebook/esm1v_t33_650M_UR90S_1"
    model_version: str = "v1"
    scoring_strategy: str = "masked_marginal"
    cache_dir: str | Path | None = None
    device: str = "cpu"
    mock: bool = True  # Default to True during local testing to avoid 2.6GB download
    batch_size: int = 1


class ESMScorer:
    """Modular zero-shot scorer for ESM-1v with masked-marginal scoring.

    Attributes:
        config: ESMScorerConfig specifying model parameters and execution mode.
    """

    def __init__(self, config: ESMScorerConfig | None = None) -> None:
        self.config = config or ESMScorerConfig()
        self._model = None
        self._tokenizer = None

        if not self.config.mock:
            self._load_real_model()

    def _load_real_model(self) -> None:
        """Load pretrained model from local cache directory with strict offline guarantees."""
        cache_dir = Path(self.config.cache_dir) if self.config.cache_dir else None

        if cache_dir is None or not cache_dir.exists():
            raise FileNotFoundError(
                f"Local ESM-1v model checkpoint not found in configured cache: '{cache_dir}'. "
                "Per project rules, large checkpoints (>200MB) must not be downloaded during home sessions. "
                "To run locally, initialize ESMScorer with config.mock=True."
            )

        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(cache_dir), local_files_only=True
            )
            self._model = AutoModelForMaskedLM.from_pretrained(
                str(cache_dir), local_files_only=True
            )
            self._model.to(self.config.device)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load cached ESM model from {cache_dir}: {exc}"
            ) from exc

    def compute_sequence_hash(self, wt_seq: str, mutant_str: str) -> str:
        """Compute stable SHA256 sequence hash for provenance tracking."""
        key = f"{wt_seq.strip()}:{canonicalize_mutations(mutant_str)}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _mock_masked_marginal_score(self, wt_aa: str, pos: int, mut_aa: str) -> float:
        """Compute realistic zero-shot score using calibrated log-odds substitution differences.

        LLR approximation: (BLOSUM62(wt, mut) - BLOSUM62(wt, wt)) / 2.0
        Yields realistic negative values for deleterious changes and neutral/positive for conservative.
        """
        wt_self = get_blosum62_score(wt_aa, wt_aa)
        wt_mut = get_blosum62_score(wt_aa, mut_aa)
        # Scaled to typical ESM-1v LLR scale (roughly -12 to +2)
        llr = (wt_mut - wt_self) / 2.0
        return round(llr, 4)

    def score_variant(
        self,
        wt_seq: str,
        mutant_str: str,
        parent_enzyme: str = "IsPETase",
        variant_id: str = "",
    ) -> ESMScoreResult:
        """Calculate mutation effect score for a variant sequence.

        Args:
            wt_seq: Wild-type parent amino acid sequence.
            mutant_str: Substitution string (e.g. 'N37P' or 'A1M:C3D' or 'WT').
            parent_enzyme: Parent enzyme name.
            variant_id: Optional stable variant ID.

        Returns:
            ESMScoreResult containing log-likelihood ratio and run metadata.

        Raises:
            ValueError: If sequence or mutations are invalid or out of bounds.
        """
        if not wt_seq or not isinstance(wt_seq, str):
            raise ValueError("Wild-type sequence must be a non-empty string.")

        parsed_muts = parse_mutations(mutant_str)
        canonical_mut = canonicalize_mutations(parsed_muts)
        vid = variant_id or f"{parent_enzyme}_{canonical_mut}"

        # Validate mutations against WT sequence
        if parsed_muts:
            is_valid, errors = validate_mutations(wt_seq, parsed_muts)
            if not is_valid:
                raise ValueError(
                    f"Invalid mutation for {parent_enzyme} '{mutant_str}': {'; '.join(errors)}"
                )

        seq_hash = self.compute_sequence_hash(wt_seq, mutant_str)
        num_muts = len(parsed_muts)
        is_multi = num_muts > 1

        # Wild-type score is identically 0.0
        if not parsed_muts:
            return ESMScoreResult(
                variant_id=vid,
                parent_enzyme=parent_enzyme,
                mutant_raw=mutant_str,
                mutations_canonical="WT",
                num_mutations=0,
                is_multi_mutant=False,
                score=0.0,
                scoring_method=self.config.scoring_strategy,
                model_name=self.config.model_name,
                model_version=self.config.model_version,
                sequence_hash=seq_hash,
                run_status="mocked" if self.config.mock else "success",
            )

        # Compute score across all mutation positions
        if self.config.mock:
            # Masked-marginal scoring via calibrated proxy
            total_score = 0.0
            for wt_aa, pos, mut_aa in parsed_muts:
                total_score += self._mock_masked_marginal_score(wt_aa, pos, mut_aa)
            run_status = "mocked"
        else:
            total_score = self._score_with_torch_model(wt_seq, parsed_muts)
            run_status = "success"

        return ESMScoreResult(
            variant_id=vid,
            parent_enzyme=parent_enzyme,
            mutant_raw=mutant_str,
            mutations_canonical=canonical_mut,
            num_mutations=num_muts,
            is_multi_mutant=is_multi,
            score=round(total_score, 4),
            scoring_method=self.config.scoring_strategy,
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            sequence_hash=seq_hash,
            run_status=run_status,
        )

    def _score_with_torch_model(
        self,
        wt_seq: str,
        parsed_muts: list[tuple[str, int, str]],
    ) -> float:
        """Run real forward passes with PyTorch and masked marginals."""
        import torch

        total_llr = 0.0
        # For masked marginal scoring, mask each position independently
        for wt_aa, pos, mut_aa in parsed_muts:
            zero_based_pos = pos - 1
            masked_chars = list(wt_seq)
            # Replace position with tokenizer's mask token
            masked_chars[zero_based_pos] = self._tokenizer.mask_token
            masked_seq = "".join(masked_chars)

            inputs = self._tokenizer(masked_seq, return_tensors="pt").to(self.config.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[0]  # Shape: (seq_len, vocab_size)

            # Account for CLS / BOS token offset
            tokens = self._tokenizer.tokenize(masked_seq)
            mask_token_idx = tokens.index(self._tokenizer.mask_token) + 1

            log_probs = torch.log_softmax(logits[mask_token_idx], dim=-1)
            wt_token_id = self._tokenizer.convert_tokens_to_ids(wt_aa)
            mut_token_id = self._tokenizer.convert_tokens_to_ids(mut_aa)

            llr = (log_probs[mut_token_id] - log_probs[wt_token_id]).item()
            total_llr += llr

        return total_llr

    def score_variants_dataframe(
        self,
        variants_df: pd.DataFrame,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        """Score rows in a variants DataFrame and return predictions DataFrame.

        Args:
            variants_df: Input DataFrame conforming to variants.parquet schema.
            max_rows: Optional row limit (useful for smoke tests).

        Returns:
            DataFrame matching results/predictions/esm1v_scores.parquet schema.
        """
        df_subset = variants_df.head(max_rows) if max_rows is not None else variants_df
        results: list[dict[str, Any]] = []

        for _, row in df_subset.iterrows():
            wt_seq = row["wildtype_sequence"]
            mutant_raw = row["mutant_raw"]
            parent_enzyme = row["parent_enzyme"]
            variant_id = row["variant_id"]

            res = self.score_variant(
                wt_seq=wt_seq,
                mutant_str=mutant_raw,
                parent_enzyme=parent_enzyme,
                variant_id=variant_id,
            )
            results.append(asdict(res))

        return pd.DataFrame(results)


def save_esm_predictions(
    predictions_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Save ESM predictions table to Parquet and verify round-trip reloading."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(output_path, engine="pyarrow", index=False)
    reloaded = pd.read_parquet(output_path, engine="pyarrow")
    return reloaded
