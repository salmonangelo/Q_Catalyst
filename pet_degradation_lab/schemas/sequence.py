"""Sequence schemas and amino-acid validation models."""
from __future__ import annotations

import hashlib
import re
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from schemas.provenance import ProvenanceRecord

CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


class SequenceRecord(BaseModel):
    """Full amino-acid sequence record."""
    variant_id: str = Field(..., description="Unique variant identifier")
    parent_enzyme: str = Field(..., description="Parent wild-type enzyme name (e.g. IsPETase)")
    sequence: str = Field(..., description="Full single-letter amino-acid sequence")
    sequence_length: int = Field(..., description="Length of the sequence")
    sequence_checksum: str = Field(..., description="SHA-256 hash of uppercase sequence string")
    validation_status: str = Field("VALID", description="VALID, INVALID, or AMBIGUOUS")
    validation_notes: List[str] = Field(default_factory=list, description="Validation notes or residue flags")
    provenance: ProvenanceRecord = Field(..., description="Provenance information")

    @field_validator("sequence")
    @classmethod
    def validate_amino_acids(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("Sequence cannot be empty")
        invalid = set(cleaned) - CANONICAL_AMINO_ACIDS
        if invalid:
            # Check if contains ambiguous residues B, Z, J, X or gaps
            raise ValueError(f"Sequence contains non-canonical amino acids: {sorted(invalid)}")
        return cleaned


class WTReferenceRecord(BaseModel):
    """Wild-type reference sequence specification."""
    wildtype_id: str = Field(..., description="Identifier for reference enzyme (e.g. IsPETase_WT)")
    sequence: str = Field(..., description="Full wild-type reference sequence")
    sequence_length: int = Field(..., description="Length of the WT sequence")
    source_identifier: str = Field(..., description="NCBI, UniProt, or PDB reference ID (e.g. 5XJH)")
    checksum: str = Field(..., description="SHA-256 hash of uppercase reference sequence")
    provenance: ProvenanceRecord = Field(..., description="Provenance of reference sequence")


class MutationDifferenceRecord(BaseModel):
    """
    Explicit mutation difference generated after sequence alignment against WT reference.
    """
    position_1based: int = Field(..., description="1-based sequence position in the full sequence")
    wt_residue: str = Field(..., description="Expected wild-type amino acid")
    mut_residue: str = Field(..., description="Mutant amino acid")
    mutation_str: str = Field(..., description="Standard notation token, e.g. 'S121E'")
    is_substitution: bool = Field(True, description="True if single residue substitution")
    is_insertion: bool = Field(False, description="True if insertion detected")
    is_deletion: bool = Field(False, description="True if deletion detected")
    alignment_conflict: bool = Field(False, description="True if WT residue does not match reference sequence")
    review_flags: List[str] = Field(default_factory=list, description="Review flags requiring human scientist attention")


def compute_sequence_checksum(sequence: str) -> str:
    """Computes standard uppercase SHA-256 sequence hash."""
    return hashlib.sha256(sequence.strip().upper().encode("utf-8")).hexdigest()


def align_and_extract_mutations(
    wt_seq: str,
    mut_seq: str,
    offset: int = 1
) -> List[MutationDifferenceRecord]:
    """
    Compares mutant sequence to WT reference and extracts differences with strict alignment checks.
    """
    wt_clean = wt_seq.strip().upper()
    mut_clean = mut_seq.strip().upper()

    diffs: List[MutationDifferenceRecord] = []

    if len(wt_clean) != len(mut_clean):
        # Indel detected
        if len(mut_clean) > len(wt_clean):
            diffs.append(MutationDifferenceRecord(
                position_1based=1,
                wt_residue="-",
                mut_residue="+",
                mutation_str=f"INS_len_{len(mut_clean)-len(wt_clean)}",
                is_substitution=False,
                is_insertion=True,
                is_deletion=False,
                alignment_conflict=True,
                review_flags=["Sequence length mismatch: insertion suspected. Human review required."]
            ))
        else:
            diffs.append(MutationDifferenceRecord(
                position_1based=1,
                wt_residue="+",
                mut_residue="-",
                mutation_str=f"DEL_len_{len(wt_clean)-len(mut_clean)}",
                is_substitution=False,
                is_insertion=False,
                is_deletion=True,
                alignment_conflict=True,
                review_flags=["Sequence length mismatch: deletion suspected. Human review required."]
            ))
        return diffs

    for idx, (wt_aa, mut_aa) in enumerate(zip(wt_clean, mut_clean)):
        pos = idx + offset
        if wt_aa != mut_aa:
            token = f"{wt_aa}{pos}{mut_aa}"
            diffs.append(MutationDifferenceRecord(
                position_1based=pos,
                wt_residue=wt_aa,
                mut_residue=mut_aa,
                mutation_str=token,
                is_substitution=True,
                is_insertion=False,
                is_deletion=False,
                alignment_conflict=False,
                review_flags=[]
            ))

    return diffs
