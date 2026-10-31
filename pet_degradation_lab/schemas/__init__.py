"""Standardized scientific data schemas for pet_degradation_lab."""
from schemas.evidence_status import EvidenceClassification
from schemas.provenance import ProvenanceRecord
from schemas.sequence import (
    SequenceRecord,
    WTReferenceRecord,
    MutationDifferenceRecord,
    compute_sequence_checksum,
    align_and_extract_mutations,
)
from schemas.structure import ResidueMappingEntry, StructuralModelRecord
from schemas.structural_risk import StructuralRiskRecord
from schemas.dft import DFTResultRecord
from schemas.quantum import QuantumResultRecord
from schemas.candidate import AIPredictionRecord, CandidateVariant

__all__ = [
    "EvidenceClassification",
    "ProvenanceRecord",
    "SequenceRecord",
    "WTReferenceRecord",
    "MutationDifferenceRecord",
    "compute_sequence_checksum",
    "align_and_extract_mutations",
    "ResidueMappingEntry",
    "StructuralModelRecord",
    "StructuralRiskRecord",
    "DFTResultRecord",
    "QuantumResultRecord",
    "AIPredictionRecord",
    "CandidateVariant",
]
