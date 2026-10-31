"""Unit tests verifying amino-acid sequence validation and mutation alignment."""
import pytest
from pydantic import ValidationError

from schemas.evidence_status import EvidenceClassification
from schemas.provenance import ProvenanceRecord
from schemas.sequence import (
    SequenceRecord,
    WTReferenceRecord,
    align_and_extract_mutations,
    compute_sequence_checksum,
)


@pytest.fixture
def dummy_prov():
    return ProvenanceRecord(
        source_file="test_seq.fasta",
        source_repo_or_origin="Test Fixture",
        source_relative_path="tests/fixtures/test_seq.fasta",
        source_sha256="0" * 64,
        evidence_classification=EvidenceClassification.REPORTED,
    )


def test_valid_sequence_record(dummy_prov):
    seq_str = "MNFPRASRLMQAAVLGGLMAVSAAATAQT"
    rec = SequenceRecord(
        variant_id="VAR-TEST-001",
        parent_enzyme="IsPETase",
        sequence=seq_str,
        sequence_length=len(seq_str),
        sequence_checksum=compute_sequence_checksum(seq_str),
        provenance=dummy_prov,
    )
    assert rec.sequence == seq_str
    assert rec.sequence_length == 29
    assert len(rec.sequence_checksum) == 64


def test_invalid_amino_acid_rejected(dummy_prov):
    with pytest.raises(ValidationError, match="non-canonical amino acids"):
        SequenceRecord(
            variant_id="VAR-TEST-BAD",
            parent_enzyme="IsPETase",
            sequence="MNFPRBZXJQAAVLG",  # B, Z, X, J are non-canonical
            sequence_length=15,
            sequence_checksum="dummy",
            provenance=dummy_prov,
        )


def test_align_and_extract_mutations_substitutions():
    wt = "MNFPRASRLM"
    # S7E mutation (index 6, 1-based pos 7)
    mut = "MNFPRAERLM"
    diffs = align_and_extract_mutations(wt_seq=wt, mut_seq=mut, offset=1)
    assert len(diffs) == 1
    assert diffs[0].mutation_str == "S7E"
    assert diffs[0].wt_residue == "S"
    assert diffs[0].mut_residue == "E"
    assert diffs[0].position_1based == 7
    assert diffs[0].is_substitution is True
    assert diffs[0].alignment_conflict is False


def test_align_and_extract_mutations_flags_indels():
    wt = "MNFPRASRLM"
    mut_ins = "MNFPRASRLMAA"  # 2 amino acid insertion
    diffs = align_and_extract_mutations(wt_seq=wt, mut_seq=mut_ins, offset=1)
    assert len(diffs) == 1
    assert diffs[0].is_insertion is True
    assert diffs[0].alignment_conflict is True
    assert "insertion suspected" in diffs[0].review_flags[0]
