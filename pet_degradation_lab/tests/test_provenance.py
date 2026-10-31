"""Unit tests verifying provenance preservation and evidence classification invariants."""
import pytest
from pydantic import ValidationError

from schemas.evidence_status import EvidenceClassification
from schemas.provenance import ProvenanceRecord


def test_evidence_classification_categories():
    expected = {
        "EXPERIMENTAL",
        "COMPUTED",
        "PREDICTED",
        "REPORTED",
        "NOT_AVAILABLE",
        "PENDING_VALIDATION",
    }
    actual = {e.value for e in EvidenceClassification}
    assert actual == expected


def test_provenance_record_creation_and_integrity():
    rec = ProvenanceRecord(
        source_file="5XJH.pdb",
        source_repo_or_origin="Protein Data Bank",
        source_relative_path="data/structures/5XJH.pdb",
        original_field_name="CRYST1 / ATOM",
        source_sha256="a0410e1407" + "0" * 54,
        software_and_method="X-ray crystallography (1.5Å)",
        evidence_classification=EvidenceClassification.EXPERIMENTAL,
        validation_warnings=["Crystallographic B-factors elevated in loop region"],
        raw_attributes={"r_free": 0.21, "resolution_angstrom": 1.5},
    )

    assert rec.source_file == "5XJH.pdb"
    assert rec.evidence_classification == EvidenceClassification.EXPERIMENTAL
    assert len(rec.validation_warnings) == 1
    assert rec.raw_attributes["resolution_angstrom"] == 1.5


def test_provenance_record_rejects_missing_required_fields():
    with pytest.raises(ValidationError):
        # Missing source_sha256 and evidence_classification
        ProvenanceRecord(
            source_file="test.pdb",
            source_repo_or_origin="origin",
            source_relative_path="path",
        )
