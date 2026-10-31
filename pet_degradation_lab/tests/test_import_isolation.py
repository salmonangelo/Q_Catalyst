"""Isolation test suite verifying that pet_degradation_lab never imports qcatalyst
and enforces non-invention of missing scientific data.
"""
import sys
import pytest

from schemas.evidence_status import EvidenceClassification
from schemas.provenance import ProvenanceRecord
from schemas.dft import DFTResultRecord
from schemas.quantum import QuantumResultRecord
from schemas.candidate import CandidateVariant
from ingestion.validation import ScientificValidator


def test_zero_qcatalyst_imports_in_sys_modules():
    """Confirms that no qcatalyst modules are imported anywhere in the module runtime."""
    imported_qcatalyst = [mod for mod in sys.modules if "qcatalyst" in mod]
    assert len(imported_qcatalyst) == 0, f"VIOLATION: Found imported qcatalyst modules: {imported_qcatalyst}"


def test_dft_result_preserves_method_basis_and_units():
    prov = ProvenanceRecord(
        source_file="chem_results.parquet",
        source_repo_or_origin="Test Snapshot",
        source_relative_path="raw/results/chemistry/chem_results.parquet",
        source_sha256="0" * 64,
        evidence_classification=EvidenceClassification.COMPUTED,
    )

    dft = DFTResultRecord(
        calculation_id="CHEM_VAR_A4_01",
        variant_id="VAR_A4_01",
        mutation_str="S121E",
        calculated_energy_hartree=-1029.538509,
        energy_units="Hartree",
        method="DFT",
        functional="B3LYP",
        basis_set="def2-SVP",
        software_and_version="PySCF 2.14.0",
        net_charge=-1,
        spin_multiplicity=1,
        descriptors={"delta_esp_kcal_mol": 9.18, "delta_charge_ser_og": -0.0436},
        calculation_status="CONVERGED",
        provenance=prov,
    )

    assert dft.method == "DFT"
    assert dft.functional == "B3LYP"
    assert dft.basis_set == "def2-SVP"
    assert dft.energy_units == "Hartree"
    assert dft.calculated_energy_hartree == -1029.538509
    assert dft.descriptors["delta_esp_kcal_mol"] == 9.18


def test_missing_quantum_remains_not_available_without_dummy_values():
    candidate = CandidateVariant(
        variant_id="VAR-NO-QUANTUM",
        parent_enzyme="IsPETase",
        mutations_raw="T88A",
        mutations_canonical="T88A",
        mutation_count=1,
        quantum_result=None,
        evidence_status={
            "quantum": EvidenceClassification.NOT_AVAILABLE
        }
    )

    # Scientific validator must pass when missing data is explicitly None
    is_valid, issues = ScientificValidator.validate_candidate_integrity(candidate)
    assert is_valid is True
    assert len(issues) == 0


def test_scientific_validator_flags_dummy_invention_violations():
    prov = ProvenanceRecord(
        source_file="dummy.parquet",
        source_repo_or_origin="Test",
        source_relative_path="test",
        source_sha256="0" * 64,
        evidence_classification=EvidenceClassification.NOT_AVAILABLE,
    )

    # Fabricating a dummy VQE energy while marking status NOT_AVAILABLE
    dummy_quantum = QuantumResultRecord(
        calculation_id="FAKE_01",
        variant_id="VAR-FAKE",
        vqe_energy_hartree=-78.4520,  # Fabricated value
        provenance=prov,
    )

    candidate = CandidateVariant(
        variant_id="VAR-FAKE",
        parent_enzyme="IsPETase",
        mutations_raw="A162G",
        mutations_canonical="A162G",
        mutation_count=1,
        quantum_result=dummy_quantum,
        evidence_status={
            "quantum": EvidenceClassification.NOT_AVAILABLE
        }
    )

    is_valid, issues = ScientificValidator.validate_candidate_integrity(candidate)
    assert is_valid is False
    assert any("VIOLATION: Quantum marked NOT_AVAILABLE but contains non-null VQE value" in i for i in issues)
