"""
Unit tests for Q-Catalyst Role 3 Quantum Safety Guard.
Proves that:
1. WT active-space Hamiltonian generation remains available and exact numerical results are preserved.
2. W159H (both HID and HIE tautomers) Hamiltonian generation is blocked.
3. The abort error message explicitly explains the 4 unresolved scientific blockers.
4. Direct embedding validation flags distinguish between valid WT and invalid W159H.
"""

import os
import json
import pytest
import numpy as np
from pathlib import Path

from qcatalyst.chemistry.cluster import ActiveSiteClusterBuilder
from qcatalyst.chemistry.embedding import (
    ElectrostaticEmbeddingBuilder,
    validate_embedding_for_quantum_execution,
)
from qcatalyst.quantum.hamiltonian import generate_active_space_hamiltonian
from qcatalyst.chemistry.pyscf_engine import PySCFChemistryEngine


EXPECTED_BLOCKERS = [
    "Unminimized steric clash: His159 HE1 is within 1.848 A of catalytic His237 carbonyl oxygen.",
    "Un-hydrogenated scaffold: 5XJH crystal structure lacks explicit hydrogens, resulting in an unphysical -85 e net MM charge bath.",
    "Non-neutral isolated AMBER ff14SB residue charges: tabulated HID (+0.1750 e) and HIE (+0.1358 e) fragments lack integer neutrality.",
    "QM/MM boundary over-polarization: His159 C and O point charges reside 2.385 A from Ser160 capping hydrogens without link-atom electrostatic exclusion.",
]


def test_wt_hamiltonian_generation_remains_available():
    """Requirement 8A & 4: WT active-space Hamiltonian retrieval remains available and verified."""
    res = generate_active_space_hamiltonian(mutation="WT")

    assert res["status"] == "available"
    assert res["mutation"] == "WT"
    assert res["variant_id"] == "VAR_WT"
    assert res["scientific_validation_status"] == "PASS"

    artifacts = res["artifacts"]
    assert os.path.exists(artifacts["json"]), f"Missing {artifacts['json']}"
    assert os.path.exists(artifacts["npz"]), f"Missing {artifacts['npz']}"
    assert os.path.exists(artifacts["fcidump"]), f"Missing {artifacts['fcidump']}"

    # Verify numerical invariants preserved exactly
    meta = res["hamiltonian_metadata"]
    assert meta["electronic_structure"]["ncas_orbitals"] == 4
    assert meta["electronic_structure"]["nelecas"] == [2, 2]
    assert abs(meta["energies_hartree"]["e_casci_total"] - (-1024.2596230878212)) < 1e-10
    assert abs(meta["energies_hartree"]["e_hf_ref"] - (-1024.1988022357216)) < 1e-10


def test_w159h_hamiltonian_generation_is_blocked_with_blockers():
    """Requirement 8B & 8C: W159H Hamiltonian generation is blocked and reports all 4 blockers."""
    with pytest.raises(RuntimeError) as excinfo:
        generate_active_space_hamiltonian(mutation="W159H")

    err_msg = str(excinfo.value)
    assert "Active-space Hamiltonian generation aborted" in err_msg
    assert "QM/MM electrostatic embedding validation status is FAIL" in err_msg
    assert "Safety guard active: W159H quantum calculations are frozen" in err_msg

    # Verify each of the 4 unresolved scientific blockers is explicitly reported
    for blocker in EXPECTED_BLOCKERS:
        assert blocker in err_msg, f"Expected blocker missing from error message: {blocker}"


def test_w159h_hie_tautomer_also_blocked():
    """Verify that both neutral tautomers (HID and HIE) are safety-frozen."""
    with pytest.raises(RuntimeError) as excinfo:
        generate_active_space_hamiltonian(mutation="W159H_HIE")

    err_msg = str(excinfo.value)
    assert "Active-space Hamiltonian generation aborted" in err_msg
    for blocker in EXPECTED_BLOCKERS:
        assert blocker in err_msg


def test_embedding_metadata_validation_status():
    """Requirement 1 & 2: Embedding metadata explicitly marks W159H as FAIL and WT as PASS."""
    pdb_path = "data/structures/5XJH.pdb"
    builder = ActiveSiteClusterBuilder(pdb_path)
    cluster = builder.build_tier_s()
    qm_coords = np.array([c for _, c in cluster["atoms"]])

    embed_builder = ElectrostaticEmbeddingBuilder(pdb_path)

    # WT embedding
    _, _, wt_meta = embed_builder.get_environment_charges(qm_coords, radius_cutoff=10.0, mutation="WT")
    assert wt_meta["scientific_validation_status"] == "PASS"
    assert wt_meta["scientific_valid_for_quantum"] is True
    assert wt_meta["scientific_validation_blockers"] == []
    assert "passes standard boundary" in wt_meta["scientific_validation_reason"]

    # W159H embedding
    _, _, mut_meta = embed_builder.get_environment_charges(qm_coords, radius_cutoff=10.0, mutation="W159H")
    assert mut_meta["scientific_validation_status"] == "FAIL"
    assert mut_meta["scientific_valid_for_quantum"] is False
    assert len(mut_meta["scientific_validation_blockers"]) == 4
    for blocker in EXPECTED_BLOCKERS:
        assert blocker in mut_meta["scientific_validation_blockers"]


def test_validate_embedding_function_direct_guard():
    """Requirement 3: validate_embedding_for_quantum_execution raises on FAIL and passes on PASS."""
    # PASS case
    validate_embedding_for_quantum_execution({"scientific_validation_status": "PASS"})
    validate_embedding_for_quantum_execution(None)

    # FAIL case
    bad_meta = {
        "mutation": "W159H",
        "scientific_validation_status": "FAIL",
        "scientific_validation_blockers": EXPECTED_BLOCKERS,
    }
    with pytest.raises(RuntimeError) as excinfo:
        validate_embedding_for_quantum_execution(bad_meta)

    err = str(excinfo.value)
    assert "W159H quantum calculations are frozen" in err
    assert "Unminimized steric clash" in err


def test_pyscf_engine_export_blocked_on_invalid_metadata():
    """Verify PySCFChemistryEngine export blocks when given invalid validation metadata."""
    engine = PySCFChemistryEngine()
    fake_casci = {
        "ncas_orbitals": 4,
        "nelecas": (2, 2),
        "h1eff": np.zeros((4, 4)),
        "h2eff_2d": np.zeros((10, 10)),
        "h2eff_4d": np.zeros((4, 4, 4, 4)),
        "energy_core": -1000.0,
        "e_hf_ref": -1024.0,
        "e_casci_total": -1024.2,
        "correlation_energy_ha": -0.2,
        "mo_coeff_active": np.zeros((40, 4)),
    }
    invalid_meta = {
        "mutation": "W159H",
        "scientific_validation_status": "FAIL",
        "scientific_validation_blockers": EXPECTED_BLOCKERS,
    }

    with pytest.raises(RuntimeError) as excinfo:
        engine.export_active_space_hamiltonian(
            casci_result=fake_casci,
            output_dir="results/quantum",
            prefix="test_invalid",
            metadata=invalid_meta,
        )
    assert "W159H quantum calculations are frozen" in str(excinfo.value)

