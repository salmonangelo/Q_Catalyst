"""
Unit and integration tests for W159H electrostatic embedding, force-field charge
assignment, ghost atom removal, atom completeness, and QM/MM boundary validation.
"""

from pathlib import Path
import numpy as np
import pytest

from qcatalyst.chemistry.amber_ff14sb import AMBER_FF14SB_CHARGES, get_amber_charge
from qcatalyst.chemistry.cluster import ActiveSiteClusterBuilder
from qcatalyst.chemistry.embedding import (
    EXPECTED_HID_ATOMS,
    EXPECTED_HIE_ATOMS,
    WT_TRP_INDOLE_GHOST_ATOMS,
    ElectrostaticEmbeddingBuilder,
)


@pytest.fixture
def qm_cluster_coords() -> np.ndarray:
    """Fixture providing active-site Tier S QM coordinates from 5XJH crystal structure."""
    builder = ActiveSiteClusterBuilder("data/structures/5XJH.pdb")
    tier_s = builder.build_tier_s()
    return np.array([coord for _, coord in tier_s["atoms"]], dtype=np.float64)


def test_amber_ff14sb_hid_hie_recognition():
    """Verify explicit HID/HIE recognition and strict force-field parameter lookup."""
    # 1. Parameter dictionary completeness
    assert "HID" in AMBER_FF14SB_CHARGES
    assert "HIE" in AMBER_FF14SB_CHARGES

    # 2. Key tautomeric atom charge lookups
    # HID: proton on ND1 (HD1), NE2 unprotonated
    assert np.isclose(get_amber_charge("HID", "ND1"), -0.3809)
    assert np.isclose(get_amber_charge("HID", "HD1"), 0.3649)
    assert np.isclose(get_amber_charge("HID", "NE2"), -0.5727)

    # HIE: proton on NE2 (HE2), ND1 unprotonated
    assert np.isclose(get_amber_charge("HIE", "NE2"), -0.3456)
    assert np.isclose(get_amber_charge("HIE", "HE2"), 0.3212)
    assert np.isclose(get_amber_charge("HIE", "ND1"), -0.5292)

    # 3. Strict failure policy: never silently assign 0.0 to unsupported atoms
    with pytest.raises(KeyError, match="Unsupported atom 'NE1' in residue 'HID'"):
        get_amber_charge("HID", "NE1", strict=True)

    with pytest.raises(KeyError, match="Unsupported atom 'XYZ' in residue 'HIE'"):
        get_amber_charge("HIE", "XYZ", strict=True)

    with pytest.raises(KeyError, match="Unsupported residue 'NONEXISTENT'"):
        get_amber_charge("NONEXISTENT", "CA", strict=True)


def test_w159h_embedding_ghost_atom_removal(qm_cluster_coords: np.ndarray, tmp_path: Path):
    """Verify that WT Trp159 indole atoms are completely removed and cannot contaminate His159."""
    embed_builder = ElectrostaticEmbeddingBuilder("data/structures/5XJH.pdb")

    # Generate W159H embedding using default validated candidate
    coords_hid, charges_hid, meta_hid = embed_builder.get_environment_charges(
        qm_coords=qm_cluster_coords, radius_cutoff=10.0, mutation="W159H"
    )

    assert "mutant_residue_159" in meta_hid
    m_info = meta_hid["mutant_residue_159"]
    assert m_info["res_name"] == "HID"
    assert m_info["ghost_atoms_count"] == 0

    # Corrupt a mutant PDB with residual WT indole atom (e.g. NE1, CZ2)
    base_hid_pdb = Path("results/structures/W159H/W159H_HID_t+80.pdb")
    if not base_hid_pdb.exists():
        pytest.skip(f"Candidate PDB not found at {base_hid_pdb}")

    corrupt_lines = []
    with open(base_hid_pdb, "r", encoding="utf-8") as f:
        for line in f:
            corrupt_lines.append(line)
            if "HID A 159" in line and "NE2" in line:
                # Inject a ghost NE1 indole atom
                corrupt_lines.append(
                    "ATOM    999  NE1 HID A 159      -2.000  25.000 -23.000  1.00  9.32           N\n"
                )

    corrupt_pdb = tmp_path / "corrupt_ghost.pdb"
    corrupt_pdb.write_text("".join(corrupt_lines), encoding="utf-8")

    # Embedding engine must detect and reject indole ghost atoms
    with pytest.raises(ValueError, match="Residual WT Trp159 indole ghost atoms detected"):
        embed_builder.get_environment_charges(
            qm_coords=qm_cluster_coords,
            radius_cutoff=10.0,
            mutation="W159H",
            mutant_pdb_path=corrupt_pdb,
        )


def test_w159h_embedding_his159_atom_completeness(qm_cluster_coords: np.ndarray, tmp_path: Path):
    """Verify that His159 has all 17 required atoms and fails explicitly if incomplete."""
    embed_builder = ElectrostaticEmbeddingBuilder("data/structures/5XJH.pdb")

    # 1. HID completeness (17 atoms)
    _, _, meta_hid = embed_builder.get_environment_charges(
        qm_coords=qm_cluster_coords, radius_cutoff=10.0, mutation="W159H_HID"
    )
    assert meta_hid["mutant_residue_159"]["atom_count"] == 17

    # 2. HIE completeness (17 atoms)
    _, _, meta_hie = embed_builder.get_environment_charges(
        qm_coords=qm_cluster_coords, radius_cutoff=10.0, mutation="W159H_HIE"
    )
    assert meta_hie["mutant_residue_159"]["atom_count"] == 17

    # 3. Test missing atom failure: drop HD1 from HID candidate
    base_hid_pdb = Path("results/structures/W159H/W159H_HID_t+80.pdb")
    if not base_hid_pdb.exists():
        pytest.skip(f"Candidate PDB not found at {base_hid_pdb}")

    incomplete_lines = [
        line for line in open(base_hid_pdb, encoding="utf-8")
        if not ("HID A 159" in line and "HD1" in line)
    ]
    incomplete_pdb = tmp_path / "incomplete_hid.pdb"
    incomplete_pdb.write_text("".join(incomplete_lines), encoding="utf-8")

    with pytest.raises(ValueError, match="missing expected force-field atoms"):
        embed_builder.get_environment_charges(
            qm_coords=qm_cluster_coords,
            radius_cutoff=10.0,
            mutation="W159H",
            mutant_pdb_path=incomplete_pdb,
        )


def test_w159h_embedding_charge_validation(qm_cluster_coords: np.ndarray):
    """Verify atom charges, residue charge reporting, and total MM charge."""
    embed_builder = ElectrostaticEmbeddingBuilder("data/structures/5XJH.pdb")

    # Check HID
    coords_hid, charges_hid, meta_hid = embed_builder.get_environment_charges(
        qm_coords=qm_cluster_coords, radius_cutoff=10.0, mutation="W159H_HID"
    )
    # None of the 17 atoms should have zero charge (every atom has non-zero ff14SB parameter)
    # The residue charge must match the exact sum of HID tabulated parameters
    expected_hid_sum = round(sum(AMBER_FF14SB_CHARGES["HID"].values()), 4)
    assert np.isclose(meta_hid["mutant_residue_159"]["residue_charge"], expected_hid_sum)
    assert np.isclose(meta_hid["net_mm_charge"], np.sum(charges_hid), atol=1e-3)

    # Check HIE
    coords_hie, charges_hie, meta_hie = embed_builder.get_environment_charges(
        qm_coords=qm_cluster_coords, radius_cutoff=10.0, mutation="W159H_HIE"
    )
    expected_hie_sum = round(sum(AMBER_FF14SB_CHARGES["HIE"].values()), 4)
    assert np.isclose(meta_hie["mutant_residue_159"]["residue_charge"], expected_hie_sum)
    assert np.isclose(meta_hie["net_mm_charge"], np.sum(charges_hie), atol=1e-3)

    # Verify that HID and HIE produce distinct point-charge fields
    assert not np.allclose(charges_hid, charges_hie)


def test_w159h_embedding_qm_mm_boundary_and_scaffold_hydrogens(qm_cluster_coords: np.ndarray):
    """Verify QM/MM boundary distance safety and scaffold hydrogen reporting."""
    embed_builder = ElectrostaticEmbeddingBuilder("data/structures/5XJH.pdb")

    coords_hid, _, meta_hid = embed_builder.get_environment_charges(
        qm_coords=qm_cluster_coords, radius_cutoff=10.0, mutation="W159H"
    )

    # 1. Boundary check: ensure no point charge dangerously penetrates the QM cluster core (< 1.0 A)
    assert meta_hid["qm_mm_boundary_intact"] is True
    assert meta_hid["min_dist_to_qm_A"] >= 1.0

    # 2. Scaffold hydrogen detection: 5XJH crystal structure lacks explicit hydrogens
    assert meta_hid["scaffold_has_hydrogens"] is False
    assert meta_hid["scaffold_hydrogens_detected"] == 0
    assert "lacks explicit hydrogens" in meta_hid["scaffold_missing_hydrogens_warning"]

    # 3. Preservation of WT behavior
    coords_wt, charges_wt, meta_wt = embed_builder.get_environment_charges(
        qm_coords=qm_cluster_coords, radius_cutoff=10.0, mutation="WT"
    )
    assert "mutant_residue_159" not in meta_wt
    assert meta_wt["n_point_charges"] == 563
    # WT Trp159 (14 heavy atoms) replaced by His159 (17 atoms including explicit H) = +3 atoms
    assert meta_hid["n_point_charges"] == len(charges_wt) + 3

