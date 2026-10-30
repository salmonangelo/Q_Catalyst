"""
Focused unit test suite for MutantStructureBuilder (Task: W159H in IsPETase).
Tests residue replacement, atom completeness, backbone preservation, peptide connectivity,
charge neutrality, and validation report outputs.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from qcatalyst.chemistry.mutant_builder import (
    MutantStructureBuilder,
    DUNBRACK_HISTIDINE_ROTAMERS,
)


@pytest.fixture
def builder():
    return MutantStructureBuilder("data/structures/5XJH.pdb")


def test_scaffold_target_verification(builder):
    """Verifies that residue 159 in 5XJH Chain A is strictly verified as TRP."""
    # Should pass without error
    builder.verify_target_residue("A", 159, "TRP")

    # Mismatched expected residue name should raise ValueError
    with pytest.raises(ValueError, match="Target residue mismatch"):
        builder.verify_target_residue("A", 159, "HIS")

    # Non-existent residue sequence number should raise ValueError
    with pytest.raises(ValueError, match="Target residue A:9999 not found"):
        builder.verify_target_residue("A", 9999, "TRP")


def test_histidine_building_atom_completeness_and_tautomers(builder):
    """Verifies that both HID and HIE tautomers are chemically complete with 17 atoms."""
    atom_map = {(a.res_seq, a.atom_name): a.coord for a in builder.atoms if a.chain_id == "A"}
    n = atom_map[(159, "N")]
    ca = atom_map[(159, "CA")]
    c = atom_map[(159, "C")]
    o = atom_map[(159, "O")]
    cb = atom_map[(159, "CB")]
    c_prev = atom_map[(158, "C")]

    for taut in ["HID", "HIE"]:
        atoms = builder._build_histidine_residue(
            n_coord=n,
            ca_coord=ca,
            c_coord=c,
            o_coord=o,
            cb_coord=cb,
            c_prev=c_prev,
            chi1_deg=180.0,
            chi2_deg=80.0,
            tautomer=taut,
        )

        # 1. Total atom count: 10 heavy + 7 hydrogens = 17
        assert len(atoms) == 17, f"Expected 17 atoms for {taut}, got {len(atoms)}"

        # 2. Heavy atom completeness
        expected_heavy = ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"]
        for name in expected_heavy:
            assert name in atoms, f"Missing heavy atom {name} in {taut}"

        # 3. Tautomer specific protonation
        if taut == "HID":
            assert "HD1" in atoms, "HID must have HD1 proton on ND1"
            assert "HE2" not in atoms, "HID must not have HE2"
        elif taut == "HIE":
            assert "HE2" in atoms, "HIE must have HE2 proton on NE2"
            assert "HD1" not in atoms, "HIE must not have HD1"


def test_backbone_preservation_and_peptide_connectivity(builder):
    """Verifies that backbone N, CA, C, O coordinates match WT with sub-milliangstrom fidelity."""
    # Run candidate generation without minimization for fast geometric verification
    reports = builder.build_w159h_candidates(
        output_dir="results/structures/W159H",
        minimize_structures=False,
    )

    atom_map = {(a.res_seq, a.atom_name): a.coord for a in builder.atoms if a.chain_id == "A"}
    wt_ca = atom_map[(159, "CA")]
    wt_n = atom_map[(159, "N")]
    wt_c = atom_map[(159, "C")]
    wt_o = atom_map[(159, "O")]
    gly158_c = atom_map[(158, "C")]
    ser160_n = atom_map[(160, "N")]

    for r in reports:
        # Check report metrics
        assert r.backbone_preserved is True
        assert r.ca_rmsd_angstrom < 1e-4, f"CA displacement {r.ca_rmsd_angstrom} exceeded 1e-4 A"
        assert r.backbone_rmsd_angstrom < 1e-4

        # Read back candidate PDB
        pdb_path = Path("results/structures/W159H") / f"{r.candidate_id}.pdb"
        assert pdb_path.exists()

        cand_coords = {}
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM  "):
                    rseq = int(line[22:26].strip())
                    aname = line[12:16].strip()
                    xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    cand_coords[(rseq, aname)] = xyz

        # Check CA position matches exactly
        cand_ca = cand_coords[(159, "CA")]
        assert np.linalg.norm(cand_ca - wt_ca) < 1e-3, "CA moved in PDB file"

        # Check peptide bond to Gly158: distance C(158) - N(159)
        d_prev = np.linalg.norm(cand_coords[(159, "N")] - gly158_c)
        assert abs(d_prev - 1.33) < 0.05, f"Gly158-His159 peptide bond disrupted: {d_prev:.3f} A"

        # Check peptide bond to Ser160: distance C(159) - N(160)
        d_next = np.linalg.norm(cand_coords[(159, "C")] - ser160_n)
        assert abs(d_next - 1.33) < 0.05, f"His159-Ser160 peptide bond disrupted: {d_next:.3f} A"


def test_charge_neutrality(builder):
    """Verifies that all candidates are formal neutrals and report AMBER table partial charge sums."""
    reports = builder.build_w159h_candidates(
        output_dir="results/structures/W159H",
        minimize_structures=False,
    )
    for r in reports:
        assert r.net_formal_charge == 0
        if r.tautomer == "HID":
            assert abs(r.sum_partial_charges - 0.1750) < 1e-3
        elif r.tautomer == "HIE":
            assert abs(r.sum_partial_charges - 0.1358) < 1e-3


def test_validation_reports_exist_and_conform():
    """Verifies that JSON validation reports exist for all 12 candidates and match schema."""
    summary_path = Path("results/structures/W159H/W159H_validation_summary.json")
    assert summary_path.exists()

    with open(summary_path, "r") as f:
        summary = json.load(f)

    assert len(summary) == 12, f"Expected 12 candidates in summary, found {len(summary)}"

    required_keys = [
        "candidate_id", "mutation", "tautomer", "rotamer_name", "atom_completeness",
        "total_atoms", "net_formal_charge", "sum_partial_charges", "backbone_preserved",
        "ca_rmsd_angstrom", "backbone_rmsd_angstrom", "min_scaffold_distance_angstrom",
        "validation_verdict"
    ]
    for cand in summary:
        for k in required_keys:
            assert k in cand, f"Candidate {cand.get('candidate_id')} missing key '{k}'"
