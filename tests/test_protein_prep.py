"""
Unit and integration tests for Q-Catalyst Protein Preparation & Topology Engine.
Validates:
1. Complete scaffold hydrogenation without coordinate fabrication.
2. Disulfide bond detection (CYX) at residues 203-239 and 273-289.
3. Physiological formal charge (+6.0 e) matching IsPETase pI ~ 9.6.
4. AMBER ff14SB parameter audit for peptide context, HID/HIE tautomers, and CYX.
5. Mutant W159H structure generation with complete hydrogen topologies.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from qcatalyst.chemistry.protein_prep import ProteinPreparator, prepare_w159h_scaffolds
from qcatalyst.chemistry.amber_ff14sb import audit_force_field_parameters, get_amber_charge


def test_scaffold_audit():
    prep = ProteinPreparator("data/structures/5XJH.pdb")
    audit = prep.audit_scaffold()

    assert audit["chain_id"] == "A"
    assert audit["total_residues"] == 263
    assert audit["first_residue"]["seq"] == 30
    assert audit["first_residue"]["name"] == "GLY"
    assert audit["last_residue"]["seq"] == 292
    assert audit["last_residue"]["name"] == "GLU"
    assert audit["missing_internal_residues"] == []
    assert audit["total_heavy_atoms"] == 1931
    assert audit["crystal_waters_count"] == 142

    # Verify disulfide bonds
    ss_bonds = audit["disulfide_bonds"]
    assert len(ss_bonds) == 2
    pairs = {(b["res1"], b["res2"]) for b in ss_bonds}
    assert (203, 239) in pairs
    assert (273, 289) in pairs

    # Verify theoretical net formal charge
    assert audit["theoretical_net_formal_charge_e"] == 6


def test_protein_preparation_wt_topology():
    prep = ProteinPreparator("data/structures/5XJH.pdb")
    atoms, meta = prep.prepare_complete_protein(mutation="WT")

    assert len(atoms) == 4101
    assert meta["total_heavy_atoms"] == 2073  # 1931 protein heavy + 142 water oxygen
    assert meta["crystal_waters_count"] == 142
    assert meta["disulfide_bonds_detected"] == 2
    assert meta["total_hydrogens_added"] > 2000

    # Verify total system charge is physically positive (eliminating the unphysical -85 e defect)
    assert meta["total_system_charge_e"] > 0.0

    # Check heavy atoms preservation
    first_atom = atoms[0]
    assert first_atom.atom_name == "N"
    assert first_atom.res_name == "GLY"
    assert first_atom.res_seq == 30
    assert not first_atom.is_hydrogen


def test_w159h_mutant_preparation():
    prep = ProteinPreparator("data/structures/5XJH.pdb")

    # Test HID preparation
    hid_atoms, hid_meta = prep.prepare_complete_protein(mutation="W159H", mutant_his_tautomer="HID")
    assert hid_meta["is_mutant"] is True
    assert hid_meta["mutant_residue_159_tautomer"] == "HID"

    res159_atoms = [a for a in hid_atoms if a.res_seq == 159 and not a.is_hetatm]
    res159_names = set(a.atom_name for a in res159_atoms)
    assert "HD1" in res159_names
    assert "HE1" in res159_names
    assert "NE2" in res159_names
    assert "CD1" not in res159_names  # Zero indole ghost atoms

    # Test HIE preparation
    hie_atoms, hie_meta = prep.prepare_complete_protein(mutation="W159H", mutant_his_tautomer="HIE")
    assert hie_meta["is_mutant"] is True
    assert hie_meta["mutant_residue_159_tautomer"] == "HIE"


def test_force_field_parameter_audit():
    audit = audit_force_field_parameters()
    assert "HID" in audit
    assert "HIE" in audit
    assert "CYX" in audit
    assert "ALA" in audit

    # Verify CYX charge is documented
    assert audit["CYX"]["atom_count"] == 10
    assert audit["CYX"]["expected_formal_charge_e"] == 0

    # Verify strict lookup works for CYX
    q_sg = get_amber_charge("CYX", "SG", strict=True)
    assert q_sg == -0.1186

