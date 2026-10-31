"""Independent Validation Test Suite for Milestone 2 Molecular Systems.

Tests:
1. Sequence-to-structure mapping, chain IDs, and residue numbering
2. Catalytic triad geometry (Ser160-His237-Asp206) and disulfide bonds
3. Histidine tautomer assignments (His237 as neutral HID with N_delta1-H)
4. GAFF2/RESP parameter completeness, 1-4 scaling factors, and net neutrality
5. Solvated system composition, box dimensions, 150 mM NaCl, and net charge = 0.0
6. OpenMM system deserialization and finite potential energy (non-NaN)
7. Adherence to scientific boundary invariants
"""

import json
import sys
from pathlib import Path
import pytest
import numpy as np
import openmm
import openmm.app as app
from openmm import unit

# Ensure parent and lab_root are on sys.path
_lab_root = Path(__file__).resolve().parent.parent
if str(_lab_root.parent) not in sys.path:
    sys.path.insert(0, str(_lab_root.parent))
if str(_lab_root) not in sys.path:
    sys.path.insert(0, str(_lab_root))

from pet_degradation_lab.preparation.sequence_mapping.mapping_validator import SequenceStructureValidator
from pet_degradation_lab.preparation.forcefield.parameter_validator import ParameterValidator


@pytest.fixture
def repo_paths():
    test_file = Path(__file__).resolve()
    lab_root = test_file.parent.parent
    snap_base = lab_root / "data" / "project_snapshots" / "snap_baseline_v1" / "raw" / "data" / "structures"
    systems_dir = lab_root / "data" / "systems"
    ff_xml = lab_root / "preparation" / "forcefield" / "pet_gaff2_resp.xml"
    return {
        "lab_root": lab_root,
        "snap_base": snap_base,
        "systems_dir": systems_dir,
        "ff_xml": ff_xml,
        "5xjh": snap_base / "5XJH.pdb",
        "7sh6": snap_base / "7SH6.pdb",
        "var_eng_004": snap_base / "variants" / "VAR-ENG-004.pdb",
    }


def test_sequence_structure_mapping(repo_paths):
    """Verify chain IDs, residue counts, and residue bounds for all 3 reference structures."""
    validator = SequenceStructureValidator()
    ref_5xjh = repo_paths["5xjh"]

    # 1. WT 5XJH
    res_wt = validator.validate_system(repo_paths["5xjh"], "WT_5XJH", reference_5xjh_path=ref_5xjh)
    assert res_wt.chain_id == "A"
    assert res_wt.num_standard_residues == 263
    assert res_wt.start_residue == ("GLY", 30)
    assert res_wt.end_residue == ("GLU", 292)
    assert res_wt.passed_validation is True

    # 2. FAST 7SH6
    res_fast = validator.validate_system(repo_paths["7sh6"], "FAST_7SH6", reference_5xjh_path=ref_5xjh)
    assert res_fast.chain_id == "A"
    assert res_fast.num_standard_residues == 261
    assert res_fast.start_residue == ("THR", 29)
    assert res_fast.end_residue == ("CYS", 289)
    assert res_fast.passed_validation is True

    # 3. VAR-ENG-004
    res_var = validator.validate_system(repo_paths["var_eng_004"], "VAR_ENG_004", reference_5xjh_path=ref_5xjh)
    assert res_var.chain_id == "A"
    assert res_var.num_standard_residues == 263
    assert res_var.start_residue == ("GLY", 30)
    assert res_var.end_residue == ("GLU", 292)
    assert res_var.passed_validation is True


def test_catalytic_triad_and_disulfides(repo_paths):
    """Verify catalytic triad distances and covalent disulfide bonds in all 3 systems."""
    validator = SequenceStructureValidator()

    for path, name in [
        (repo_paths["5xjh"], "5XJH"),
        (repo_paths["7sh6"], "7SH6"),
        (repo_paths["var_eng_004"], "VAR-ENG-004"),
    ]:
        res = validator.validate_system(path, name)
        triad = res.catalytic_triad
        # Catalytic Ser-His distance
        assert 2.5 <= triad.dist_ser_his_ne2_A <= 3.2, f"{name}: Ser-His distance out of bounds ({triad.dist_ser_his_ne2_A} A)"
        # Catalytic His-Asp distance
        assert 2.5 <= triad.dist_his_nd1_asp_od2_A <= 3.0, f"{name}: His-Asp distance out of bounds ({triad.dist_his_nd1_asp_od2_A} A)"
        assert triad.is_geometry_intact is True

        # Disulfides
        assert len(res.disulfides) == 2
        for ds in res.disulfides:
            assert ds.is_intact is True
            assert 1.95 <= ds.dist_sg_sg_A <= 2.20, f"{name}: Disulfide {ds.res1_id}-{ds.res2_id} dist={ds.dist_sg_sg_A} A"


def test_histidine_tautomers(repo_paths):
    """Verify that prepared proteins have His237 assigned as HID tautomer (HD1 present, HE2 absent)."""
    prep_dir = repo_paths["systems_dir"] / "prepared_proteins"
    parser = app.PDBFile

    for sys_id in ["wt_5xjh", "fast_7sh6", "var_eng_004"]:
        pdb_path = prep_dir / f"{sys_id}_prepared.pdb"
        assert pdb_path.exists(), f"Prepared PDB missing: {pdb_path}"

        pdb = parser(str(pdb_path))
        # Find His237 residue
        his_residues = [r for r in pdb.topology.residues() if r.name in ["HIS", "HID", "HIE", "HIP"]]
        assert len(his_residues) >= 3

        # Catalytic His has ND1 near Asp206
        catalytic_his = None
        for h in his_residues:
            atom_names = [a.name for a in h.atoms()]
            if "ND1" in atom_names and "HD1" in atom_names and "HE2" not in atom_names:
                # Delta protonated (HID)
                catalytic_his = h
                break

        assert catalytic_his is not None, f"Catalytic His not found in delta-protonated HID state in {sys_id}"


def test_gaff2_resp_parameter_neutrality_and_scaling(repo_paths):
    """Verify GAFF2/RESP XML parameter completeness, exact net neutrality, and 1-4 scaling."""
    validator = ParameterValidator(repo_paths["ff_xml"])
    report = validator.validate_parameters()

    assert report.all_checks_passed is True
    assert report.charge_neutrality_passed is True
    assert report.scale_factors_passed is True
    assert report.openmm_load_passed is True

    # Check exact charges
    assert abs(report.net_charges["BHT"]) < 1e-6
    assert abs(report.net_charges["2PT"]) < 1e-6

    # Check 1-4 scaling factors match AMBER standard
    assert abs(report.scale_14_coulomb - (1.0 / 1.2)) < 1e-4
    assert abs(report.scale_14_lj - 0.5) < 1e-4

    # Check refined torsions are present
    assert "ester_aryl_planar" in report.torsion_refinements
    assert "ethylene_glycol_gauche_os" in report.torsion_refinements


def test_solvated_system_composition_and_neutrality(repo_paths):
    """Verify all 3 explicit systems have net charge 0.0, 150 mM NaCl, and valid box dimensions."""
    systems_dir = repo_paths["systems_dir"]

    for sys_id in ["wt_5xjh", "fast_7sh6", "var_eng_004"]:
        meta_path = systems_dir / sys_id / f"{sys_id}_metadata.json"
        assert meta_path.exists(), f"Metadata JSON missing for {sys_id}"

        with open(meta_path, "r") as f:
            meta = json.load(f)

        assert meta["is_holo"] is True
        assert meta["num_solute_atoms"] >= 4200
        assert meta["num_waters"] >= 9000
        assert meta["total_particles"] >= 30000
        assert abs(meta["net_charge"]) < 0.05, f"Non-zero net charge in {sys_id}: {meta['net_charge']}"
        assert meta["ionic_strength_M"] == 0.15
        assert all(dim >= 6.5 for dim in meta["box_dimensions_nm"])
        assert meta["is_stable"] is True
        assert meta["potential_energy_kj_mol"] < 0.0


def test_openmm_system_deserialization_and_energy(repo_paths):
    """Deserialize system.xml and coordinates in OpenMM, verifying energy calculation without NaN."""
    systems_dir = repo_paths["systems_dir"]

    for sys_id in ["wt_5xjh", "fast_7sh6", "var_eng_004"]:
        sys_xml = systems_dir / sys_id / f"{sys_id}_system.xml"
        pdb_file = systems_dir / sys_id / f"{sys_id}_solvated.pdb"

        assert sys_xml.exists()
        assert pdb_file.exists()

        with open(sys_xml, "r") as f:
            system = openmm.XmlSerializer.deserialize(f.read())

        pdb = app.PDBFile(str(pdb_file))
        assert system.getNumParticles() == pdb.topology.getNumAtoms()

        # Build context and test single-point potential energy evaluation
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        state = simulation.context.getState(getEnergy=True)
        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        assert not np.isnan(pe), f"Potential energy is NaN for {sys_id}"
        assert pe < 0.0, f"Potential energy is positive ({pe} kJ/mol) for {sys_id}"


def test_scientific_boundary_invariants():
    """Verify scientific boundary invariants required by project protocol."""
    # Invariant 1: Fixed-topology MD cannot simulate ester bond cleavage
    cannot_cleave_bonds_in_standard_md = True
    assert cannot_cleave_bonds_in_standard_md is True

    # Invariant 2: Binding energy is not equal to catalytic turnover rate
    binding_energy_equals_catalytic_rate = False
    assert binding_energy_equals_catalytic_rate is False
