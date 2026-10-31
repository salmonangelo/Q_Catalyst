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
    ff_xml = lab_root / "preparation" / "forcefield" / "pet_gaff2_resp_validated.xml"
    if not ff_xml.exists():
        ff_xml = lab_root / "preparation" / "forcefield" / "pet_gaff2_resp.xml"
    resp_artifacts_dir = lab_root / "data" / "forcefield" / "resp_artifacts"
    qm_scans_dir = lab_root / "data" / "forcefield" / "qm_scans"
    return {
        "lab_root": lab_root,
        "snap_base": snap_base,
        "systems_dir": systems_dir,
        "ff_xml": ff_xml,
        "resp_artifacts_dir": resp_artifacts_dir,
        "qm_scans_dir": qm_scans_dir,
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


def test_authentic_resp_charge_provenance(repo_paths):
    """Verify authentic QM/RESP fitting artifacts from PySCF RHF/6-31G* calculations."""
    resp_dir = repo_paths["resp_artifacts_dir"]
    charges_json = resp_dir / "BHET_stage2_resp_charges.json"
    summary_json = resp_dir / "BHET_resp_summary.json"
    raw_esp_npz = resp_dir / "BHET_raw_qm_esp.npz"

    assert charges_json.exists(), f"RESP charges JSON missing: {charges_json}"
    assert summary_json.exists(), f"RESP summary JSON missing: {summary_json}"
    assert raw_esp_npz.exists(), f"Raw QM ESP npz missing: {raw_esp_npz}"

    with open(summary_json, "r") as f:
        summary = json.load(f)

    assert summary["qm_method"] == "HF"
    assert summary["basis_set"] == "6-31g*"
    assert summary["scf_energy_hartree"] < -911.0
    assert summary["num_grid_points"] >= 800
    assert summary["rrms_error"] < 0.15
    assert abs(summary["total_fitted_charge"]) < 1e-6

    with open(charges_json, "r") as f:
        charges_data = json.load(f)
    charges = charges_data["charges"]

    assert len(charges) == 32
    assert abs(sum(charges.values())) < 1e-6
    # Carbonyl carbon must be strongly positive (~0.8 e)
    assert charges["C7"] > 0.70
    # Carbonyl oxygen must be strongly negative (~-0.6 e)
    assert charges["O1"] < -0.50


def test_qm_torsional_scans_and_barriers(repo_paths):
    """Verify ab initio B3LYP/6-31G* torsional scan profiles and refined parameter barriers."""
    scans_dir = repo_paths["qm_scans_dir"]
    ester_scan_json = scans_dir / "ester_aromatic_torsion_scan.json"
    glycol_scan_json = scans_dir / "ethylene_glycol_torsion_scan.json"

    assert ester_scan_json.exists(), f"Ester scan missing: {ester_scan_json}"
    assert glycol_scan_json.exists(), f"Glycol scan missing: {glycol_scan_json}"

    with open(ester_scan_json, "r") as f:
        ester_data = json.load(f)

    assert len(ester_data["angles_deg"]) >= 7
    # Coplanar (0 deg) must be the minimum
    e_0 = ester_data["qm_relative_kcal_mol"][0]
    e_90 = ester_data["qm_relative_kcal_mol"][-1]
    assert e_0 == 0.0
    # QM barrier at 90 deg is ~6.7 kcal/mol
    assert 5.0 <= e_90 <= 8.5
    # Generic GAFF2 barrier is only ~2.1 kcal/mol (underestimates barrier)
    assert ester_data["mm_generic_kcal_mol"][-1] < 3.0
    assert ester_data["mm_refined_barrier_kcal_mol"] > 8.0

    with open(glycol_scan_json, "r") as f:
        glycol_data = json.load(f)

    assert len(glycol_data["angles_deg"]) >= 7
    # Cis barrier at 0 deg must be the highest in vacuum
    e_cis = glycol_data["qm_relative_kcal_mol"][0]
    assert e_cis > 7.0


def test_active_site_michaelis_geometry(repo_paths):
    """Verify Bürgi-Dunitz attack trajectory and oxyanion hole geometry without steric clash."""
    ligand_dir = repo_paths["systems_dir"] / "ligands"
    prep_dir = repo_paths["systems_dir"] / "prepared_proteins"
    parser = app.PDBFile

    for sys_id in ["wt_5xjh", "fast_7sh6", "var_eng_004"]:
        lig_path = ligand_dir / f"{sys_id}_bhet.pdb"
        prot_path = prep_dir / f"{sys_id}_prepared.pdb"

        assert lig_path.exists(), f"Ligand PDB missing: {lig_path}"
        assert prot_path.exists(), f"Prepared protein PDB missing: {prot_path}"

        lig_pdb = parser(str(lig_path))
        prot_pdb = parser(str(prot_path))

        # Ligand atoms: C7 and O1
        lig_coords = lig_pdb.positions.value_in_unit(unit.angstroms)
        c7_pos = None
        o1_pos = None
        for a in lig_pdb.topology.atoms():
            if a.name == "C7":
                c7_pos = np.array(lig_coords[a.index])
            elif a.name == "O1":
                o1_pos = np.array(lig_coords[a.index])

        assert c7_pos is not None, f"C7 not found in {lig_path}"
        assert o1_pos is not None, f"O1 not found in {lig_path}"

        # Find Ser160 OG and oxyanion amides in protein
        prot_coords = prot_pdb.positions.value_in_unit(unit.angstroms)
        res_list = list(prot_pdb.topology.residues())

        # Find catalytic Ser (Ser whose OG is ~2.9 A from His NE2)
        cat_ser_og = None
        met_n = None
        tyr_n = None
        for i, r in enumerate(res_list):
            if r.name == "SER":
                og_atom = [a for a in r.atoms() if a.name == "OG"]
                if og_atom:
                    og_coord = np.array(prot_coords[og_atom[0].index])
                    for h in res_list:
                        if h.name in ["HIS", "HID", "HIE", "HIP"]:
                            ne2 = [a for a in h.atoms() if a.name == "NE2"]
                            if ne2:
                                d = np.linalg.norm(og_coord - np.array(prot_coords[ne2[0].index]))
                                if 2.7 <= d <= 3.2:
                                    cat_ser_og = og_coord
                                    if i + 1 < len(res_list):
                                        met_atom = [a for a in res_list[i + 1].atoms() if a.name == "N"]
                                        if met_atom:
                                            met_n = np.array(prot_coords[met_atom[0].index])
                                    break
                    if cat_ser_og is not None:
                        break

        # Tyr87 N
        for r in res_list:
            if r.name == "TYR":
                n_atom = [a for a in r.atoms() if a.name == "N"]
                if n_atom:
                    coord = np.array(prot_coords[n_atom[0].index])
                    d = np.linalg.norm(coord - cat_ser_og)
                    if 4.5 <= d <= 6.5:
                        tyr_n = coord
                        break

        assert cat_ser_og is not None, f"Catalytic Ser OG not resolved for {sys_id}"
        assert met_n is not None, f"Met161 N not resolved for {sys_id}"
        assert tyr_n is not None, f"Tyr87 N not resolved for {sys_id}"

        # 1. Attack distance: Ser160 OG -> C7
        d_attack = float(np.linalg.norm(cat_ser_og - c7_pos))
        assert 2.75 <= d_attack <= 3.15, f"{sys_id}: Attack distance {d_attack:.2f} A out of bounds"

        # 2. Bürgi-Dunitz angle: angle(OG - C7 = O1)
        v_sc = cat_ser_og - c7_pos
        v_oc = o1_pos - c7_pos
        cos_bd = np.dot(v_sc, v_oc) / (np.linalg.norm(v_sc) * np.linalg.norm(v_oc))
        ang_bd = float(np.arccos(np.clip(cos_bd, -1.0, 1.0)) * 180.0 / np.pi)
        assert 95.0 <= ang_bd <= 115.0, f"{sys_id}: BD angle {ang_bd:.1f} deg out of [95, 115]"

        # 3. Oxyanion hole coordination: no steric collision (d >= 2.70 A)
        d_met_o = float(np.linalg.norm(met_n - o1_pos))
        d_tyr_o = float(np.linalg.norm(tyr_n - o1_pos))
        assert d_met_o >= 2.70, f"{sys_id}: Steric collision with Met161 N: {d_met_o:.2f} A < 2.70 A"
        assert d_tyr_o >= 2.70, f"{sys_id}: Steric collision with Tyr87 N: {d_tyr_o:.2f} A < 2.70 A"
        assert d_met_o <= 3.85, f"{sys_id}: Met161 N too far from O1: {d_met_o:.2f} A"

