#!/usr/bin/env python3
"""Milestone 2 System Preparation CLI Script.

Prepares and validates molecular systems for:
1. WT IsPETase (5XJH)
2. FAST-PETase (7SH6)
3. Engineered Q-Catalyst Candidate (VAR-ENG-004)

Usage:
    python pet_degradation_lab/scripts/prepare_systems.py [--holo] [--padding 1.0] [--salt 0.15]
"""

import argparse
import sys
from pathlib import Path

# Add parent of pet_degradation_lab root to pythonpath
lab_root = Path(__file__).resolve().parent.parent
if str(lab_root.parent) not in sys.path:
    sys.path.insert(0, str(lab_root.parent))
if str(lab_root) not in sys.path:
    sys.path.insert(0, str(lab_root))

from pet_degradation_lab.preparation.sequence_mapping.mapping_validator import SequenceStructureValidator
from pet_degradation_lab.preparation.forcefield.parameter_validator import ParameterValidator
from pet_degradation_lab.preparation.pet_substrate.substrate_builder import PETSubstrateBuilder
from pet_degradation_lab.preparation.protein_preparer import ProteinPreparer
from pet_degradation_lab.preparation.solvated_system.system_builder import SolvatedSystemBuilder


def main():
    parser = argparse.ArgumentParser(description="Prepare validated molecular systems for PETase MD.")
    parser.add_argument("--holo", action="store_true", default=True, help="Include docked BHET substrate in binding cleft")
    parser.add_argument("--padding", type=float, default=1.0, help="Solvent box padding in nanometers (default: 1.0 nm)")
    parser.add_argument("--salt", type=float, default=0.15, help="NaCl ionic strength in Molar (default: 0.15 M)")
    args = parser.parse_args()

    print("======================================================================")
    print("   PET DEGRADATION LAB — MILESTONE 2 SYSTEM PREPARATION PIPELINE")
    print("======================================================================")

    # 1. Parameter Validation
    print("\n[Step 1/5] Validating GAFF2/RESP Ligand Parameters & ForceField Compatibility...")
    param_val = ParameterValidator()
    param_report = param_val.validate_parameters()
    print(f"  * Residues validated: {param_report.residues_validated}")
    print(f"  * Net charges: {param_report.net_charges} (Neutrality: {param_report.charge_neutrality_passed})")
    print(f"  * 1-4 Scaling factors: Coulomb={param_report.scale_14_coulomb:.4f}, LJ={param_report.scale_14_lj:.4f}")
    print(f"  * OpenMM Load Status: {param_report.openmm_load_passed}")
    if not param_report.all_checks_passed:
        print("ERROR: Forcefield parameter validation failed!")
        sys.exit(1)
    print("  -> Forcefield parameters verified successfully.")

    # Paths
    snap_base = lab_root / "data" / "project_snapshots" / "snap_baseline_v1" / "raw" / "data" / "structures"
    systems_out_base = lab_root / "data" / "systems"

    systems = [
        ("wt_5xjh", snap_base / "5XJH.pdb"),
        ("fast_7sh6", snap_base / "7SH6.pdb"),
        ("var_eng_004", snap_base / "variants" / "VAR-ENG-004.pdb"),
    ]

    # 2. Structural & Mapping Validation
    print("\n[Step 2/5] Validating Sequence-to-Structure, Catalytic Triad & Disulfides...")
    struct_val = SequenceStructureValidator()
    ref_5xjh = snap_base / "5XJH.pdb"

    for sys_id, pdb_file in systems:
        res = struct_val.validate_system(pdb_file, sys_id, reference_5xjh_path=ref_5xjh)
        triad = res.catalytic_triad
        print(f"  [{sys_id.upper()}] Standard residues: {res.num_standard_residues} ({res.start_residue[0]}{res.start_residue[1]} -> {res.end_residue[0]}{res.end_residue[1]})")
        print(f"     Catalytic Triad: Ser160-His237={triad.dist_ser_his_ne2_A:.2f}A, His237-Asp206={triad.dist_his_nd1_asp_od2_A:.2f}A (Intact: {triad.is_geometry_intact})")
        print(f"     Disulfides: {[d.dist_sg_sg_A for d in res.disulfides]} A (Intact: {all(d.is_intact for d in res.disulfides)})")
        print(f"     Histidines: {res.histidines}")
        if res.mutations_from_5xjh:
            print(f"     Mutations vs 5XJH: {res.mutations_from_5xjh}")
        if not res.passed_validation:
            print(f"ERROR: Structure validation failed for {sys_id}")
            sys.exit(1)
    print("  -> All three structures passed sequence and structural integrity checks.")

    # 3. Protein Cleaning & Protonation
    print("\n[Step 3/5] Preparing Proteins (Capping, Heavy Atoms, Protonation at pH 7.5)...")
    prot_prep = ProteinPreparer(pH=7.5)
    prepared_prot_dir = systems_out_base / "prepared_proteins"

    prepared_pdbs = {}
    for sys_id, pdb_file in systems:
        out_prot_pdb = prepared_prot_dir / f"{sys_id}_prepared.pdb"
        prep_res = prot_prep.prepare_protein(pdb_file, sys_id, out_prot_pdb)
        prepared_pdbs[sys_id] = out_prot_pdb
        print(f"  [{sys_id.upper()}] {prep_res.summary}")

    # 4. PET Substrate Modeling & Active Site Alignment
    print("\n[Step 4/5] Building & Aligning PET Substrate (BHET) into Active Site Cleft...")
    sub_builder = PETSubstrateBuilder()
    ligand_pdbs = {}

    if args.holo:
        ligand_dir = systems_out_base / "ligands"
        for sys_id in ["wt_5xjh", "fast_7sh6", "var_eng_004"]:
            out_lig_pdb = ligand_dir / f"{sys_id}_bhet.pdb"
            mol, metrics = sub_builder.align_bhet_to_active_site(
                prepared_pdbs[sys_id],
                chain_id="A",
                output_pdb_path=out_lig_pdb,
            )
            ligand_pdbs[sys_id] = out_lig_pdb
            print(f"  [{sys_id.upper()}] {metrics.summary} (Valid: {metrics.is_michaelis_geometry_valid})")

    # 5. Explicit-Solvent Assembly, Neutralization & OpenMM Serialization
    print(f"\n[Step 5/5] Building Solvated Explicit Systems (TIP3P, {args.salt*1000:.0f} mM NaCl, {args.padding*10:.0f} A Buffer)...")
    sys_builder = SolvatedSystemBuilder(
        padding_nm=args.padding,
        ionic_strength_M=args.salt,
    )

    for sys_id in ["wt_5xjh", "fast_7sh6", "var_eng_004"]:
        sys_dir = systems_out_base / sys_id
        lig_path = ligand_pdbs.get(sys_id) if args.holo else None
        metrics = sys_builder.build_system(
            protein_pdb_path=prepared_pdbs[sys_id],
            system_id=sys_id,
            output_dir=sys_dir,
            ligand_pdb_path=lig_path,
            max_minimization_steps=100,
        )
        print(f"\n  [{sys_id.upper()} COMPLETE]")
        print(f"    * Solute atoms: {metrics.num_solute_atoms}")
        print(f"    * Water molecules: {metrics.num_waters}")
        print(f"    * Counterions: {metrics.num_na_ions} Na+, {metrics.num_cl_ions} Cl-")
        print(f"    * Total system particles: {metrics.total_particles}")
        print(f"    * Net charge: {metrics.net_charge:.2f} e")
        print(f"    * Box vectors: {metrics.box_dimensions_nm[0]:.2f} x {metrics.box_dimensions_nm[1]:.2f} x {metrics.box_dimensions_nm[2]:.2f} nm")
        print(f"    * Potential energy: {metrics.potential_energy_kj_mol:.1f} kJ/mol")
        print(f"    * Coordinates: {metrics.pdb_path}")
        print(f"    * Serialized OpenMM System: {metrics.system_xml_path}")
        print(f"    * Serialized OpenMM Integrator: {metrics.integrator_xml_path}")

    print("\n======================================================================")
    print("   MILESTONE 2 PIPELINE COMPLETED SUCCESSFULLY!")
    print("======================================================================")


if __name__ == "__main__":
    main()
