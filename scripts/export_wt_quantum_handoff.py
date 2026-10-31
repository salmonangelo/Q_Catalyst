#!/usr/bin/env python3
"""
Exports the complete, verified classical reference active-space Hamiltonian
for Wild-Type PETase Tier S catalytic cluster with QM/MM electrostatic embedding.
Deliverable for Role 3 Quantum Lead (VQE reference).
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from qcatalyst.chemistry.cluster import ActiveSiteClusterBuilder
from qcatalyst.chemistry.embedding import ElectrostaticEmbeddingBuilder
from qcatalyst.chemistry.pyscf_engine import PySCFChemistryEngine


def export_wt_pilot():
    print("=" * 70)
    print("  Q-CATALYST: EXPORT WT ACTIVE-SPACE HAMILTONIAN FOR ROLE 3 (VQE)")
    print("=" * 70)

    # 1. Build Tier S Cluster
    pdb_path = "data/structures/5XJH.pdb"
    cluster_builder = ActiveSiteClusterBuilder(pdb_path)
    cluster = cluster_builder.build_tier_s()
    qm_coords = np.array([c for _, c in cluster["atoms"]])
    print(f"[Step 1] Built Tier S Cluster: {cluster['natoms']} atoms, net charge = {cluster['charge']}")

    # 2. Build Electrostatic Embedding with Confirmed Boundary Fix (all QM residues excluded)
    embed_builder = ElectrostaticEmbeddingBuilder(pdb_path)
    mm_coords, mm_charges, embed_meta = embed_builder.get_environment_charges(
        qm_coords=qm_coords,
        radius_cutoff=10.0,
        mutation="WT",
    )
    print(f"[Step 2] MM Embedding: {len(mm_charges)} point charges, sum(q) = {np.sum(mm_charges):.4f} e")

    # 3. Compute RHF and CASCI Reference with Convergence Validation & Integral Extraction
    engine = PySCFChemistryEngine(basis="def2-svp", conv_tol=1e-6)
    print("[Step 3] Running converged RHF and CASCI(4e, 4o) active space...")
    cas_res = engine.run_casci_reference(
        geometry_str=cluster["geometry_str"],
        ncas=4,
        nelecas=(2, 2),
        charge=cluster["charge"],
        spin=cluster["spin"],
        mm_coords=mm_coords,
        mm_charges=mm_charges,
    )
    print(f"  RHF Converged:   {cas_res['hf_converged']} | E_HF = {cas_res['e_hf_ref']:.8f} Ha")
    print(f"  CASCI Converged: {cas_res['casci_converged']} | E_CASCI = {cas_res['e_casci_total']:.8f} Ha")
    print(f"  Correlation:     {cas_res['correlation_energy_ha']:.8f} Ha")
    print(f"  Energy Core:     {cas_res['energy_core']:.8f} Ha")

    # 4. Export Artifacts for Role 3
    out_dir = "results/quantum"
    os.makedirs(out_dir, exist_ok=True)
    metadata = {
        "variant_id": "VAR_WT",
        "mutation": "WT",
        "pdb_structure": "5XJH (Chain A)",
        "qm_cluster_tier": "Tier S (42 atoms)",
        "mm_point_charges_count": len(mm_charges),
        "mm_net_charge_e": float(np.sum(mm_charges)),
        "boundary_exclusion_fix": "All atoms of residues {160, 237, 206, 161} excluded from MM point charges",
    }
    paths = engine.export_active_space_hamiltonian(
        casci_result=cas_res,
        output_dir=out_dir,
        prefix="wt_active_space_hamiltonian",
        metadata=metadata,
    )
    print(f"[Step 4] Export Complete!")
    for k, v in paths.items():
        print(f"  - {k.upper()}: {v}")

    return cas_res, paths


if __name__ == "__main__":
    export_wt_pilot()
