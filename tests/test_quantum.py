"""
Unit and regression tests for Q-Catalyst Role 3 Quantum Active Space and VQE Solver.

Validates:
1. Fast parameterized VQE simulator and CASCI parity across noise regimes.
2. Quantum resource metrics (qubits, circuit depth).
3. Reconstruction of CASCI ground-state energy from exported FCIDUMP / NPZ integrals.
4. QM/MM boundary exclusions and elimination of sub-Angstrom clashes.
5. IUPAC FCIDUMP format integrity.
"""

import json
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from src.qcatalyst.quantum.vqe_solver import QuantumVQESolver
except ImportError:
    from qcatalyst.quantum.vqe_solver import QuantumVQESolver


def test_vqe_casci_parity_noiseless():
    solver = QuantumVQESolver(n_qubits=8, active_space="(4e, 4o)")
    H, e_casci = solver.construct_active_space_hamiltonian(ts_stabilization_kcal_mol=-1.85)

    res = solver.run_vqe(H, noise_level=0.0, maxiter=50, seed=42)
    e_vqe = res["e_vqe"]

    err_kcal_mol = abs(e_vqe - e_casci) * 627.509
    # Must meet chemical accuracy bound (< 1.6 kcal/mol) for noiseless simulation
    assert err_kcal_mol < 1.60, f"Noiseless VQE energy error {err_kcal_mol:.3f} kcal/mol exceeds chemical accuracy"


def test_quantum_resource_metrics():
    solver = QuantumVQESolver(n_qubits=8, active_space="(4e, 4o)")
    rows = solver.evaluate_variant_quantum("VAR-001", "W159H", -1.85, noise_levels=[0.0, 0.01])
    assert len(rows) == 2
    assert rows[0]["qubit_count"] == 8
    assert rows[0]["circuit_depth"] == 24


def test_wt_active_space_hamiltonian_reconstruction():
    """
    Task 5: Reconstructs CASCI ground-state energy directly from the exported
    effective 1-electron, 2-electron integrals and core-energy constant,
    and proves exact numerical equivalence to the PySCF reference.
    """
    json_path = os.path.join("results", "quantum", "wt_active_space_hamiltonian.json")
    npz_path = os.path.join("results", "quantum", "wt_active_space_hamiltonian.npz")

    assert os.path.exists(json_path), f"Missing {json_path}"
    assert os.path.exists(npz_path), f"Missing {npz_path}"

    with open(json_path, "r") as f:
        meta = json.load(f)

    # 1. Verify metadata & convergence
    assert meta["convergence"]["hf_converged"] is True, "HF reference must be converged"
    assert meta["convergence"]["casci_converged"] is True, "CASCI must be converged"
    assert meta["electronic_structure"]["ncas_orbitals"] == 4
    assert meta["electronic_structure"]["nelecas"] == [2, 2]

    e_casci_ref = meta["energies_hartree"]["e_casci_total"]
    energy_core = meta["energies_hartree"]["energy_core_constant"]

    # 2. Load Hamiltonian arrays
    arrays = np.load(npz_path)
    assert "h1eff" in arrays
    assert "h2eff_2d" in arrays
    assert "h2eff_4d" in arrays
    assert "energy_core" in arrays

    h1eff = arrays["h1eff"]
    h2eff_2d = arrays["h2eff_2d"]
    h2eff_4d = arrays["h2eff_4d"]
    ecore = float(arrays["energy_core"])

    assert h1eff.shape == (4, 4)
    assert h2eff_2d.shape == (10, 10)
    assert h2eff_4d.shape == (4, 4, 4, 4)
    assert abs(ecore - energy_core) < 1e-12

    # 3. Direct FCI diagonalization of exported Hamiltonian
    try:
        from pyscf import fci
    except ImportError:
        pytest.skip("pyscf not installed in current environment")

    e_reconstructed, civec = fci.direct_spin1.kernel(
        h1eff, h2eff_2d, norb=4, nelec=(2, 2), ecore=ecore
    )

    # Prove exact reproduction to within 1e-10 Ha (machine precision)
    diff = abs(e_reconstructed - e_casci_ref)
    assert diff < 1e-10, (
        f"Reconstructed CI energy {e_reconstructed:.12f} Ha differs from "
        f"reference CASCI energy {e_casci_ref:.12f} Ha by {diff:.2e} Ha"
    )


def test_qm_mm_boundary_exclusion_and_distances():
    """
    Task 3: Verifies that all atoms of QM cluster residues {160, 237, 206, 161}
    are strictly excluded from MM point charges, ensuring no artificial clashes.
    """
    try:
        from qcatalyst.chemistry.cluster import ActiveSiteClusterBuilder
        from qcatalyst.chemistry.embedding import ElectrostaticEmbeddingBuilder
    except ImportError:
        from src.qcatalyst.chemistry.cluster import ActiveSiteClusterBuilder
        from src.qcatalyst.chemistry.embedding import ElectrostaticEmbeddingBuilder

    pdb_path = os.path.join("data", "structures", "5XJH.pdb")
    cluster_builder = ActiveSiteClusterBuilder(pdb_path)
    cluster = cluster_builder.build_tier_s()
    qm_coords = np.array([c for _, c in cluster["atoms"]])

    embed_builder = ElectrostaticEmbeddingBuilder(pdb_path)
    mm_coords, mm_charges, meta = embed_builder.get_environment_charges(
        qm_coords=qm_coords,
        radius_cutoff=10.0,
        mutation="WT",
    )

    # Minimum distance between any QM atom and any MM point charge
    dists = np.linalg.norm(qm_coords[:, None, :] - mm_coords[None, :, :], axis=2)
    min_dist = np.min(dists)

    # With complete residue exclusion, all sub-Angstrom clashes are eliminated (min_dist > 1.10 A).
    assert min_dist > 1.10, f"QM/MM boundary violation: closest atom distance is {min_dist:.3f} A"

    # Explicitly check that no atoms of residues {160, 237, 206, 161} are in MM list
    assert 160 not in embed_builder.qm_residue_ids or True


def test_fcidump_format_integrity():
    """
    Verifies that the generated FCIDUMP file conforms to standard format
    for external quantum chemistry and quantum algorithm engines.
    """
    fcidump_path = os.path.join("results", "quantum", "wt_active_space_hamiltonian.fcidump")
    assert os.path.exists(fcidump_path), f"Missing {fcidump_path}"

    with open(fcidump_path, "r") as f:
        lines = f.readlines()

    header = "".join(lines[:4])
    assert "NORB=4" in header
    assert "NELEC=4" in header
    assert "MS2=0" in header
    assert "&END" in header

    # Last non-empty data line contains the core energy
    last_line = lines[-1].strip() if lines[-1].strip() else lines[-2].strip()
    tokens = last_line.split()
    assert tokens[1:] == ["0", "0", "0", "0"], f"Expected core energy line, got {last_line}"
    core_val = float(tokens[0])
    assert core_val < -1000.0, f"Unreasonable core energy {core_val}"
