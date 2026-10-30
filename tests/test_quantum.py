"""
Unit tests validating the quantum active space (4e, 4o) solver and CASCI-VQE parity.
"""
import pytest
from src.qcatalyst.quantum.vqe_solver import QuantumVQESolver


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
