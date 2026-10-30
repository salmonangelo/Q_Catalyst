#!/usr/bin/env python3
"""WT PETase active-space VQE benchmark for Q-Catalyst Role 3."""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh

from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP

from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD


BASE = Path("results/quantum")
FCIDUMP_PATH = BASE / "wt_active_space_hamiltonian.fcidump"
METADATA_PATH = BASE / "wt_active_space_hamiltonian.json"
OUTPUT_PATH = BASE / "wt_vqe_result.json"

NUM_SPATIAL_ORBITALS = 4
NUM_PARTICLES = (2, 2)
EXACT_TOLERANCE = 1e-8


def main():
    print("=" * 65)
    print("Q-CATALYST | WT ACTIVE-SPACE VQE")
    print("=" * 65)

    if not FCIDUMP_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("WT Hamiltonian handoff files are missing.")

    metadata = json.loads(METADATA_PATH.read_text())
    reference_energy = metadata["energies_hartree"]["e_casci_total"]

    # 1. Read the exported Hamiltonian.
    data = FCIDump.from_file(str(FCIDUMP_PATH))

    if data.num_electrons != sum(NUM_PARTICLES):
        raise ValueError("FCIDUMP electron count does not match CAS(4e,4o).")

    if data.hij.shape != (NUM_SPATIAL_ORBITALS, NUM_SPATIAL_ORBITALS):
        raise ValueError("Unexpected one-electron integral dimensions.")

    # 2. Construct the electronic Hamiltonian.
    electronic_energy = ElectronicEnergy.from_raw_integrals(
        h1_a=data.hij,
        h2_aa=data.hijkl,
        validate=True,
        auto_index_order=True,
    )
    electronic_energy.constants["FCIDUMP core energy"] = data.constant_energy

    fermionic_op = electronic_energy.second_q_op()
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(fermionic_op)

    # Explicitly include the frozen-core / nuclear energy constant.
    # ElectronicEnergy.constants is not automatically included in second_q_op().
    core_constant = SparsePauliOp(
        "I" * qubit_op.num_qubits,
        coeffs=[data.constant_energy],
    )
    qubit_op = qubit_op + core_constant

    print(f"[1] Spatial orbitals: {NUM_SPATIAL_ORBITALS}")
    print(f"[2] Active electrons: {NUM_PARTICLES}")
    print(f"[3] Qubits: {qubit_op.num_qubits}")
    print(f"[4] Pauli terms: {len(qubit_op)}")

    # 3. Independently validate the fixed (2 alpha, 2 beta) sector.
    matrix = qubit_op.to_matrix()

    alpha_modes = range(4)
    beta_modes = range(4, 8)

    sector_indices = [
        state for state in range(2 ** qubit_op.num_qubits)
        if sum((state >> i) & 1 for i in alpha_modes) == 2
        and sum((state >> i) & 1 for i in beta_modes) == 2
    ]

    if len(sector_indices) != 36:
        raise RuntimeError("Unexpected CAS(4e,4o) sector dimension.")

    sector_matrix = matrix[np.ix_(sector_indices, sector_indices)]
    exact_energy = float(eigvalsh(sector_matrix)[0].real)
    exact_error = abs(exact_energy - reference_energy)

    print(f"\n[5] PySCF CASCI reference: {reference_energy:.12f} Ha")
    print(f"[6] Qiskit exact reference: {exact_energy:.12f} Ha")
    print(f"[7] Exact reference difference: {exact_error:.3e} Ha")

    if exact_error > EXACT_TOLERANCE:
        raise RuntimeError(
            "Hamiltonian validation failed. VQE optimization aborted."
        )

    print("\n[PASS] Exact Hamiltonian validation.")

    # 4. Build a particle-number-conserving UCCSD ansatz.
    initial_state = HartreeFock(
        num_spatial_orbitals=NUM_SPATIAL_ORBITALS,
        num_particles=NUM_PARTICLES,
        qubit_mapper=mapper,
    )

    ansatz = UCCSD(
        num_spatial_orbitals=NUM_SPATIAL_ORBITALS,
        num_particles=NUM_PARTICLES,
        qubit_mapper=mapper,
        initial_state=initial_state,
        reps=1,
        preserve_spin=True,
    )

    print(f"[8] Ansatz: UCCSD")
    print(f"[9] Ansatz parameters: {ansatz.num_parameters}")
    print(f"[10] Ansatz depth: {ansatz.depth()}")

    # 5. Run ideal statevector VQE.
    evaluations = []
    checkpoint_path = BASE / "wt_vqe_checkpoint.json"

    def callback(eval_count, parameters, energy, metadata):
        current_energy = float(np.real(energy))
        evaluations.append(current_energy)

        # Save progress periodically so a crash does not erase the history.
        if eval_count == 1 or eval_count % 10 == 0:
            checkpoint = {
                "experiment": "WT PETase CAS(4e,4o) UCCSD-VQE",
                "status": "running",
                "evaluation_count": eval_count,
                "latest_energy_hartree": current_energy,
                "exact_reference_hartree": exact_energy,
                "absolute_error_hartree": abs(current_energy - exact_energy),
                "evaluation_energies_hartree": evaluations,
            }
            temporary_path = checkpoint_path.with_suffix(".tmp")
            temporary_path.write_text(json.dumps(checkpoint, indent=2))
            temporary_path.replace(checkpoint_path)

            print(
                f"    Evaluation {eval_count:4d}: "
                f"{current_energy:.12f} Ha"
            )

    optimizer = SLSQP(
        maxiter=300,
        ftol=1e-9,
    )

    vqe = VQE(
        estimator=StatevectorEstimator(),
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
    )

    print("\n[11] Starting ideal statevector VQE...")
    result = vqe.compute_minimum_eigenvalue(qubit_op)

    vqe_energy = float(np.real(result.eigenvalue))
    vqe_error = abs(vqe_energy - exact_energy)

    optimizer_result = getattr(result, "optimizer_result", None)

    # Qiskit OptimizerResult does not guarantee a "success" attribute.
    # Record only fields actually provided by the installed version.
    optimizer_success = (
        getattr(optimizer_result, "success", None)
        if optimizer_result is not None else None
    )
    optimizer_message = (
        str(getattr(optimizer_result, "message", None))
        if optimizer_result is not None else None
    )
    optimizer_iterations = (
        getattr(optimizer_result, "nit", None)
        if optimizer_result is not None else None
    )

    # 6. Save reproducible benchmark results.
    output = {
        "experiment": "WT PETase CAS(4e,4o) UCCSD-VQE",
        "simulation": "Ideal statevector; noiseless; no hardware execution",
        "hamiltonian_source": str(FCIDUMP_PATH),
        "mapper": "Jordan-Wigner",
        "spin_orbital_ordering": "Blocked (alpha orbitals, then beta orbitals)",
        "num_spatial_orbitals": NUM_SPATIAL_ORBITALS,
        "num_particles": list(NUM_PARTICLES),
        "num_qubits": qubit_op.num_qubits,
        "pauli_terms": len(qubit_op),
        "ansatz": "UCCSD",
        "ansatz_parameters": ansatz.num_parameters,
        "ansatz_reps": 1,
        "optimizer": "SLSQP",
        "optimizer_maxiter": 300,
        "optimizer_success": optimizer_success,
        "optimizer_message": optimizer_message,
        "optimizer_iterations": (
            int(optimizer_iterations)
            if optimizer_iterations is not None else None
        ),
        "function_evaluations": getattr(result, "cost_function_evals", None),
        "callback_evaluations": len(evaluations),
        "pyscf_casci_reference_hartree": reference_energy,
        "qiskit_exact_sector_hartree": exact_energy,
        "vqe_energy_hartree": vqe_energy,
        "absolute_vqe_error_hartree": vqe_error,
        "exact_validation_error_hartree": exact_error,
        "vqe_converged_to_exact_tolerance": vqe_error < 1e-6,
        "evaluation_energies_hartree": evaluations,
    }

    BASE.mkdir(parents=True, exist_ok=True)
    temporary_output = OUTPUT_PATH.with_suffix(".tmp")
    temporary_output.write_text(json.dumps(output, indent=2))
    temporary_output.replace(OUTPUT_PATH)

    print("\n" + "=" * 65)
    print("VQE BENCHMARK COMPLETE")
    print("=" * 65)
    print(f"Exact energy: {exact_energy:.12f} Ha")
    print(f"VQE energy:   {vqe_energy:.12f} Ha")
    print(f"Absolute error: {vqe_error:.8e} Ha")
    print(f"Optimizer success: {optimizer_success}")
    print(f"Function evaluations: {output['function_evaluations']}")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
