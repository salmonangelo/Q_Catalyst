#!/usr/bin/env python3
"""
Inspect Circuit Resources for WT PETase CAS(4e,4o) UCCSD Ansatz.

Extracts simulated quantum circuit resource metrics (qubits, parameters,
original and transpiled depth, operation counts, two-qubit gate counts)
without executing VQE or altering existing scientific result files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import qiskit
from qiskit import transpile
import qiskit_nature
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract circuit resource metrics for WT CAS(4e,4o) UCCSD ansatz."
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="Qiskit transpiler optimization level (default: 2).",
    )
    parser.add_argument(
        "--basis-gates",
        nargs="+",
        default=["cx", "id", "rz", "sx", "x"],
        help="Basis gate set for transpilation (default: cx id rz sx x).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Transpiler random seed for deterministic decomposition (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/quantum/wt_circuit_resources.json",
        help="Path for exported JSON metrics (default: results/quantum/wt_circuit_resources.json).",
    )
    return parser.parse_args()


def inspect_wt_circuit_resources(
    opt_level: int = 2,
    basis_gates: list[str] | None = None,
    seed: int = 42,
    output_path: str | Path = "results/quantum/wt_circuit_resources.json",
) -> dict:
    if basis_gates is None:
        basis_gates = ["cx", "id", "rz", "sx", "x"]

    print("=" * 70)
    print("Q-CATALYST | WT ACTIVE-SPACE UCCSD CIRCUIT RESOURCE INSPECTION")
    print("=" * 70)

    # 1. Reconstruct identical WT quantum setup from scripts/run_quantum.py
    num_spatial_orbitals = 4
    num_particles = (2, 2)
    mapper = JordanWignerMapper()

    print("[1] Reconstructing reference quantum setup:")
    print(f"    - Spatial orbitals: {num_spatial_orbitals}")
    print(f"    - Active particles: {num_particles} (2 alpha, 2 beta)")
    print(f"    - Mapper:           Jordan-Wigner")

    initial_state = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
    )

    ansatz = UCCSD(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=initial_state,
        reps=1,
        preserve_spin=True,
    )

    # 2. Basic validation against existing WT experiment
    num_qubits = ansatz.num_qubits
    num_parameters = ansatz.num_parameters
    original_depth = int(ansatz.depth())

    print("\n[2] Validating against reference WT benchmark:")
    print(f"    - Qubits:           {num_qubits} (expected 8)")
    print(f"    - Ansatz params:    {num_parameters} (expected 26)")
    print(f"    - Original depth:   {original_depth}")

    if num_qubits != 8:
        raise ValueError(f"Validation failed: expected 8 qubits, got {num_qubits}")
    if num_parameters != 26:
        raise ValueError(f"Validation failed: expected 26 parameters, got {num_parameters}")

    print("    -> Validation PASSED: Circuit matches WT VQE benchmark specification.")

    # 3. Transpile parameterized ansatz to target basis
    print("\n[3] Transpiling parameterized ansatz:")
    print(f"    - Target basis:     {basis_gates}")
    print(f"    - Opt level:        {opt_level}")
    print(f"    - Seed:             {seed}")

    transpiled = transpile(
        ansatz,
        basis_gates=basis_gates,
        optimization_level=opt_level,
        seed_transpiler=seed,
    )

    transpiled_depth = int(transpiled.depth())
    op_counts = {k: int(v) for k, v in transpiled.count_ops().items()}

    # Count all two-qubit operations explicitly from circuit instruction data
    two_qubit_gate_count = sum(1 for inst in transpiled.data if len(inst.qubits) == 2)
    cx_count = int(op_counts.get("cx", 0))
    total_gate_count = sum(op_counts.values())

    print("\n[4] Extracted Resource Metrics:")
    print(f"    - Transpiled depth:      {transpiled_depth}")
    print(f"    - Two-qubit gates:       {two_qubit_gate_count}")
    print(f"    - CX gates:              {cx_count}")
    print(f"    - Total gate operations: {total_gate_count}")
    print(f"    - Detailed op counts:    {op_counts}")

    # 4. Assemble structured result
    result_data = {
        "experiment": "WT PETase CAS(4e,4o) UCCSD Circuit Resource Inspection",
        "simulation_note": (
            "These metrics represent simulated circuit resource estimates from algorithmic "
            "transpilation, not physical quantum hardware measurements."
        ),
        "environment": {
            "qiskit_version": getattr(qiskit, "__version__", "unknown"),
            "qiskit_nature_version": getattr(qiskit_nature, "__version__", "unknown"),
        },
        "active_space_specification": {
            "num_spatial_orbitals": num_spatial_orbitals,
            "num_particles": list(num_particles),
            "mapper": "Jordan-Wigner",
            "ansatz": "UCCSD",
            "reps": 1,
            "preserve_spin": True,
        },
        "circuit_validation": {
            "num_qubits": num_qubits,
            "num_parameters": num_parameters,
            "matches_wt_vqe_experiment": bool(num_qubits == 8 and num_parameters == 26),
        },
        "transpilation_settings": {
            "basis_gates": basis_gates,
            "optimization_level": opt_level,
            "seed_transpiler": seed,
        },
        "circuit_resources": {
            "num_qubits": num_qubits,
            "num_parameters": num_parameters,
            "original_ansatz_depth": original_depth,
            "transpiled_circuit_depth": transpiled_depth,
            "two_qubit_gate_count": two_qubit_gate_count,
            "cx_gate_count": cx_count,
            "total_gate_count": total_gate_count,
            "operation_counts": op_counts,
        },
    }

    # 5. Save atomically to output file
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = out_file.with_suffix(".tmp")
    temp_file.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    temp_file.replace(out_file)

    print(f"\n[5] Metrics successfully saved to: {out_file}")
    print("=" * 70)
    return result_data


def main():
    args = parse_args()
    inspect_wt_circuit_resources(
        opt_level=args.opt_level,
        basis_gates=args.basis_gates,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
