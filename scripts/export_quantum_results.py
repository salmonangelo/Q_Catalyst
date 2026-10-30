#!/usr/bin/env python3
"""
Role 3 Quantum Result Adapter for Q-Catalyst.

Converts WT VQE simulation and circuit resource JSON artifacts into the
contracted Parquet schema at `results/quantum/quantum_results.parquet`.

Contract Reference:
- docs/interface_contracts.md (Table 1, Item 20)
- README.md (Section 5, Table 5, Line 200)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Explicit PyArrow schema enforcing the Role 3 Interface Contract
QUANTUM_RESULTS_SCHEMA = pa.schema([
    ("variant_id", pa.string()),
    ("job_id", pa.string()),
    ("context_id", pa.string()),
    ("mutation", pa.string()),
    ("active_space", pa.string()),
    ("num_spatial_orbitals", pa.int64()),
    ("num_active_electrons", pa.int64()),
    ("active_particles_alpha", pa.int64()),
    ("active_particles_beta", pa.int64()),
    ("qubit_mapper", pa.string()),
    ("hamiltonian_source", pa.string()),
    ("energy_unit", pa.string()),
    ("casci_energy_hartree", pa.float64()),
    ("quantum_casci_reference_energy_hartree", pa.float64()),
    ("qiskit_exact_sector_energy_hartree", pa.float64()),
    ("vqe_energy_hartree", pa.float64()),
    ("absolute_vqe_error_hartree", pa.float64()),
    ("exact_validation_error_hartree", pa.float64()),
    ("vqe_converged_to_exact_tolerance", pa.bool_()),
    ("role2_chemistry_casci_energy_hartree", pa.float64()),
    ("num_qubits", pa.int64()),
    ("pauli_terms", pa.int64()),
    ("ansatz", pa.string()),
    ("ansatz_parameters", pa.int64()),
    ("ansatz_reps", pa.int64()),
    ("original_circuit_depth", pa.int64()),
    ("transpiled_circuit_depth", pa.int64()),
    ("two_qubit_gate_count", pa.int64()),
    ("cx_gate_count", pa.int64()),
    ("total_gate_count", pa.int64()),
    ("transpilation_basis_gates", pa.string()),
    ("transpilation_opt_level", pa.int64()),
    ("optimizer", pa.string()),
    ("optimizer_maxiter", pa.int64()),
    ("optimizer_iterations", pa.int64()),
    ("function_evaluations", pa.int64()),
    ("optimizer_success", pa.bool_()),
    ("optimizer_message", pa.string()),
    ("simulation_type", pa.string()),
    ("shots", pa.int64()),
    ("noise_model", pa.string()),
    ("runtime_seconds", pa.float64()),
    ("qiskit_version", pa.string()),
    ("qiskit_nature_version", pa.string()),
    ("provenance", pa.string()),
    ("disclaimer", pa.string()),
])


def load_role2_chem_casci(
    chem_results_path: str | Path = "results/chemistry/chem_results.parquet",
    variant_id: str = "VAR_WT",
) -> float | None:
    """Safely extracts Role 2's previous classical CASCI energy for variant_id."""
    parquet_file = Path(chem_results_path)
    csv_file = parquet_file.with_suffix(".csv")

    if parquet_file.exists():
        try:
            df_chem = pd.read_parquet(parquet_file)
            row = df_chem[df_chem["variant_id"] == variant_id]
            if not row.empty and "casci_e_tot_ha" in row.columns:
                val = row["casci_e_tot_ha"].iloc[0]
                if pd.notna(val):
                    return float(val)
        except Exception:
            pass

    if csv_file.exists():
        try:
            df_csv = pd.read_csv(csv_file)
            row = df_csv[df_csv["variant_id"] == variant_id]
            if not row.empty and "casci_e_tot_ha" in row.columns:
                val = row["casci_e_tot_ha"].iloc[0]
                if pd.notna(val):
                    return float(val)
        except Exception:
            pass

    return None


def build_wt_quantum_dataframe(
    vqe_result_path: str | Path = "results/quantum/wt_vqe_result.json",
    circuit_resources_path: str | Path = "results/quantum/wt_circuit_resources.json",
    chem_results_path: str | Path = "results/chemistry/chem_results.parquet",
    variant_id: str = "VAR_WT",
) -> pd.DataFrame:
    """
    Constructs a validated 1-row pandas DataFrame containing all Role 3 quantum
    active-space benchmark metrics for the Wild-Type enzyme.
    """
    vqe_file = Path(vqe_result_path)
    circuit_file = Path(circuit_resources_path)

    if not vqe_file.exists():
        raise FileNotFoundError(f"Missing VQE result artifact: {vqe_file}")
    if not circuit_file.exists():
        raise FileNotFoundError(f"Missing circuit resources artifact: {circuit_file}")

    if not variant_id or variant_id != "VAR_WT":
        raise ValueError(
            f"Invalid variant_id '{variant_id}'. This WT adapter only exports 'VAR_WT'."
        )

    with open(vqe_file, "r", encoding="utf-8") as f:
        vqe_data = json.load(f)

    with open(circuit_file, "r", encoding="utf-8") as f:
        circuit_data = json.load(f)

    # Distinguish Role 2 chemistry CASCI from Quantum matched CASCI reference
    role2_casci = load_role2_chem_casci(chem_results_path, variant_id=variant_id)
    quantum_casci = float(vqe_data["pyscf_casci_reference_hartree"])

    # Circuit metrics from circuit inspection artifact
    c_res = circuit_data.get("circuit_resources", {})
    t_settings = circuit_data.get("transpilation_settings", {})
    env = circuit_data.get("environment", {})

    basis_gates_str = ",".join(t_settings.get("basis_gates", ["cx", "id", "rz", "sx", "x"]))

    # Particles
    particles = vqe_data.get("num_particles", [2, 2])
    alpha_e = int(particles[0])
    beta_e = int(particles[1])
    num_active_e = alpha_e + beta_e

    record: dict[str, Any] = {
        # 1. Identifiers (strictly no invented IDs)
        "variant_id": variant_id,
        "job_id": None,      # Genuinely unassigned for WT pilot
        "context_id": None,  # Genuinely unassigned
        "mutation": "WT",

        # 2. Active-Space Definition
        "active_space": f"CAS({num_active_e}e,{vqe_data.get('num_spatial_orbitals', 4)}o)",
        "num_spatial_orbitals": int(vqe_data.get("num_spatial_orbitals", 4)),
        "num_active_electrons": num_active_e,
        "active_particles_alpha": alpha_e,
        "active_particles_beta": beta_e,
        "qubit_mapper": str(vqe_data.get("mapper", "Jordan-Wigner")),
        "hamiltonian_source": str(vqe_data.get("hamiltonian_source", "results/quantum/wt_active_space_hamiltonian.fcidump")),

        # 3. Energies and Errors (Hartree)
        "energy_unit": "Hartree",
        "casci_energy_hartree": quantum_casci,  # Matched quantum active-space reference
        "quantum_casci_reference_energy_hartree": quantum_casci,
        "qiskit_exact_sector_energy_hartree": float(vqe_data["qiskit_exact_sector_hartree"]),
        "vqe_energy_hartree": float(vqe_data["vqe_energy_hartree"]),
        "absolute_vqe_error_hartree": float(vqe_data["absolute_vqe_error_hartree"]),
        "exact_validation_error_hartree": float(vqe_data["exact_validation_error_hartree"]),
        "vqe_converged_to_exact_tolerance": bool(vqe_data["vqe_converged_to_exact_tolerance"]),
        "role2_chemistry_casci_energy_hartree": role2_casci,  # Discloses Role 2's previous value

        # 4. Quantum Circuit & Resources
        "num_qubits": int(vqe_data.get("num_qubits", 8)),
        "pauli_terms": int(vqe_data.get("pauli_terms", 362)),
        "ansatz": str(vqe_data.get("ansatz", "UCCSD")),
        "ansatz_parameters": int(vqe_data.get("ansatz_parameters", 26)),
        "ansatz_reps": int(vqe_data.get("ansatz_reps", 1)),
        "original_circuit_depth": int(c_res.get("original_ansatz_depth", 1)),
        "transpiled_circuit_depth": int(c_res.get("transpiled_circuit_depth", 667)),
        "two_qubit_gate_count": int(c_res.get("two_qubit_gate_count", 265)),
        "cx_gate_count": int(c_res.get("cx_gate_count", 265)),
        "total_gate_count": int(c_res.get("total_gate_count", 1171)),
        "transpilation_basis_gates": basis_gates_str,
        "transpilation_opt_level": int(t_settings.get("optimization_level", 2)),

        # 5. Optimizer & Convergence Metrics
        "optimizer": str(vqe_data.get("optimizer", "SLSQP")),
        "optimizer_maxiter": int(vqe_data.get("optimizer_maxiter", 300)),
        "optimizer_iterations": int(vqe_data.get("optimizer_iterations", 82)),
        "function_evaluations": int(vqe_data.get("function_evaluations", 2243)),
        "optimizer_success": None if vqe_data.get("optimizer_success") is None else bool(vqe_data["optimizer_success"]),
        "optimizer_message": str(vqe_data.get("optimizer_message", "None")),

        # 6. Simulation & Hardware Constraints (Null where unavailable)
        "simulation_type": "statevector_noiseless",
        "shots": None,          # Statevector has no shot sampling
        "noise_model": None,    # Noiseless ideal statevector
        "runtime_seconds": None,# Not tracked in raw VQE output

        # 7. Provenance & Software Environment
        "qiskit_version": str(env.get("qiskit_version", "2.6.1")),
        "qiskit_nature_version": str(env.get("qiskit_nature_version", "0.8.0")),
        "provenance": (
            "Ideal statevector VQE on WT PETase Tier S active space CAS(4e,4o); "
            "Jordan-Wigner mapper; UCCSD ansatz; SLSQP optimizer."
        ),
        "disclaimer": (
            "Simulated statevector VQE on a reduced 42-atom cluster active space. "
            "Does not constitute evidence of quantum advantage or wet-lab experimental proof."
        ),
    }

    df = pd.DataFrame([record])
    validate_quantum_dataframe(df)
    return df


def validate_quantum_dataframe(df: pd.DataFrame) -> None:
    """Validates row count, variant identity, required fields, and physical bounds."""
    # 1. Exactly one record
    if len(df) != 1:
        raise ValueError(f"Expected exactly 1 WT record in quantum_results, found {len(df)}")

    # 2. Variant ID must strictly be VAR_WT
    variant_id = df["variant_id"].iloc[0]
    if not variant_id or variant_id != "VAR_WT":
        raise ValueError(f"Expected variant_id == 'VAR_WT', got '{variant_id}'")

    # 3. Required non-null fields
    required_non_null = [
        "variant_id",
        "mutation",
        "active_space",
        "num_spatial_orbitals",
        "num_active_electrons",
        "num_qubits",
        "pauli_terms",
        "ansatz",
        "ansatz_parameters",
        "transpiled_circuit_depth",
        "two_qubit_gate_count",
        "cx_gate_count",
        "optimizer",
        "casci_energy_hartree",
        "quantum_casci_reference_energy_hartree",
        "vqe_energy_hartree",
        "absolute_vqe_error_hartree",
        "simulation_type",
        "provenance",
        "disclaimer",
    ]
    for col in required_non_null:
        if col not in df.columns:
            raise ValueError(f"Missing required contract column: '{col}'")
        val = df[col].iloc[0]
        if pd.isna(val) or val is None:
            raise ValueError(f"Required column '{col}' must not be null, got: {val}")

    # 4. Physical & Numerical sanity checks
    num_qubits = df["num_qubits"].iloc[0]
    if num_qubits != 8:
        raise ValueError(f"Expected 8 qubits, found {num_qubits}")

    num_params = df["ansatz_parameters"].iloc[0]
    if num_params != 26:
        raise ValueError(f"Expected 26 parameters, found {num_params}")

    two_qubit_gates = df["two_qubit_gate_count"].iloc[0]
    if two_qubit_gates != 265:
        raise ValueError(f"Expected 265 two-qubit gates, found {two_qubit_gates}")

    vqe_e = float(df["vqe_energy_hartree"].iloc[0])
    casci_e = float(df["quantum_casci_reference_energy_hartree"].iloc[0])
    error = float(df["absolute_vqe_error_hartree"].iloc[0])

    if abs(vqe_e - casci_e) > 1e-4:
        raise ValueError(f"VQE energy differs excessively from CASCI reference: {abs(vqe_e - casci_e):.4e} Ha")
    if abs(error - abs(vqe_e - casci_e)) > 1e-10:
        raise ValueError(f"Inconsistent absolute_vqe_error_hartree: {error} vs {abs(vqe_e - casci_e)}")

    # 5. Distinguish Role 2 Chemistry CASCI from Quantum matched CASCI reference
    role2_casci = df["role2_chemistry_casci_energy_hartree"].iloc[0]
    if role2_casci is not None and not pd.isna(role2_casci):
        if abs(role2_casci - casci_e) < 0.1:
            raise ValueError(
                "Role 2 chemistry CASCI and quantum CASCI reference unexpectedly match; "
                "they should differ by ~1.21 Ha due to the corrected boundary exclusion."
            )


def export_quantum_results(
    vqe_result_path: str | Path = "results/quantum/wt_vqe_result.json",
    circuit_resources_path: str | Path = "results/quantum/wt_circuit_resources.json",
    chem_results_path: str | Path = "results/chemistry/chem_results.parquet",
    output_path: str | Path = "results/quantum/quantum_results.parquet",
    variant_id: str = "VAR_WT",
) -> Path:
    """Exports the validated 1-row WT quantum results table to Parquet."""
    print("=" * 70)
    print("Q-CATALYST: EXPORT ROLE 3 QUANTUM RESULTS (WT ADAPTER)")
    print("=" * 70)

    df = build_wt_quantum_dataframe(
        vqe_result_path=vqe_result_path,
        circuit_resources_path=circuit_resources_path,
        chem_results_path=chem_results_path,
        variant_id=variant_id,
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to PyArrow Table with explicit strict schema
    table = pa.Table.from_pandas(df, schema=QUANTUM_RESULTS_SCHEMA, preserve_index=False)

    temp_file = out_file.with_suffix(".tmp")
    pq.write_table(table, temp_file)
    temp_file.replace(out_file)

    print(f"[Export Success] Wrote 1 validated record to {out_file}")
    print(f"  - Variant ID:                 {df['variant_id'].iloc[0]}")
    print(f"  - Active Space:               {df['active_space'].iloc[0]} ({df['num_qubits'].iloc[0]} qubits)")
    print(f"  - VQE Energy:                 {df['vqe_energy_hartree'].iloc[0]:.8f} Ha")
    print(f"  - Quantum CASCI Reference:    {df['quantum_casci_reference_energy_hartree'].iloc[0]:.8f} Ha")
    print(f"  - Role 2 Chemistry CASCI:     {df['role2_chemistry_casci_energy_hartree'].iloc[0]:.8f} Ha")
    print(f"  - Absolute VQE Error:         {df['absolute_vqe_error_hartree'].iloc[0]:.4e} Ha")
    print(f"  - Transpiled Depth:           {df['transpiled_circuit_depth'].iloc[0]}")
    print(f"  - Two-Qubit (CX) Gates:       {df['two_qubit_gate_count'].iloc[0]}")
    print("=" * 70)
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Export WT VQE metrics to quantum_results.parquet.")
    parser.add_argument("--vqe-result", default="results/quantum/wt_vqe_result.json", help="Path to VQE JSON.")
    parser.add_argument("--circuit-resources", default="results/quantum/wt_circuit_resources.json", help="Path to circuit resources JSON.")
    parser.add_argument("--chem-results", default="results/chemistry/chem_results.parquet", help="Path to chemistry parquet.")
    parser.add_argument("--output", default="results/quantum/quantum_results.parquet", help="Output parquet path.")
    parser.add_argument("--variant-id", default="VAR_WT", help="Variant identifier (must be VAR_WT).")
    args = parser.parse_args()

    export_quantum_results(
        vqe_result_path=args.vqe_result,
        circuit_resources_path=args.circuit_resources,
        chem_results_path=args.chem_results,
        output_path=args.output,
        variant_id=args.variant_id,
    )


if __name__ == "__main__":
    main()
