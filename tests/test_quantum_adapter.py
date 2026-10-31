"""
Focused unit and regression tests for the Role 3 Quantum Result Adapter.

Validates the export of `results/quantum/quantum_results.parquet`, compliance
with interface contracts, numerical accuracy, explicit null preservation,
and robust rejection of missing or mismatched variant IDs.
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath("."))

from scripts.export_quantum_results import (
    build_wt_quantum_dataframe,
    validate_quantum_dataframe,
    export_quantum_results,
    QUANTUM_RESULTS_SCHEMA,
)


def test_quantum_results_parquet_contract():
    """
    Validates that quantum_results.parquet satisfies the Role 3 interface contract:
    - Exactly 1 record for WT
    - Correct variant_id ('VAR_WT') and mutation ('WT')
    - Correct active space CAS(4e,4o), 8 qubits, 26 parameters
    - Accurately extracted transpiled depth (667) and two-qubit CX gates (265)
    - Distinguishes quantum CASCI reference from Role 2 chemistry CASCI
    - Truly null fields for unassigned IDs, shots, noise, and runtime
    """
    parquet_path = Path("results/quantum/quantum_results.parquet")
    assert parquet_path.exists(), f"Target parquet file missing at {parquet_path}"

    df = pd.read_parquet(parquet_path)

    # 1. Row count and variant identity
    assert len(df) == 1, f"Expected 1 record, got {len(df)}"
    assert df["variant_id"].iloc[0] == "VAR_WT"
    assert df["mutation"].iloc[0] == "WT"

    # 2. Active-space and circuit specifications
    assert df["active_space"].iloc[0] == "CAS(4e,4o)"
    assert df["num_spatial_orbitals"].iloc[0] == 4
    assert df["num_active_electrons"].iloc[0] == 4
    assert df["active_particles_alpha"].iloc[0] == 2
    assert df["active_particles_beta"].iloc[0] == 2
    assert df["qubit_mapper"].iloc[0] == "Jordan-Wigner"
    assert df["num_qubits"].iloc[0] == 8
    assert df["pauli_terms"].iloc[0] == 362
    assert df["ansatz"].iloc[0] == "UCCSD"
    assert df["ansatz_parameters"].iloc[0] == 26
    assert df["transpiled_circuit_depth"].iloc[0] == 667
    assert df["two_qubit_gate_count"].iloc[0] == 265
    assert df["cx_gate_count"].iloc[0] == 265
    assert df["total_gate_count"].iloc[0] == 1171

    # 3. Energy metrics and distinct CASCI values
    vqe_e = df["vqe_energy_hartree"].iloc[0]
    quant_casci = df["quantum_casci_reference_energy_hartree"].iloc[0]
    casci_alias = df["casci_energy_hartree"].iloc[0]
    role2_casci = df["role2_chemistry_casci_energy_hartree"].iloc[0]
    err = df["absolute_vqe_error_hartree"].iloc[0]

    assert quant_casci == pytest.approx(-1024.259623, abs=1e-5)
    assert casci_alias == quant_casci
    assert vqe_e == pytest.approx(-1024.259599, abs=1e-5)
    assert err == pytest.approx(2.40e-5, abs=1e-6)
    assert abs(vqe_e - quant_casci) < 1e-4

    # Requirement 7: Explicitly distinguish quantum handoff CASCI from Role 2 chemistry CASCI
    assert role2_casci == pytest.approx(-1023.047399, abs=1e-5)
    assert abs(quant_casci - role2_casci) > 1.0, (
        f"Quantum CASCI ({quant_casci:.4f}) and Role 2 CASCI ({role2_casci:.4f}) "
        f"must differ because quantum handoff fixed the boundary exclusion clashes."
    )

    # 4. Explicit null fields (Requirement 6: do not invent values)
    assert pd.isna(df["job_id"].iloc[0])
    assert pd.isna(df["context_id"].iloc[0])
    assert pd.isna(df["shots"].iloc[0])
    assert pd.isna(df["noise_model"].iloc[0])
    assert pd.isna(df["runtime_seconds"].iloc[0])
    assert pd.isna(df["optimizer_success"].iloc[0])

    # 5. Simulation type & provenance
    assert df["simulation_type"].iloc[0] == "statevector_noiseless"
    assert "UCCSD" in df["provenance"].iloc[0]
    assert "SLSQP" in df["provenance"].iloc[0]


def test_quantum_adapter_rejects_mismatched_variant_id():
    """Validates that the adapter strictly rejects a mutant or mismatched variant ID."""
    with pytest.raises(ValueError, match="Invalid variant_id"):
        build_wt_quantum_dataframe(variant_id="VAR_A4_01")

    with pytest.raises(ValueError, match="Invalid variant_id"):
        build_wt_quantum_dataframe(variant_id="S121E")

    with pytest.raises(ValueError, match="Invalid variant_id"):
        build_wt_quantum_dataframe(variant_id="UNKNOWN")


def test_quantum_adapter_rejects_missing_variant_id():
    """Validates that the adapter strictly rejects missing or empty variant IDs."""
    with pytest.raises(ValueError, match="Invalid variant_id"):
        build_wt_quantum_dataframe(variant_id="")

    with pytest.raises(ValueError, match="Invalid variant_id"):
        build_wt_quantum_dataframe(variant_id=None)  # type: ignore


def test_quantum_adapter_rejects_invalid_row_counts():
    """Validates that the validator strictly enforces a 1-row invariant."""
    df_valid = build_wt_quantum_dataframe()
    df_two_rows = pd.concat([df_valid, df_valid], ignore_index=True)

    with pytest.raises(ValueError, match="Expected exactly 1 WT record"):
        validate_quantum_dataframe(df_two_rows)

    df_empty = df_valid.iloc[0:0]
    with pytest.raises(ValueError, match="Expected exactly 1 WT record"):
        validate_quantum_dataframe(df_empty)


def test_quantum_adapter_rejects_missing_required_column():
    """Validates that dropping any required schema column fails validation."""
    df_valid = build_wt_quantum_dataframe()

    for required_col in ["num_qubits", "vqe_energy_hartree", "ansatz", "optimizer"]:
        df_invalid = df_valid.drop(columns=[required_col])
        with pytest.raises(ValueError, match="Missing required contract column"):
            validate_quantum_dataframe(df_invalid)
