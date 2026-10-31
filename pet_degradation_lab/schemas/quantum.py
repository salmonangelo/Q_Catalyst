"""Quantum active-space and VQE calculation schemas."""
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from schemas.provenance import ProvenanceRecord


class QuantumResultRecord(BaseModel):
    """
    Preserves multireference active-space CASCI reference and VQE simulation results.
    Agreement between VQE and CASCI benchmarks quantum solver implementation;
    it is never proof of PET degradation rate or enzymatic turnover improvement.
    """
    calculation_id: str = Field(..., description="Unique quantum run identifier")
    variant_id: str = Field(..., description="Variant identifier or WT")
    active_space_definition: str = Field("(4e, 4o)", description="Active space specification")
    n_electrons: int = Field(4, description="Number of active electrons")
    n_orbitals: int = Field(4, description="Number of active spatial orbitals")
    n_qubits: int = Field(8, description="Number of logical qubits mapped (Jordan-Wigner)")
    casci_reference_energy_hartree: Optional[float] = Field(None, description="Exact classical CASCI eigenvalue in Hartree")
    vqe_energy_hartree: Optional[float] = Field(None, description="VQE estimated ground-state energy in Hartree")
    energy_difference_hartree: Optional[float] = Field(None, description="VQE - CASCI energy discrepancy")
    energy_difference_kcal_mol: Optional[float] = Field(None, description="Error converted to kcal/mol")
    ansatz_type: Optional[str] = Field("TwoLocal_Ry_Rz", description="Parameterized quantum circuit ansatz")
    optimizer: Optional[str] = Field("COBYLA", description="Classical parameter optimization algorithm")
    quantum_backend_or_simulator: str = Field("StatevectorSimulator", description="Simulator or hardware backend name")
    noise_level: float = Field(0.0, description="Depolarizing noise probability rate")
    convergence_status: str = Field("CONVERGED", description="CONVERGED, MAX_ITERATIONS, or FAILED")
    provenance: ProvenanceRecord = Field(..., description="Provenance record")
