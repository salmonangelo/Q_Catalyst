"""Computational chemistry and DFT calculation result schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from schemas.provenance import ProvenanceRecord


class DFTResultRecord(BaseModel):
    """
    Preserves quantum chemical calculation evidence (PySCF DFT/HF).
    Do not interpret a single-point DFT energy as an enzyme activity value or PET degradation rate.
    """
    calculation_id: str = Field(..., description="Unique calculation identifier (e.g. CHEM_5XJH_S121E)")
    variant_id: str = Field(..., description="Variant identifier or WT")
    mutation_str: str = Field(..., description="Evaluated mutation")
    calculated_energy_hartree: Optional[float] = Field(None, description="Ground state electronic energy in Hartree")
    energy_units: str = Field("Hartree", description="Primary energy units")
    method: str = Field("DFT", description="Electronic structure method (e.g. DFT, RHF)")
    functional: str = Field("B3LYP", description="Exchange-correlation functional")
    basis_set: str = Field("def2-SVP", description="Gaussian basis set")
    software_and_version: str = Field("PySCF 2.14.0", description="Computational chemistry package")
    molecular_system_identity: str = Field("Tier S Catalytic Triad Cluster (42 atoms)", description="Cluster description")
    net_charge: int = Field(-1, description="Net cluster charge")
    spin_multiplicity: int = Field(1, description="Singlet spin multiplicity 2S+1")
    density_fitting: bool = Field(True, description="Whether RI-J / density fitting was used")
    embedding_scheme: Optional[str] = Field("AMBER ff14SB (10.0 Å sphere)", description="QM/MM embedding environment")
    descriptors: Dict[str, Any] = Field(default_factory=dict, description="Mechanistic descriptors: ESP, partial charges, proxies")
    calculation_status: str = Field("CONVERGED", description="CONVERGED, FAILED, or PENDING")
    wall_clock_seconds: Optional[float] = Field(None, description="Computation runtime")
    provenance: ProvenanceRecord = Field(..., description="Provenance record")
