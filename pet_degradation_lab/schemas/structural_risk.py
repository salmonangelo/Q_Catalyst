"""Structural risk assessment and active-site proximity schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from schemas.provenance import ProvenanceRecord


class StructuralRiskRecord(BaseModel):
    """
    Preserves structural-risk metrics and geometry checks.
    Sequence-numbering heuristics are strictly distinguished from genuine 3D structural analyses.
    """
    variant_id: str = Field(..., description="Variant identifier")
    structural_risk_score: Optional[float] = Field(None, description="Quantitative structural risk score [0, 1]")
    method_used: str = Field(..., description="Method: e.g. '3D_CA_MIN_EUCLIDEAN_5XJH' or 'SEQUENCE_NUMBERING_HEURISTIC'")
    is_3d_structural_analysis: bool = Field(..., description="True if computed from 3D coordinates, False if sequence heuristic")
    active_site_proximity_angstrom: Optional[float] = Field(None, description="Minimum Euclidean distance (Å) to Ser160/catalytic triad")
    catalytic_site_warnings: List[str] = Field(default_factory=list, description="Warnings if within catalytic pocket")
    geometry_checks: Dict[str, Any] = Field(default_factory=dict, description="Geometry diagnostics")
    clash_reports: List[str] = Field(default_factory=list, description="Steric clash observations")
    validation_status: str = Field("VERIFIED", description="Status of assessment")
    unresolved_checks: List[str] = Field(default_factory=list, description="Pending or unexecuted structural checks")
    provenance: ProvenanceRecord = Field(..., description="Provenance of the risk calculation")
