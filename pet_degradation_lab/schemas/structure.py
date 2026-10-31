"""Structural model schemas and residue mapping contracts."""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from schemas.provenance import ProvenanceRecord


class ResidueMappingEntry(BaseModel):
    """
    Explicitly distinguishes sequence index, PDB residue number, insertion code, and mature numbering.
    Never assume residue 160 in sequence always maps to residue 160 in structure.
    """
    sequence_position_1based: int = Field(..., description="1-based index in the sequence")
    amino_acid: str = Field(..., description="Single-letter amino acid code")
    pdb_residue_number: int = Field(..., description="Residue sequence number in PDB ATOM record")
    chain_id: str = Field("A", description="Chain identifier in the PDB file")
    insertion_code: str = Field("", description="PDB insertion code if applicable")
    mature_numbering: Optional[int] = Field(None, description="Mature protein numbering if signal peptide removed")
    is_catalytic_triad: bool = Field(False, description="Flag if residue is part of catalytic triad (160, 206, 237)")
    is_oxyanion_hole: bool = Field(False, description="Flag if residue is oxyanion hole donor")


class StructuralModelRecord(BaseModel):
    """Structural model metadata and verification status."""
    model_id: str = Field(..., description="Unique structural model identifier")
    variant_id: str = Field(..., description="Associated candidate variant ID")
    file_format: str = Field("PDB", description="Format: PDB, mmCIF, or XYZ")
    structure_source: str = Field(..., description="EXPERIMENTAL_PDB, HOMOLOGY_MODEL, ALPHA_FOLD, or RELAXED_CLUSTER")
    relative_path: str = Field(..., description="Relative file path within lab data directory")
    file_sha256: str = Field(..., description="SHA-256 checksum of structural file")
    chain_id: str = Field("A", description="Primary catalytic protein chain")
    resolved_residue_count: int = Field(0, description="Total number of resolved amino-acid residues")
    has_water_molecules: bool = Field(False, description="Whether crystallographic waters are present")
    residue_mappings: List[ResidueMappingEntry] = Field(default_factory=list, description="Explicit residue coordinate map")
    validation_status: str = Field("UNVALIDATED", description="PASSED, CAUTION, or FAILED")
    validation_notes: List[str] = Field(default_factory=list, description="Structural audit notes")
    provenance: ProvenanceRecord = Field(..., description="Full provenance record")
