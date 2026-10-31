"""Data provenance model retaining source tracking for every scientific value."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from schemas.evidence_status import EvidenceClassification


class ProvenanceRecord(BaseModel):
    """
    Mandatory provenance record tracking origin, checksum, and evidence category.
    """
    source_file: str = Field(..., description="Name or relative identifier of the source file")
    source_repo_or_origin: str = Field(..., description="Origin repository or data collection")
    source_relative_path: str = Field(..., description="Path to source file relative to source root")
    original_field_name: Optional[str] = Field(None, description="Original raw column or JSON field name")
    import_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp of data ingestion"
    )
    source_sha256: str = Field(..., description="SHA-256 checksum of source file at ingestion time")
    software_and_method: Optional[str] = Field(None, description="Software package, model, or algorithm name")
    evidence_classification: EvidenceClassification = Field(..., description="Classification category")
    validation_warnings: List[str] = Field(default_factory=list, description="Warnings or caveats identified during audit")
    raw_attributes: Dict[str, Any] = Field(default_factory=dict, description="Preserved unmapped raw attributes")
