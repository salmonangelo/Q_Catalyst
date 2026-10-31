"""Candidate enzyme variant master schema and AI prediction models."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from schemas.evidence_status import EvidenceClassification
from schemas.provenance import ProvenanceRecord
from schemas.sequence import SequenceRecord, WTReferenceRecord, MutationDifferenceRecord
from schemas.structure import StructuralModelRecord
from schemas.structural_risk import StructuralRiskRecord
from schemas.dft import DFTResultRecord
from schemas.quantum import QuantumResultRecord


class AIPredictionRecord(BaseModel):
    """Preserves original selection and model inference evidence."""
    ai_rank: Optional[int] = Field(None, description="Original AI screening rank")
    model_name: str = Field(..., description="Model name (e.g. XGBoost, ESM-1v, Ridge)")
    model_version: str = Field("v1.0", description="Model version")
    predicted_activity: Optional[float] = Field(None, description="Predicted relative activity vs WT")
    predicted_activity_lower: Optional[float] = Field(None, description="Conformal prediction lower bound")
    predicted_activity_upper: Optional[float] = Field(None, description="Conformal prediction upper bound")
    predicted_delta_tm: Optional[float] = Field(None, description="Predicted change in melting temperature (°C)")
    zero_shot_esm1v_score: Optional[float] = Field(None, description="ESM-1v masked-marginal log-likelihood ratio")
    ensemble_disagreement_std: Optional[float] = Field(None, description="Bootstrap ensemble standard deviation")
    conformal_interval_width: Optional[float] = Field(None, description="Upper - lower conformal interval width")
    ood_embedding_distance: Optional[float] = Field(None, description="Normalized ESM-2 embedding OOD distance")
    aggregate_uncertainty_score: Optional[float] = Field(None, description="Composite uncertainty score U(x) in [0, 1]")
    gate0_status: str = Field("FAIL_FALLBACK_DIVERSITY", description="Gate 0 uncertainty validation outcome")
    provenance: ProvenanceRecord = Field(..., description="Model prediction provenance")


class CandidateVariant(BaseModel):
    """
    Master standardized scientific record for a PETase candidate variant.
    Preserves raw source records alongside validated and normalized attributes.
    """
    variant_id: str = Field(..., description="Unique variant identifier (e.g. VAR-ENG-004, S121E)")
    parent_enzyme: str = Field("IsPETase", description="Parent wild-type scaffold enzyme")
    mutations_raw: str = Field(..., description="Original raw mutation notation")
    mutations_canonical: str = Field(..., description="Standard sorted mutation representation")
    mutation_count: int = Field(..., description="Number of amino-acid point mutations")
    
    # Scientific Evidence Subsections
    sequence: Optional[SequenceRecord] = Field(None, description="Full amino-acid sequence record")
    wt_reference: Optional[WTReferenceRecord] = Field(None, description="Wild-type reference sequence")
    mutation_differences: List[MutationDifferenceRecord] = Field(default_factory=list, description="Validated sequence differences")
    ai_prediction: Optional[AIPredictionRecord] = Field(None, description="AI ranking, scores, and uncertainty")
    structural_model: Optional[StructuralModelRecord] = Field(None, description="Structural coordinates and residue map")
    structural_risk: Optional[StructuralRiskRecord] = Field(None, description="Active-site proximity and structural risk")
    dft_result: Optional[DFTResultRecord] = Field(None, description="Active-site cluster DFT descriptors")
    quantum_result: Optional[QuantumResultRecord] = Field(None, description="Active-space CASCI/VQE calculations")
    
    # Metadata & Prior Art Flags
    is_prior_art: bool = Field(False, description="Flag indicating previously published benchmark variant")
    prior_art_name: Optional[str] = Field(None, description="Literature enzyme name (e.g. FAST-PETase, DuraPETase)")
    confidence_tier: Optional[str] = Field(None, description="High, Medium, or Low Confidence Tier")
    recommendation_reason: Optional[str] = Field(None, description="Algorithmic or biochemical selection reason")
    limitations: Optional[str] = Field(None, description="Documented physical, structural, or assay caveats")
    
    # Evidence Classification Matrix
    evidence_status: Dict[str, EvidenceClassification] = Field(
        default_factory=dict,
        description="Explicit breakdown of evidence type across each scientific dimension"
    )
    
    # Raw unmapped record preservation
    raw_record: Dict[str, Any] = Field(default_factory=dict, description="Complete unedited source record")
