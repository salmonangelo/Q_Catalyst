"""Scientific validation engine for pet_degradation_lab.
Verifies data integrity, sequence alignments, and ensures missing values are never invented.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
from schemas.candidate import CandidateVariant
from schemas.evidence_status import EvidenceClassification


class ScientificValidator:
    """
    Validates imported candidates against scientific rules and integrity contracts.
    """

    @staticmethod
    def validate_candidate_integrity(candidate: CandidateVariant) -> Tuple[bool, List[str]]:
        """
        Validates individual candidate variant.
        Returns (is_valid, list_of_issues).
        """
        issues: List[str] = []

        # 1. Identifier checks
        if not candidate.variant_id:
            issues.append("Missing variant_id")

        # 2. Sequence checks
        if candidate.sequence:
            if candidate.sequence.validation_status != "VALID":
                issues.append(f"Sequence flagged as {candidate.sequence.validation_status}: {candidate.sequence.validation_notes}")

        # 3. Alignment check if WT reference is present
        if candidate.sequence and candidate.wt_reference:
            # Check length sanity
            if abs(candidate.sequence.sequence_length - candidate.wt_reference.sequence_length) > 10:
                issues.append(
                    f"Severe sequence length discrepancy: mutant={candidate.sequence.sequence_length} vs WT={candidate.wt_reference.sequence_length}"
                )

        # 4. Structural model check
        if candidate.structural_model:
            if candidate.structural_model.resolved_residue_count == 0:
                issues.append("Structural model contains 0 resolved CA residues")

        # 5. Non-invention rule: check that missing values are NOT populated with dummy values
        if candidate.evidence_status.get("dft") == EvidenceClassification.NOT_AVAILABLE:
            if candidate.dft_result is not None and candidate.dft_result.calculated_energy_hartree is not None:
                issues.append("VIOLATION: DFT marked NOT_AVAILABLE but contains non-null energy value")

        if candidate.evidence_status.get("quantum") == EvidenceClassification.NOT_AVAILABLE:
            if candidate.quantum_result is not None and candidate.quantum_result.vqe_energy_hartree is not None:
                issues.append("VIOLATION: Quantum marked NOT_AVAILABLE but contains non-null VQE value")

        return len(issues) == 0, issues

    @classmethod
    def audit_candidate_cohort(cls, candidates: List[CandidateVariant]) -> Dict[str, Any]:
        """Audits entire candidate cohort and generates a validation scorecard."""
        total = len(candidates)
        passed = 0
        total_issues = []
        with_sequence = 0
        with_structure = 0
        with_dft = 0
        with_quantum = 0

        for c in candidates:
            ok, issues = cls.validate_candidate_integrity(c)
            if ok:
                passed += 1
            else:
                total_issues.extend([f"[{c.variant_id}] {i}" for i in issues])

            if c.sequence is not None:
                with_sequence += 1
            if c.structural_model is not None:
                with_structure += 1
            if c.dft_result is not None:
                with_dft += 1
            if c.quantum_result is not None:
                with_quantum += 1

        return {
            "total_candidates": total,
            "validation_passed": passed,
            "validation_failed": total - passed,
            "pass_rate_pct": (passed / total * 100) if total > 0 else 0.0,
            "coverage": {
                "sequence_coverage_pct": (with_sequence / total * 100) if total > 0 else 0.0,
                "structure_coverage_pct": (with_structure / total * 100) if total > 0 else 0.0,
                "dft_coverage_pct": (with_dft / total * 100) if total > 0 else 0.0,
                "quantum_coverage_pct": (with_quantum / total * 100) if total > 0 else 0.0,
            },
            "issues": total_issues,
        }
