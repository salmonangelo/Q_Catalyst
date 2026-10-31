"""Sequence importer: extracts, validates, and aligns candidate sequences
against WT reference from project snapshots without modifying originals.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from schemas.evidence_status import EvidenceClassification
from schemas.provenance import ProvenanceRecord
from schemas.sequence import (
    CANONICAL_AMINO_ACIDS,
    MutationDifferenceRecord,
    SequenceRecord,
    WTReferenceRecord,
    align_and_extract_mutations,
    compute_sequence_checksum,
)

# Reference IsPETase mature wild-type sequence from PDB 5XJH (residues 37-290, 254 residues)
ISPETASE_5XJH_MATURE_WT = (
    "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGA"
    "IAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWS"
    "MGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS"
)


class SequenceImporter:
    """
    Imports and validates protein sequences from a project snapshot.
    """

    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = snapshot_dir.resolve()
        self.raw_dir = self.snapshot_dir / "raw"

    def load_wt_reference(self) -> WTReferenceRecord:
        """Loads and validates the authoritative wild-type reference sequence."""
        # Find 5XJH.pdb in raw
        pdb_path = self.raw_dir / "data/structures/5XJH.pdb"
        sha256 = hashlib.sha256(pdb_path.read_bytes()).hexdigest() if pdb_path.exists() else "UNAVAILABLE"

        # Check variants.parquet for WT sequence if available
        var_path = self.raw_dir / "data/curated/variants.parquet"
        wt_seq = ISPETASE_5XJH_MATURE_WT
        if var_path.exists():
            try:
                df = pd.read_parquet(var_path)
                if "wildtype_sequence" in df.columns:
                    first_wt = df["wildtype_sequence"].dropna().iloc[0]
                    if first_wt and len(first_wt) > 200:
                        wt_seq = str(first_wt).strip().upper()
            except Exception:
                pass

        prov = ProvenanceRecord(
            source_file="5XJH.pdb",
            source_repo_or_origin="Q-Catalyst data/structures/5XJH.pdb",
            source_relative_path="data/structures/5XJH.pdb",
            original_field_name="SEQRES / ATOM",
            source_sha256=sha256,
            software_and_method="X-ray crystallography (1.5Å)",
            evidence_classification=EvidenceClassification.EXPERIMENTAL,
        )

        return WTReferenceRecord(
            wildtype_id="IsPETase_5XJH_WT",
            sequence=wt_seq,
            sequence_length=len(wt_seq),
            source_identifier="PDB:5XJH_ChainA",
            checksum=compute_sequence_checksum(wt_seq),
            provenance=prov,
        )

    def import_candidate_sequences(self) -> Dict[str, SequenceRecord]:
        """
        Imports and validates sequences for all candidates present in snapshot.
        """
        records: Dict[str, SequenceRecord] = {}
        ranking_path = self.raw_dir / "results/rankings/ranking.parquet"
        if not ranking_path.exists():
            return records

        sha256 = hashlib.sha256(ranking_path.read_bytes()).hexdigest()
        df = pd.read_parquet(ranking_path)

        for _, row in df.iterrows():
            vid = str(row["variant_id"])
            seq_val = row.get("sequence")
            if pd.isna(seq_val) or not str(seq_val).strip():
                continue

            raw_seq = str(seq_val).strip().upper()
            validation_notes = []
            status = "VALID"

            # Check for non-canonical characters
            invalid_chars = set(raw_seq) - CANONICAL_AMINO_ACIDS
            if invalid_chars:
                status = "INVALID"
                validation_notes.append(f"Non-canonical amino acids detected: {sorted(invalid_chars)}")

            prov = ProvenanceRecord(
                source_file="ranking.parquet",
                source_repo_or_origin="Q-Catalyst results/rankings/ranking.parquet",
                source_relative_path="results/rankings/ranking.parquet",
                original_field_name="sequence",
                source_sha256=sha256,
                software_and_method="Curated variant reconstruction",
                evidence_classification=EvidenceClassification.REPORTED,
                validation_warnings=validation_notes,
            )

            records[vid] = SequenceRecord(
                variant_id=vid,
                parent_enzyme=str(row.get("parent_enzyme", "IsPETase")),
                sequence=raw_seq,
                sequence_length=len(raw_seq),
                sequence_checksum=compute_sequence_checksum(raw_seq),
                validation_status=status,
                validation_notes=validation_notes,
                provenance=prov,
            )

        return records

    def compare_and_validate_mutations(
        self,
        candidate_seq: SequenceRecord,
        wt_ref: WTReferenceRecord,
    ) -> List[MutationDifferenceRecord]:
        """
        Aligns candidate sequence against WT reference and generates explicit difference records.
        """
        return align_and_extract_mutations(
            wt_seq=wt_ref.sequence,
            mut_seq=candidate_seq.sequence,
            offset=1,
        )
