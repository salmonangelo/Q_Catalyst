"""
Active-Space Electronic Hamiltonian Generation and Validation for Q-Catalyst Role 3.

This module provides the primary entry point for generating or retrieving the
active-space electronic Hamiltonian for catalytic variants.
Strictly enforces QM/MM electrostatic embedding scientific validation guards:
if a variant's embedding is marked scientifically invalid (e.g. W159H embedding freeze),
Hamiltonian generation is blocked before any quantum active space or PySCF execution begins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from qcatalyst.chemistry.cluster import ActiveSiteClusterBuilder
from qcatalyst.chemistry.embedding import (
    ElectrostaticEmbeddingBuilder,
    validate_embedding_for_quantum_execution,
)


def generate_active_space_hamiltonian(
    mutation: str = "WT",
    pdb_path: str = "data/structures/5XJH.pdb",
    mutant_pdb_path: Optional[str] = None,
    radius_cutoff: float = 10.0,
    output_dir: str = "results/quantum",
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates or retrieves the active-space electronic Hamiltonian for a given variant.

    1. Builds the active site cluster and electrostatic embedding.
    2. Validates the QM/MM embedding using `validate_embedding_for_quantum_execution()`.
       If the validation status is FAIL (such as W159H), execution is immediately aborted
       with a RuntimeError detailing all unresolved scientific blockers.
    3. For WT ("WT"), retrieves and returns the verified Role 2 -> Role 3 handoff artifacts
       (FCIDUMP, JSON metadata, NPZ arrays).
    4. For valid mutant candidates, delegates to PySCF CASCI reference generation.
    """
    # 1. Build cluster and electrostatic embedding
    cluster_builder = ActiveSiteClusterBuilder(pdb_path)
    cluster = cluster_builder.build_tier_s()
    qm_coords = np.array([c for _, c in cluster["atoms"]])

    embed_builder = ElectrostaticEmbeddingBuilder(pdb_path)
    mm_coords, mm_charges, embed_meta = embed_builder.get_environment_charges(
        qm_coords=qm_coords,
        radius_cutoff=radius_cutoff,
        mutation=mutation,
        mutant_pdb_path=mutant_pdb_path,
    )

    # 2. Strict Safety Guard: Validate embedding for quantum execution
    validate_embedding_for_quantum_execution(embed_meta)

    # 3. For WT reference, retrieve existing verified artifacts to ensure 100% numerical preservation
    mutation_clean = (mutation or "WT").strip().upper()
    if mutation_clean in ["WT", "NONE", ""]:
        actual_prefix = prefix or "wt_active_space_hamiltonian"
        out_base = Path(output_dir)
        json_path = out_base / f"{actual_prefix}.json"
        npz_path = out_base / f"{actual_prefix}.npz"
        fcidump_path = out_base / f"{actual_prefix}.fcidump"

        if not (json_path.exists() and npz_path.exists() and fcidump_path.exists()):
            raise FileNotFoundError(
                f"WT active-space Hamiltonian artifacts not found at {out_base / actual_prefix}.* "
                f"Expected pre-computed verified artifacts for WT reference."
            )

        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return {
            "status": "available",
            "mutation": "WT",
            "variant_id": "VAR_WT",
            "prefix": actual_prefix,
            "artifacts": {
                "json": str(json_path),
                "npz": str(npz_path),
                "fcidump": str(fcidump_path),
            },
            "hamiltonian_metadata": metadata,
            "embedding_metadata": embed_meta,
            "scientific_validation_status": "PASS",
        }

    # For any future valid mutants (if PySCF is available)
    from qcatalyst.chemistry.pyscf_engine import PySCFChemistryEngine
    engine = PySCFChemistryEngine(basis="def2-svp", conv_tol=1e-6)
    actual_prefix = prefix or f"{mutation.lower()}_active_space_hamiltonian"
    cas_res = engine.run_casci_reference(
        geometry_str=cluster["geometry_str"],
        ncas=4,
        nelecas=(2, 2),
        charge=cluster["charge"],
        spin=cluster["spin"],
        mm_coords=mm_coords,
        mm_charges=mm_charges,
        metadata=embed_meta,
    )
    paths = engine.export_active_space_hamiltonian(
        casci_result=cas_res,
        output_dir=output_dir,
        prefix=actual_prefix,
        metadata=embed_meta,
    )
    return {
        "status": "generated",
        "mutation": mutation,
        "prefix": actual_prefix,
        "artifacts": paths,
        "casci_result": cas_res,
        "embedding_metadata": embed_meta,
        "scientific_validation_status": "PASS",
    }

