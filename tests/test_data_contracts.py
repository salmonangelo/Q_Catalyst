"""
Unit tests validating data schemas and contracts across pipeline outputs.
"""
import pytest
import pandas as pd
from pathlib import Path
from src.qcatalyst.data.loader import get_curated_variants_data
from src.qcatalyst.structure.proximity import is_eligible_for_chemistry, calculate_variant_min_distance


def test_curated_variants_schema():
    df = get_curated_variants_data()
    required_cols = [
        "variant_id", "parent_enzyme", "mutations", "mutation_count",
        "relative_activity", "delta_tm", "sequence", "study_id"
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing required column: {c}"
    assert len(df) > 10, "Variant pool is unexpectedly small"
    assert (df["sequence"].str.len() == 290).all(), "Sequence length mismatch with 5XJH mature reference"


def test_structural_proximity_filter():
    # Wildtype is always eligible
    elig_wt, d_wt = is_eligible_for_chemistry("WT", radius_angstrom=8.0)
    assert elig_wt is True
    assert d_wt == 0.0

    # Catalytic residue W159H is within 8.0 A
    elig_159, d_159 = is_eligible_for_chemistry("W159H", radius_angstrom=8.0)
    assert elig_159 is True
    assert d_159 < 8.0

    # Distant surface mutation (e.g. pos 280 or pos 98)
    d_98 = calculate_variant_min_distance("K98A")
    assert d_98 > 8.0
