"""Tests verifying raw data contracts and curated variants table."""

from pathlib import Path
import pandas as pd
import pytest

from qcatalyst.data.audit import audit_all_csvs, scan_file_manifest
from qcatalyst.data.curation import build_and_save_variants_table

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
CURATED_DIR = REPO_ROOT / "data" / "curated"


def test_real_raw_directory_manifest():
    """Verify inventory of real raw files in data/raw."""
    assert RAW_DIR.exists(), f"Raw data directory does not exist: {RAW_DIR}"
    manifest = scan_file_manifest(RAW_DIR)

    # We expect PET-Gym files and PAZy files
    assert len(manifest) >= 30, f"Expected at least 30 raw files, got {len(manifest)}"

    # Check presence of expected extensions
    extensions = set(manifest["extension"].unique())
    assert ".csv" in extensions
    assert ".fasta" in extensions
    assert ".pdb" in extensions

    # Verify every file has non-empty sha256
    assert (manifest["sha256"].str.len() == 64).all()


def test_real_petgym_csv_audit():
    """Verify all real PET-Gym CSVs parse successfully and detect expected characteristics."""
    petgym_dir = RAW_DIR / "petflow" / "petflow" / "PET-Gym"
    assert petgym_dir.exists(), f"PET-Gym directory not found at {petgym_dir}"

    manifest = scan_file_manifest(petgym_dir)
    csv_rows = manifest[manifest["extension"] == ".csv"]
    assert len(csv_rows) == 13, f"Expected 13 PET-Gym CSVs, found {len(csv_rows)}"

    csv_paths = [petgym_dir / p for p in csv_rows["file_path"]]
    audit_df = audit_all_csvs(csv_paths, base_dir=petgym_dir)

    # Every CSV must parse successfully (status == 'ok')
    assert (audit_df["status"] == "ok").all()

    # Verify column structure for PET-Gym CSVs
    for _, row in audit_df.iterrows():
        assert "mutant" in row["columns"]
        assert "score" in row["columns"]
        assert row["row_count"] > 0
        assert row["missing_cells"] == 0

    # Verify known anomalies detected in BhrPETase
    bhr_act_multi = audit_df[audit_df["file_name"] == "BhrPETase_Act.csv"]
    assert any("duplicate_mutants" in flags for flags in bhr_act_multi["anomaly_flags"])

    bhr_tm_multi = audit_df[
        (audit_df["file_name"] == "BhrPETase_Tm.csv") & (audit_df["dataset_type"] == "low-to-high-mutant")
    ]
    assert len(bhr_tm_multi) == 1
    assert "unphysical_high_tm" in bhr_tm_multi.iloc[0]["anomaly_flags"]


def test_curated_variants_parquet_contract():
    """Verify schema, contracts, and measurements in data/curated/variants.parquet."""
    parquet_path = CURATED_DIR / "variants.parquet"
    if not parquet_path.exists():
        # Build if not yet present
        df = build_and_save_variants_table(RAW_DIR, parquet_path)
    else:
        df = pd.read_parquet(parquet_path, engine="pyarrow")

    assert len(df) == 430, f"Expected 430 total curated variants, found {len(df)}"

    required_columns = [
        "row_id",
        "variant_id",
        "parent_enzyme",
        "dataset_split",
        "study_id",
        "source_files",
        "mutant_raw",
        "mutations_canonical",
        "num_mutations",
        "is_multi_mutant",
        "wildtype_sequence",
        "mutant_sequence",
        "sequence_length",
        "parent_mutation_positions",
        "structure_5xjh_positions",
        "structure_mapping_status",
        "activity_raw",
        "activity_unit",
        "tm_raw",
        "tm_unit",
        "is_duplicate",
        "is_outlier",
        "data_quality_flags",
    ]

    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"

    # Verify stable identifiers
    assert df["row_id"].is_unique, "row_id must be strictly unique"
    assert (df["variant_id"].str.len() > 0).all()
    assert (df["study_id"] == "PET-Gym").all()

    # Verify sequence lengths match
    assert (df["mutant_sequence"].str.len() == df["sequence_length"]).all()
    assert (df["wildtype_sequence"].str.len() == df["sequence_length"]).all()

    # Verify separate activity and Tm fields
    act_series = df["activity_raw"]
    tm_series = df["tm_raw"]

    # At least some rows have activity, some have Tm, some have both
    assert act_series.notnull().sum() > 0
    assert tm_series.notnull().sum() > 0
    assert (act_series.notnull() & tm_series.notnull()).sum() > 0

    # KubuPETase has Tm but no activity
    kubu_rows = df[df["parent_enzyme"] == "KubuPETase"]
    assert len(kubu_rows) == 12
    assert kubu_rows["activity_raw"].isnull().all()
    assert kubu_rows["tm_raw"].notnull().all()

    # Structure mapping: IsPETase positions must map to 5XJH
    ispet_singles = df[(df["parent_enzyme"] == "IsPETase") & (~df["is_multi_mutant"])]
    assert (ispet_singles["structure_mapping_status"] == "exact_1to1").all()

    # Verify duplicate flag is present on the duplicate row
    dup_rows = df[df["is_duplicate"]]
    assert len(dup_rows) == 1
    assert "H218S:F222I:H191D" in dup_rows.iloc[0]["mutant_raw"]

    # Verify outlier flag is present on the 842.0 C row
    outlier_rows = df[df["is_outlier"]]
    assert len(outlier_rows) == 1
    assert outlier_rows.iloc[0]["tm_raw"] == 842.0
