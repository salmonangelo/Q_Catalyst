"""Unit tests for Q-Catalyst data audit module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from qcatalyst.data.audit import (
    audit_all_csvs,
    audit_csv_file,
    compute_sha256,
    generate_readable_summary,
    infer_metadata_from_path,
    scan_file_manifest,
)


@pytest.fixture
def temp_data_env():
    """Create a temporary directory structure mimicking raw datasets."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)

        # 1. Valid single-site CSV
        single_dir = base / "single-site-mutant" / "IsPETase_Act"
        single_dir.mkdir(parents=True)
        csv1 = single_dir / "IsPETase_Act.csv"
        csv1.write_text("mutant,score\nN37P,1.0\nA40P,0.5\n", encoding="utf-8")
        fasta1 = single_dir / "IsPETase_Act.fasta"
        fasta1.write_text(">IsPETase\nMNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARG\n", encoding="utf-8")

        # 2. CSV with duplicate mutant and Tm outlier
        multi_dir = base / "low-to-high-mutant" / "BhrPETase_Tm"
        multi_dir.mkdir(parents=True)
        csv2 = multi_dir / "BhrPETase_Tm.csv"
        csv2.write_text(
            "mutant,score\nW104L,85.5\nH218S:F222I:H191D,88.0\nH218S:F222I:H191D,88.0\nH218S:F222I:F243T,842.0\n",
            encoding="utf-8",
        )

        # 3. Empty CSV (0 bytes)
        empty_dir = base / "single-site-mutant" / "Empty_Act"
        empty_dir.mkdir(parents=True)
        csv_empty = empty_dir / "Empty_Act.csv"
        csv_empty.write_text("", encoding="utf-8")

        # 4. Malformed CSV (ragged lines causing ParserError)
        malformed_dir = base / "single-site-mutant" / "Malformed_Act"
        malformed_dir.mkdir(parents=True)
        csv_malformed = malformed_dir / "Malformed_Act.csv"
        csv_malformed.write_text("mutant,score\nN37P,1.0\nextra,fields,detected,here,now\n", encoding="utf-8")

        yield base


def test_infer_metadata():
    dataset_type, enzyme, target_type = infer_metadata_from_path(
        "single-site-mutant/IsPETase_Act/IsPETase_Act.csv"
    )
    assert dataset_type == "single-site-mutant"
    assert enzyme == "IsPETase"
    assert target_type == "Act"

    dataset_type, enzyme, target_type = infer_metadata_from_path(
        "low-to-high-mutant/mutant/BhrPETase_Tm/BhrPETase_Tm.csv"
    )
    assert dataset_type == "low-to-high-mutant"
    assert enzyme == "BhrPETase"
    assert target_type == "Tm"


def test_scan_file_manifest(temp_data_env):
    manifest = scan_file_manifest(temp_data_env)
    assert len(manifest) == 5  # csv1, fasta1, csv2, csv_empty, csv_malformed
    assert set(manifest["extension"].unique()) == {".csv", ".fasta"}

    empty_row = manifest[manifest["file_name"] == "Empty_Act.csv"].iloc[0]
    assert bool(empty_row["is_empty"]) is True
    assert empty_row["size_bytes"] == 0

    valid_row = manifest[manifest["file_name"] == "IsPETase_Act.csv"].iloc[0]
    assert not valid_row["is_empty"]
    assert valid_row["size_bytes"] > 0
    assert len(valid_row["sha256"]) == 64


def test_audit_valid_csv(temp_data_env):
    csv_path = temp_data_env / "single-site-mutant" / "IsPETase_Act" / "IsPETase_Act.csv"
    res = audit_csv_file(csv_path, base_dir=temp_data_env)

    assert res.status == "ok"
    assert res.row_count == 2
    assert res.col_count == 2
    assert "mutant" in res.columns
    assert "score" in res.columns
    assert res.min_score == 0.5
    assert res.max_score == 1.0
    assert res.duplicate_rows == 0
    assert res.duplicate_mutants == 0
    assert res.anomaly_flags == "none"


def test_audit_empty_csv(temp_data_env):
    csv_path = temp_data_env / "single-site-mutant" / "Empty_Act" / "Empty_Act.csv"
    res = audit_csv_file(csv_path, base_dir=temp_data_env)

    assert res.status == "empty"
    assert res.row_count == 0
    assert "0 bytes" in res.error_message or "EmptyDataError" in res.error_message


def test_audit_malformed_csv(temp_data_env):
    csv_path = temp_data_env / "single-site-mutant" / "Malformed_Act" / "Malformed_Act.csv"
    res = audit_csv_file(csv_path, base_dir=temp_data_env)

    assert res.status == "malformed"
    assert "Parser failure" in res.error_message


def test_audit_anomalies_and_duplicates(temp_data_env):
    csv_path = temp_data_env / "low-to-high-mutant" / "BhrPETase_Tm" / "BhrPETase_Tm.csv"
    res = audit_csv_file(csv_path, base_dir=temp_data_env)

    assert res.status == "ok"
    assert res.row_count == 4
    assert res.duplicate_mutants == 1
    assert "1_duplicate_mutants" in res.anomaly_flags
    assert "unphysical_high_tm(max=842.0)" in res.anomaly_flags


def test_audit_idempotency_and_immutability(temp_data_env):
    """Ensure running the audit multiple times never alters input files."""
    csv_path = temp_data_env / "single-site-mutant" / "IsPETase_Act" / "IsPETase_Act.csv"
    initial_sha = compute_sha256(csv_path)

    # Run multiple times
    _ = audit_csv_file(csv_path, base_dir=temp_data_env)
    _ = audit_csv_file(csv_path, base_dir=temp_data_env)
    _ = scan_file_manifest(temp_data_env)

    final_sha = compute_sha256(csv_path)
    assert initial_sha == final_sha


def test_generate_readable_summary(temp_data_env):
    manifest = scan_file_manifest(temp_data_env)
    csv_paths = [temp_data_env / p for p in manifest[manifest["extension"] == ".csv"]["file_path"]]
    audit_df = audit_all_csvs(csv_paths, base_dir=temp_data_env)

    summary = generate_readable_summary(manifest, audit_df)
    assert "# Q-Catalyst Raw Data Audit" in summary
    assert "Total Inventoried Files" in summary
    assert "Zero FASTA Mismatches" in summary
