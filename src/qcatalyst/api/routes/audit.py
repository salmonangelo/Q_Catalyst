"""Data audit route providing read-only inspection, live audit execution, and downloads."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from qcatalyst.data.audit import (
    audit_all_csvs,
    generate_readable_summary,
    scan_file_manifest,
)

router = APIRouter(prefix="/api/audit", tags=["Data Audit"])

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
REPORTS_DIR = REPO_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"

ALLOWED_DOWNLOADS = {
    "petgym_file_manifest.csv": TABLES_DIR / "petgym_file_manifest.csv",
    "petgym_csv_audit.csv": TABLES_DIR / "petgym_csv_audit.csv",
    "data_audit_summary.md": REPORTS_DIR / "data_audit_summary.md",
}


class AuditCsvSummary(BaseModel):
    file_path: str
    file_name: str
    dataset_type: str
    enzyme: str
    target_type: str
    status: str
    row_count: int
    col_count: int
    columns: str
    missing_cells: int
    duplicate_rows: int
    duplicate_mutants: int
    min_score: float | None
    max_score: float | None
    anomaly_flags: str
    error_message: str


class AuditDataResponse(BaseModel):
    status: str  # 'completed' or 'not_run'
    raw_dir: str
    total_files: int
    total_csv_files: int
    total_records: int
    manifest_file: str
    manifest_file_size: int
    audit_file: str
    audit_file_size: int
    summary_file: str
    summary_file_size: int
    csv_audits: list[AuditCsvSummary]
    summary_markdown: str


class RunAuditResponse(BaseModel):
    status: str
    runtime_seconds: float
    total_files_scanned: int
    total_csvs_audited: int
    total_measurement_records: int
    manifest_path: str
    manifest_size_bytes: int
    audit_path: str
    audit_size_bytes: int
    summary_path: str
    summary_size_bytes: int
    message: str


@router.get("", response_model=AuditDataResponse)
def get_audit_data() -> AuditDataResponse:
    """Retrieve existing data audit report, manifest stats, and CSV audit entries."""
    manifest_out = TABLES_DIR / "petgym_file_manifest.csv"
    audit_out = TABLES_DIR / "petgym_csv_audit.csv"
    summary_out = REPORTS_DIR / "data_audit_summary.md"

    if not audit_out.exists() or not manifest_out.exists():
        # Audit has not yet been executed
        return AuditDataResponse(
            status="not_run",
            raw_dir=str(RAW_DIR.resolve()),
            total_files=0,
            total_csv_files=0,
            total_records=0,
            manifest_file=str(manifest_out),
            manifest_file_size=0,
            audit_file=str(audit_out),
            audit_file_size=0,
            summary_file=str(summary_out),
            summary_file_size=0,
            csv_audits=[],
            summary_markdown="No audit report found. Click 'Run Data Audit' to inspect raw files.",
        )

    # Read existing audit files
    try:
        manifest_df = pd.read_csv(manifest_out)
        audit_df = pd.read_csv(audit_out)
        summary_md = summary_out.read_text(encoding="utf-8") if summary_out.exists() else ""

        # Format records for response
        csv_audits: list[AuditCsvSummary] = []
        for _, row in audit_df.iterrows():
            csv_audits.append(
                AuditCsvSummary(
                    file_path=str(row.get("file_path", "")),
                    file_name=str(row.get("file_name", "")),
                    dataset_type=str(row.get("dataset_type", "")),
                    enzyme=str(row.get("enzyme", "")),
                    target_type=str(row.get("target_type", "")),
                    status=str(row.get("status", "")),
                    row_count=int(row.get("row_count", 0)),
                    col_count=int(row.get("col_count", 0)),
                    columns=str(row.get("columns", "")),
                    missing_cells=int(row.get("missing_cells", 0)),
                    duplicate_rows=int(row.get("duplicate_rows", 0)),
                    duplicate_mutants=int(row.get("duplicate_mutants", 0)),
                    min_score=float(row["min_score"]) if pd.notnull(row.get("min_score")) else None,
                    max_score=float(row["max_score"]) if pd.notnull(row.get("max_score")) else None,
                    anomaly_flags=str(row.get("anomaly_flags", "none")),
                    error_message=str(row.get("error_message", "")) if pd.notnull(row.get("error_message")) else "",
                )
            )

        total_records = int(audit_df[audit_df["status"] == "ok"]["row_count"].sum())

        return AuditDataResponse(
            status="completed",
            raw_dir=str(RAW_DIR.resolve()),
            total_files=len(manifest_df),
            total_csv_files=len(audit_df),
            total_records=total_records,
            manifest_file=str(manifest_out.resolve()),
            manifest_file_size=manifest_out.stat().st_size if manifest_out.exists() else 0,
            audit_file=str(audit_out.resolve()),
            audit_file_size=audit_out.stat().st_size if audit_out.exists() else 0,
            summary_file=str(summary_out.resolve()),
            summary_file_size=summary_out.stat().st_size if summary_out.exists() else 0,
            csv_audits=csv_audits,
            summary_markdown=summary_md,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read audit files: {str(exc)}")


@router.post("/run", response_model=RunAuditResponse)
def run_data_audit() -> RunAuditResponse:
    """Execute live data audit pipeline on raw data files (strictly local & read-only)."""
    if not RAW_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Raw data directory not found at {RAW_DIR}. Verify repository data/raw folder.",
        )

    start_time = time.time()
    try:
        # 1. Scan file manifest
        manifest_df = scan_file_manifest(RAW_DIR)

        # 2. Audit all CSV files
        csv_rows = manifest_df[manifest_df["extension"] == ".csv"]
        csv_paths = [RAW_DIR / row["file_path"] for _, row in csv_rows.iterrows()]
        audit_df = audit_all_csvs(csv_paths, base_dir=RAW_DIR)

        # 3. Save table outputs
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        manifest_out = TABLES_DIR / "petgym_file_manifest.csv"
        audit_out = TABLES_DIR / "petgym_csv_audit.csv"
        summary_out = REPORTS_DIR / "data_audit_summary.md"

        manifest_df.to_csv(manifest_out, index=False)
        audit_df.to_csv(audit_out, index=False)

        # 4. Generate readable summary markdown
        summary_text = generate_readable_summary(manifest_df, audit_df)
        summary_out.write_text(summary_text, encoding="utf-8")

        elapsed = time.time() - start_time
        total_records = int(audit_df[audit_df["status"] == "ok"]["row_count"].sum())

        return RunAuditResponse(
            status="success",
            runtime_seconds=round(elapsed, 3),
            total_files_scanned=len(manifest_df),
            total_csvs_audited=len(audit_df),
            total_measurement_records=total_records,
            manifest_path=str(manifest_out.resolve()),
            manifest_size_bytes=manifest_out.stat().st_size,
            audit_path=str(audit_out.resolve()),
            audit_size_bytes=audit_out.stat().st_size,
            summary_path=str(summary_out.resolve()),
            summary_size_bytes=summary_out.stat().st_size,
            message="Data audit completed successfully with zero modifications to raw data.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data audit failed: {str(exc)}")


@router.get("/download/{filename}")
def download_audit_report(filename: str):
    """Download approved audit report files with strict directory-traversal prevention."""
    # Strict filename whitelist
    if filename not in ALLOWED_DOWNLOADS:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: '{filename}' is not an authorized audit export. "
                   f"Allowed files: {list(ALLOWED_DOWNLOADS.keys())}",
        )

    file_path = ALLOWED_DOWNLOADS[filename]
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report file '{filename}' does not exist on disk yet. Run the audit first.",
        )

    # Resolve and ensure inside REPO_ROOT
    resolved = file_path.resolve()
    if not str(resolved).startswith(str(REPO_ROOT.resolve())):
        raise HTTPException(status_code=403, detail="Path traversal detected.")

    media_type = "text/csv" if filename.endswith(".csv") else "text/markdown"
    return FileResponse(
        path=resolved,
        filename=filename,
        media_type=media_type,
    )
