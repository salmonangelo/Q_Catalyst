"""Unit and integration tests for FastAPI backend (Status, Audit, Security)."""

import pytest
from starlette.testclient import TestClient

from qcatalyst.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root info endpoint responds with online status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data


def test_status_endpoint():
    """Verify /api/status returns environment diagnostics and all 8 stages."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert "environment" in data
    assert "python_version" in data["environment"]
    assert "device" in data["environment"]

    stages = data["stages"]
    assert len(stages) == 8
    stage_names = [s["name"] for s in stages]
    assert "Data Audit and Cleaning" in stage_names
    assert "Mutation Inspector" in stage_names
    assert "Gate 0 and Benchmarks" in stage_names

    # Check that stage 1 is marked implemented
    stage_1 = next(s for s in stages if s["stage_id"] == 1)
    assert stage_1["implemented"] is True


def test_audit_get_endpoint():
    """Verify /api/audit returns manifest counts, records, and markdown summary."""
    response = client.get("/api/audit")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ["completed", "not_run"]
    assert "raw_dir" in data
    assert "csv_audits" in data
    assert "summary_markdown" in data

    if data["status"] == "completed":
        assert data["total_files"] > 0
        assert data["total_records"] > 0
        assert len(data["csv_audits"]) > 0


def test_audit_run_endpoint():
    """Verify /api/audit/run executes data audit and returns real results."""
    response = client.post("/api/audit/run")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["total_files_scanned"] >= 50
    assert data["total_csvs_audited"] >= 10
    assert data["total_measurement_records"] > 0
    assert data["runtime_seconds"] > 0.0
    assert "petgym_file_manifest.csv" in data["manifest_path"]
    assert "petgym_csv_audit.csv" in data["audit_path"]


def test_audit_download_authorized():
    """Verify authorized report downloads succeed with appropriate content type."""
    response = client.get("/api/audit/download/petgym_csv_audit.csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert b"file_path" in response.content

    response_md = client.get("/api/audit/download/data_audit_summary.md")
    assert response_md.status_code == 200
    assert "text/markdown" in response_md.headers["content-type"]


def test_audit_download_path_traversal_blocked():
    """Verify directory traversal and unauthorized filenames are strictly blocked (403)."""
    # Directory traversal attempt
    response = client.get("/api/audit/download/../../secret.txt")
    assert response.status_code in [403, 404]

    # Unauthorized file not in whitelist
    response = client.get("/api/audit/download/passwords.txt")
    assert response.status_code == 403
    assert "Access denied" in response.json()["detail"]
