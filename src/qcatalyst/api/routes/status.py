"""System status, environment diagnostics, and stage readiness endpoint."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/status", tags=["Status"])

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


class EnvironmentInfo(BaseModel):
    python_version: str
    platform: str
    torch_version: str | None
    transformers_version: str | None
    xgboost_version: str | None
    sklearn_version: str | None
    device: str


class StageStatus(BaseModel):
    stage_id: int
    name: str
    status: str  # 'not_implemented', 'ready', 'running', 'completed', 'failed'
    description: str
    implemented: bool


class SystemStatusResponse(BaseModel):
    status: str
    environment: EnvironmentInfo
    raw_data_available: bool
    curated_data_available: bool
    predictions_available: bool
    uncertainty_available: bool
    gate0_available: bool
    benchmarks_available: bool
    stages: list[StageStatus]


@router.get("", response_model=SystemStatusResponse)
def get_system_status() -> SystemStatusResponse:
    """Return system diagnostics, package versions, and stage readiness."""
    # Check package versions safely
    def get_version(mod_name: str) -> str | None:
        try:
            mod = __import__(mod_name)
            return getattr(mod, "__version__", None)
        except Exception:
            return None

    # Check CUDA / Torch
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass

    env = EnvironmentInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        torch_version=get_version("torch"),
        transformers_version=get_version("transformers"),
        xgboost_version=get_version("xgboost"),
        sklearn_version=get_version("sklearn"),
        device=device,
    )

    # Check file availability
    raw_petflow = REPO_ROOT / "data" / "raw" / "petflow"
    raw_pazy = REPO_ROOT / "data" / "raw" / "pazy"
    raw_available = raw_petflow.exists() or raw_pazy.exists()

    curated_path = REPO_ROOT / "data" / "curated" / "variants.parquet"
    predictions_path = REPO_ROOT / "results" / "predictions" / "predictions.parquet"
    uncertainty_path = REPO_ROOT / "results" / "uncertainty" / "uncertainty.parquet"
    gate0_path = REPO_ROOT / "reports" / "tables" / "gate0_verdict.json"
    benchmarks_path = REPO_ROOT / "results" / "benchmarks" / "benchmark_results.parquet"
    audit_manifest_path = REPO_ROOT / "reports" / "tables" / "petgym_file_manifest.csv"

    # Define stage states
    stages = [
        StageStatus(
            stage_id=1,
            name="Data Audit and Cleaning",
            status="completed" if audit_manifest_path.exists() else "ready",
            description="Audit raw PET-Gym and PAZy files, manifest generation, anomaly detection",
            implemented=True,
        ),
        StageStatus(
            stage_id=2,
            name="Mutation Inspector",
            status="ready" if curated_path.exists() else "not_implemented",
            description="Notation validation, coordinate parsing, 5XJH crystal alignment",
            implemented=True,
        ),
        StageStatus(
            stage_id=3,
            name="ESM-1v Scoring",
            status="ready",
            description="Zero-shot masked-marginal mutation effect scoring",
            implemented=True,
        ),
        StageStatus(
            stage_id=4,
            name="ESM-2 Embeddings",
            status="ready",
            description="Pretrained sequence embedding extraction with SHA-256 caching",
            implemented=True,
        ),
        StageStatus(
            stage_id=5,
            name="XGBoost Prediction",
            status="completed" if predictions_path.exists() else "ready",
            description="Supervised activity/Tm prediction with frozen generalization splits",
            implemented=True,
        ),
        StageStatus(
            stage_id=6,
            name="Uncertainty Estimation",
            status="completed" if uncertainty_path.exists() else "ready",
            description="Bootstrap ensemble disagreement, conformal intervals, OOD distance",
            implemented=True,
        ),
        StageStatus(
            stage_id=7,
            name="Gate 0 and Benchmarks",
            status="completed" if (gate0_path.exists() and benchmarks_path.exists()) else "ready",
            description="Uncertainty error correlation, bootstrap CI, A0-A2 prioritisation arms",
            implemented=True,
        ),
        StageStatus(
            stage_id=8,
            name="Export Centre",
            status="ready",
            description="Inspect, verify, and export approved reports, figures, and parquet tables",
            implemented=True,
        ),
    ]

    return SystemStatusResponse(
        status="ok",
        environment=env,
        raw_data_available=raw_available,
        curated_data_available=curated_path.exists(),
        predictions_available=predictions_path.exists(),
        uncertainty_available=uncertainty_path.exists(),
        gate0_available=gate0_path.exists(),
        benchmarks_available=benchmarks_path.exists(),
        stages=stages,
    )
