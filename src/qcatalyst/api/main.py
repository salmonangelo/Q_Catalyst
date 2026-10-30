"""FastAPI backend application for Q-Catalyst Role 1 Developer Workbench.

Provides a local, secure REST API for developer inspection and testing across
the 8 protein engineering ML workflow stages.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from qcatalyst.api.routes import audit, status

app = FastAPI(
    title="Q-Catalyst Role 1 Developer Workbench API",
    description="Local development backend for Protein ML Lead testing, auditing, and benchmarking.",
    version="1.0.0",
)

# Enable CORS for local Vite dev server and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount active routes
app.include_router(status.router)
app.include_router(audit.router)


@app.get("/")
def root_info() -> dict[str, str]:
    """Root status indicator."""
    return {
        "service": "Q-Catalyst Role 1 Developer Workbench API",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ensure all unhandled exceptions return human-readable JSON responses."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": str(exc),
            "path": str(request.url),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("qcatalyst.api.main:app", host="127.0.0.1", port=8000, reload=True)
