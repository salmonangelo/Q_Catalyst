#!/usr/bin/env python3
"""
CLI launcher for the standalone pet_degradation_lab visualization interface.
Operates on Python standard library without touching existing project frontend or API.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from visualization.app.server import run_server


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch pet_degradation_lab standalone visualization dashboard."
    )
    parser.add_argument("--port", type=int, default=8501, help="Port to serve HTTP interface on (default: 8501)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()

    run_server(port=args.port, host=args.host)
    return 0


if __name__ == "__main__":
    sys.exit(main())
