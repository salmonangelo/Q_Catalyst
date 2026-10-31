"""Standalone HTTP server and REST API for the pet_degradation_lab Visualization Interface.
Zero external dependencies; runs entirely on Python standard library and reads from snapshot data.
"""
from __future__ import annotations

import http.server
import json
import os
import socketserver
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

APP_DIR = Path(__file__).resolve().parent
LAB_ROOT = APP_DIR.parent.parent


class LabVisualizationHandler(http.server.SimpleHTTPRequestHandler):
    """Handles API requests and serves static visualization assets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self._send_json({
                "status": "online",
                "module": "pet_degradation_lab",
                "version": "0.1.0",
                "milestone": "Milestone 1 — Architecture & Read-Only Ingestion",
            })
            return

        if path == "/api/snapshots":
            snapshots_dir = LAB_ROOT / "data" / "project_snapshots"
            snaps = []
            if snapshots_dir.exists():
                for p in sorted(snapshots_dir.iterdir(), reverse=True):
                    if p.is_dir() and (p / "snapshot_manifest.json").exists():
                        try:
                            with open(p / "snapshot_manifest.json", "r") as f:
                                meta = json.load(f)
                            snaps.append(meta)
                        except Exception:
                            snaps.append({"snapshot_id": p.name})
            self._send_json({"snapshots": snaps})
            return

        if path == "/api/candidates":
            query = parse_qs(parsed.query)
            snap_id = query.get("snapshot_id", [None])[0]
            snapshots_dir = LAB_ROOT / "data" / "project_snapshots"
            if not snap_id:
                # Get latest
                avail = sorted([p.name for p in snapshots_dir.iterdir() if p.is_dir()], reverse=True)
                snap_id = avail[0] if avail else None

            if not snap_id:
                self._send_json({"error": "No snapshots found. Run create_snapshot.py first."}, status=404)
                return

            snap_path = snapshots_dir / snap_id
            cand_manifest = snap_path / "candidate_manifest.json"
            seq_manifest = snap_path / "sequence_manifest.json"
            struct_manifest = snap_path / "structure_manifest.json"
            dft_manifest = snap_path / "dft_manifest.json"
            quantum_manifest = snap_path / "quantum_manifest.json"

            data = {
                "snapshot_id": snap_id,
                "candidates": [],
                "sequences": {},
                "structures": {},
                "dft": {},
                "quantum": {},
            }

            if cand_manifest.exists():
                with open(cand_manifest) as f:
                    data["candidates"] = json.load(f).get("candidates", [])
            if seq_manifest.exists():
                with open(seq_manifest) as f:
                    for s in json.load(f).get("sequences", []):
                        data["sequences"][s["variant_id"]] = s
            if struct_manifest.exists():
                with open(struct_manifest) as f:
                    for st in json.load(f).get("structures", []):
                        data["structures"][st["variant_id"]] = st
            if dft_manifest.exists():
                with open(dft_manifest) as f:
                    for d in json.load(f).get("dft_results", []):
                        data["dft"][d["variant_id"]] = d
            if quantum_manifest.exists():
                with open(quantum_manifest) as f:
                    q_data = json.load(f)
                    data["quantum"]["controls"] = q_data.get("casci_controls", {})
                    data["quantum"]["simulations"] = q_data.get("vqe_simulations", [])

            self._send_json(data)
            return

        if path.startswith("/api/pdb/"):
            # Serve local PDB file for Mol* viewer
            parts = path.strip("/").split("/")
            if len(parts) >= 3:
                snap_id = parts[2]
                pdb_name = parts[3]
                pdb_file = LAB_ROOT / "data" / "project_snapshots" / snap_id / "raw" / "data" / "structures"
                if (pdb_file / "variants" / pdb_name).exists():
                    target_pdb = pdb_file / "variants" / pdb_name
                elif (pdb_file / pdb_name).exists():
                    target_pdb = pdb_file / pdb_name
                else:
                    self._send_json({"error": f"PDB {pdb_name} not found"}, status=404)
                    return

                self.send_response(200)
                self.send_header("Content-Type", "chemical/x-pdb")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(target_pdb.read_bytes())
                return

        return super().do_GET()

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def run_server(port: int = 8501, host: str = "127.0.0.1") -> None:
    server_address = (host, port)
    with socketserver.TCPServer(server_address, LabVisualizationHandler) as httpd:
        print(f"=== Q-Catalyst PET Degradation Lab Visualisation Server ===")
        print(f"URL: http://{host}:{port}/")
        print(f"Serving standalone interface from: {APP_DIR}")
        print("Press Ctrl+C to terminate.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    run_server(port=p)
