"""ProjectSnapshotAdapter: creates isolated, immutable, read-only scientific snapshots
of existing Q-Catalyst outputs without modifying source files or importing qcatalyst.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import pandas as pd


class ProjectSnapshotAdapter:
    """
    Manages snapshot creation and validation for pet_degradation_lab.
    Preserves raw data, calculates SHA-256 hashes, and generates standardized manifests.
    """

    DEFAULT_ALLOWLIST = [
        "results/rankings/role1_candidate_gate.json",
        "results/rankings/role1_candidate_gate.parquet",
        "results/rankings/ranking.parquet",
        "results/predictions/predictions.parquet",
        "results/predictions/esm1v_scores.parquet",
        "results/uncertainty/uncertainty.parquet",
        "results/uncertainty/gate0_report.json",
        "results/chemistry/chem_results.parquet",
        "results/chemistry/chem_jobs.json",
        "results/chemistry/tier_s_cluster.xyz",
        "results/chemistry/fast_petase_directional_check.json",
        "results/quantum/casci_control_role3.json",
        "results/quantum/quantum_results.parquet",
        "data/curated/variants.parquet",
        "data/structures/5XJH.pdb",
        "data/structures/7SH6.pdb",
        "data/structures/variants",
        "reports/gate0_validation_report.md",
        "reports/benchmark_report.md",
        "reports/final_report.md",
    ]

    def __init__(
        self,
        source_root: Path,
        snapshots_dir: Path,
        allowlist: Optional[List[str]] = None,
    ):
        self.source_root = source_root.resolve()
        self.snapshots_dir = snapshots_dir.resolve()
        self.allowlist = allowlist or list(self.DEFAULT_ALLOWLIST)

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def generate_snapshot_id(self) -> str:
        now = datetime.now(timezone.utc)
        return f"snap_{now.strftime('%Y%m%d_%H%M%S')}"

    def plan_snapshot(
        self,
        snapshot_id: str,
        include_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates execution plan for snapshot creation (used for --dry-run and execution).
        """
        target_dir = self.snapshots_dir / snapshot_id
        patterns = include_patterns or self.allowlist

        files_to_copy = []
        missing_sources = []
        excluded_sources = []

        for rel_pat in patterns:
            src_path = self.source_root / rel_pat
            if not src_path.exists():
                missing_sources.append(rel_pat)
                continue

            if src_path.is_dir():
                subfiles = sorted(list(src_path.glob("*.pdb")))
                for sf in subfiles:
                    sub_rel = sf.relative_to(self.source_root)
                    files_to_copy.append({
                        "source_rel": str(sub_rel),
                        "source_abs": str(sf),
                        "size_bytes": sf.stat().st_size,
                        "is_dir_item": True,
                    })
            else:
                files_to_copy.append({
                    "source_rel": rel_pat,
                    "source_abs": str(src_path),
                    "size_bytes": src_path.stat().st_size,
                    "is_dir_item": False,
                })

        return {
            "snapshot_id": snapshot_id,
            "target_dir": str(target_dir),
            "destination_exists": target_dir.exists(),
            "files_to_copy": files_to_copy,
            "total_files": len(files_to_copy),
            "total_bytes": sum(f["size_bytes"] for f in files_to_copy),
            "missing_sources": missing_sources,
            "excluded_sources": excluded_sources,
        }

    def create_snapshot(
        self,
        snapshot_id: Optional[str] = None,
        include_patterns: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes snapshot creation or dry run.
        Never overwrites existing snapshots.
        """
        snap_id = snapshot_id or self.generate_snapshot_id()
        plan = self.plan_snapshot(snap_id, include_patterns)

        if dry_run:
            plan["status"] = "DRY_RUN_SUCCESS"
            return plan

        target_dir = Path(plan["target_dir"])
        if target_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing snapshot directory: {target_dir}. "
                "Snapshot directories must be immutable."
            )

        raw_dir = target_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=False)

        copied_records = []
        copied_checksums: Dict[str, str] = {}

        # 1. Copy files into raw/ preserving relative structure
        for item in plan["files_to_copy"]:
            src_file = Path(item["source_abs"])
            rel_path = item["source_rel"]
            dst_file = raw_dir / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src_file, dst_file)
            src_hash = self.compute_sha256(src_file)
            dst_hash = self.compute_sha256(dst_file)

            if src_hash != dst_hash:
                raise ValueError(
                    f"Checksum mismatch for {rel_path}: src={src_hash} vs dst={dst_hash}"
                )

            copied_checksums[rel_path] = dst_hash
            copied_records.append({
                "source_relative_path": rel_path,
                "snapshot_relative_path": f"raw/{rel_path}",
                "size_bytes": item["size_bytes"],
                "sha256": dst_hash,
                "copied_timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # 2. Compile manifests from the copied snapshot files
        manifests = self._build_manifests(target_dir, raw_dir, copied_checksums, snap_id)

        # 3. Write all manifest files
        for m_name, m_data in manifests.items():
            m_path = target_dir / m_name
            with open(m_path, "w", encoding="utf-8") as f:
                json.dump(m_data, f, indent=2)

        plan["status"] = "SNAPSHOT_CREATED"
        plan["manifests_written"] = list(manifests.keys())
        return plan

    def _build_manifests(
        self,
        snapshot_dir: Path,
        raw_dir: Path,
        checksums: Dict[str, str],
        snapshot_id: str,
    ) -> Dict[str, Any]:
        """
        Parses copied files to build standardized scientific manifests.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # Base snapshot manifest
        snapshot_manifest = {
            "snapshot_id": snapshot_id,
            "created_timestamp": now_iso,
            "source_root": str(self.source_root),
            "files_copied_count": len(checksums),
            "files": checksums,
        }

        # Candidate manifest & ranking
        ranking_path = raw_dir / "results/rankings/ranking.parquet"
        variants_path = raw_dir / "data/curated/variants.parquet"
        chem_path = raw_dir / "results/chemistry/chem_results.parquet"
        quantum_path = raw_dir / "results/quantum/quantum_results.parquet"
        casci_path = raw_dir / "results/quantum/casci_control_role3.json"
        pred_path = raw_dir / "results/predictions/predictions.parquet"
        esm1v_path = raw_dir / "results/predictions/esm1v_scores.parquet"
        unc_path = raw_dir / "results/uncertainty/uncertainty.parquet"

        candidates_list = []
        sequence_list = []
        structure_list = []
        structural_risk_list = []
        dft_list = []
        quantum_list = []

        # Read ranking.parquet
        if ranking_path.exists():
            df_rank = pd.read_parquet(ranking_path)
            for _, r in df_rank.iterrows():
                vid = str(r["variant_id"])
                candidates_list.append({
                    "variant_id": vid,
                    "parent_enzyme": str(r.get("parent_enzyme", "IsPETase")),
                    "mutations_raw": str(r.get("mutations", "")),
                    "mutation_count": int(r.get("mutation_count", 0)),
                    "final_rank": int(r.get("final_rank", 0)),
                    "confidence_tier": str(r.get("confidence_tier", "Unassigned")),
                    "is_prior_art": bool(r.get("is_prior_art", False)),
                    "prior_art_name": str(r.get("prior_art_name", "")) if pd.notna(r.get("prior_art_name")) else None,
                    "pred_activity": float(r.get("pred_activity", 0.0)) if pd.notna(r.get("pred_activity")) else None,
                    "pred_delta_tm": float(r.get("pred_delta_tm", 0.0)) if pd.notna(r.get("pred_delta_tm")) else None,
                    "aggregate_uncertainty": float(r.get("aggregate_uncertainty", 0.0)) if pd.notna(r.get("aggregate_uncertainty")) else None,
                    "chem_supported": bool(r.get("chem_supported", False)),
                    "quantum_supported": bool(r.get("quantum_supported", False)),
                    "raw_fields": r.to_dict(),
                })

                # Sequence if available
                seq_val = r.get("sequence")
                if pd.notna(seq_val) and str(seq_val).strip():
                    seq_str = str(seq_val).strip().upper()
                    sequence_list.append({
                        "variant_id": vid,
                        "sequence": seq_str,
                        "sequence_length": len(seq_str),
                        "sha256": hashlib.sha256(seq_str.encode("utf-8")).hexdigest(),
                        "source": "results/rankings/ranking.parquet",
                    })

                # Structural model if file exists
                pdb_variant_path = raw_dir / "data/structures/variants" / f"{vid}.pdb"
                if pdb_variant_path.exists():
                    structure_list.append({
                        "variant_id": vid,
                        "model_id": f"PDB_{vid}",
                        "file_path": f"raw/data/structures/variants/{vid}.pdb",
                        "sha256": checksums.get(f"data/structures/variants/{vid}.pdb", self.compute_sha256(pdb_variant_path)),
                        "file_size": pdb_variant_path.stat().st_size,
                        "structure_source": "VARIANT_RELAXED_PDB",
                    })

                # Structural risk
                s_risk = r.get("structural_risk")
                d_att = r.get("d_attack_angstrom")
                structural_risk_list.append({
                    "variant_id": vid,
                    "structural_risk_score": float(s_risk) if pd.notna(s_risk) else None,
                    "active_site_proximity_angstrom": float(d_att) if pd.notna(d_att) else None,
                    "source": "results/rankings/ranking.parquet",
                })

        # DFT calculations from chem_results.parquet
        if chem_path.exists():
            df_chem = pd.read_parquet(chem_path)
            for _, r in df_chem.iterrows():
                vid = str(r["variant_id"])
                dft_list.append({
                    "calculation_id": f"CHEM_{vid}",
                    "variant_id": vid,
                    "mutation": str(r.get("mutation", "")),
                    "arm": str(r.get("arm", "")),
                    "e_hf_hartree": float(r["e_hf"]) if pd.notna(r.get("e_hf")) else None,
                    "e_dft_hartree": float(r["e_dft"]) if pd.notna(r.get("e_dft")) else None,
                    "esp_at_ser_og_kcal_mol": float(r["esp_at_ser_og_kcal_mol"]) if pd.notna(r.get("esp_at_ser_og_kcal_mol")) else None,
                    "delta_esp_kcal_mol": float(r["delta_esp_kcal_mol"]) if pd.notna(r.get("delta_esp_kcal_mol")) else None,
                    "charge_ser_og": float(r["charge_ser_og"]) if pd.notna(r.get("charge_ser_og")) else None,
                    "delta_charge_ser_og": float(r["delta_charge_ser_og"]) if pd.notna(r.get("delta_charge_ser_og")) else None,
                    "proton_shuttle_proxy_kcal": float(r["proton_shuttle_proxy_kcal"]) if pd.notna(r.get("proton_shuttle_proxy_kcal")) else None,
                    "approach_barrier_proxy_kcal": float(r["approach_barrier_proxy_kcal"]) if pd.notna(r.get("approach_barrier_proxy_kcal")) else None,
                    "method": "DFT",
                    "functional": str(r.get("functional", "B3LYP")),
                    "basis": str(r.get("basis", "def2-SVP")),
                    "radius_cutoff_A": float(r.get("radius_cutoff_A", 10.0)),
                    "wall_clock_sec": float(r.get("wall_clock_sec", 0.0)),
                })

        # Quantum results from quantum_results.parquet
        if quantum_path.exists():
            df_quant = pd.read_parquet(quantum_path)
            for _, r in df_quant.iterrows():
                quantum_list.append({
                    "variant_id": str(r.get("variant_id")),
                    "mutations": str(r.get("mutations", "")),
                    "active_space": str(r.get("active_space", "(4e, 4o)")),
                    "e_casci_hartree": float(r["e_casci_hartree"]) if pd.notna(r.get("e_casci_hartree")) else None,
                    "e_vqe_hartree": float(r["e_vqe_hartree"]) if pd.notna(r.get("e_vqe_hartree")) else None,
                    "energy_error_hartree": float(r["energy_error_hartree"]) if pd.notna(r.get("energy_error_hartree")) else None,
                    "energy_error_kcal_mol": float(r["energy_error_kcal_mol"]) if pd.notna(r.get("energy_error_kcal_mol")) else None,
                    "noise_level": float(r.get("noise_level", 0.0)),
                    "qubit_count": int(r.get("qubit_count", 8)),
                    "circuit_depth": int(r.get("circuit_depth", 14)),
                    "cnot_gates": int(r.get("cnot_gates", 18)),
                })

        # CASCI reference json
        if casci_path.exists():
            with open(casci_path, "r", encoding="utf-8") as f:
                casci_dict = json.load(f)
        else:
            casci_dict = {}

        return {
            "snapshot_manifest.json": snapshot_manifest,
            "candidate_manifest.json": {
                "snapshot_id": snapshot_id,
                "candidate_count": len(candidates_list),
                "candidates": candidates_list,
            },
            "sequence_manifest.json": {
                "snapshot_id": snapshot_id,
                "sequence_count": len(sequence_list),
                "sequences": sequence_list,
            },
            "structure_manifest.json": {
                "snapshot_id": snapshot_id,
                "structure_count": len(structure_list),
                "structures": structure_list,
            },
            "structural_risk_manifest.json": {
                "snapshot_id": snapshot_id,
                "risk_record_count": len(structural_risk_list),
                "structural_risks": structural_risk_list,
            },
            "dft_manifest.json": {
                "snapshot_id": snapshot_id,
                "dft_calculation_count": len(dft_list),
                "dft_results": dft_list,
            },
            "quantum_manifest.json": {
                "snapshot_id": snapshot_id,
                "quantum_calculation_count": len(quantum_list),
                "casci_controls": casci_dict,
                "vqe_simulations": quantum_list,
            },
        }
