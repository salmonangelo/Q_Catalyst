"""Source inventory scanner for Q-Catalyst repository artifacts.
Produces pet_degradation_lab/reports/source_inventory.md.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


class SourceInventoryScanner:
    """
    Inspects existing project outputs in read-only mode and compiles
    a structured scientific inventory with provenance and data classifications.
    """

    CANDIDATE_SOURCES = [
        ("results/rankings/role1_candidate_gate.json", "JSON", "PREDICTED", "Role 1 Candidate Acquisition Gate Summary"),
        ("results/rankings/role1_candidate_gate.parquet", "Parquet", "PREDICTED", "Held-out candidates with gate annotation & uncertainty"),
        ("results/rankings/ranking.parquet", "Parquet", "PREDICTED / REPORTED", "Final 26 prioritized candidates with confidence tiers"),
        ("results/predictions/predictions.parquet", "Parquet", "PREDICTED", "Supervised XGBoost & Ridge predictions on Split v2"),
        ("results/predictions/esm1v_scores.parquet", "Parquet", "PREDICTED", "Real ESM-1v 650M masked-marginal zero-shot LLR scores"),
        ("results/uncertainty/uncertainty.parquet", "Parquet", "PREDICTED", "Multi-signal uncertainty and split-conformal intervals"),
        ("results/uncertainty/gate0_report.json", "JSON", "COMPUTED", "Gate 0 bootstrap validation verdict and statistical breakdown"),
        ("results/chemistry/chem_results.parquet", "Parquet", "COMPUTED", "31 Tier S cluster QM/MM DFT/RHF calculations & descriptors"),
        ("results/chemistry/chem_jobs.json", "JSON", "REPORTED", "List of 31 budgeted chemistry variant targets"),
        ("results/chemistry/tier_s_cluster.xyz", "XYZ", "COMPUTED", "42-atom catalytic cluster Cartesian geometry"),
        ("results/chemistry/fast_petase_directional_check.json", "JSON", "COMPUTED", "5XJH vs 7SH6 catalytic triad directional comparison"),
        ("results/quantum/casci_control_role3.json", "JSON", "COMPUTED", "Exact classical CASCI(4e, 4o) ground truth for Arm A5"),
        ("results/quantum/quantum_results.parquet", "Parquet", "COMPUTED", "Simulated VQE vs CASCI parity, noise sweeps & resources"),
        ("data/curated/variants.parquet", "Parquet", "EXPERIMENTAL / REPORTED", "430 curated PETase variants aligned to 5XJH"),
        ("data/structures/5XJH.pdb", "PDB", "EXPERIMENTAL", "High-resolution (1.5Å) crystal structure of native IsPETase"),
        ("data/structures/7SH6.pdb", "PDB", "EXPERIMENTAL", "Crystal structure of engineered FAST-PETase"),
        ("data/structures/variants", "Directory (PDBs)", "COMPUTED", "26 mutant structural models for prioritized candidates"),
        ("reports/gate0_validation_report.md", "Markdown", "REPORTED", "Detailed narrative of Gate 0 uncertainty failure and fallback"),
        ("reports/benchmark_report.md", "Markdown", "REPORTED", "Prioritization performance across Arms A0, A1, A2"),
        ("reports/final_report.md", "Markdown", "REPORTED", "Role 2 Computational Chemistry Lead final report"),
    ]

    def __init__(self, source_root: Path):
        self.source_root = source_root.resolve()

    def scan(self) -> List[Dict[str, Any]]:
        records = []
        for rel_path, fmt, evidence_class, desc in self.CANDIDATE_SOURCES:
            p = self.source_root / rel_path
            rec: Dict[str, Any] = {
                "relative_path": rel_path,
                "description": desc,
                "file_type": fmt,
                "evidence_class": evidence_class,
                "exists": p.exists(),
                "size_bytes": 0,
                "sha256": "N/A",
                "usable": False,
                "summary": "",
                "missing_info": "",
                "ambiguities": "",
            }

            if not p.exists():
                rec["missing_info"] = "File or directory does not exist at specified path"
                records.append(rec)
                continue

            rec["usable"] = True

            if p.is_dir():
                files = list(p.glob("*.pdb"))
                total_size = sum(f.stat().st_size for f in files)
                rec["size_bytes"] = total_size
                rec["summary"] = f"Directory containing {len(files)} PDB structure files"
                rec["sha256"] = f"{len(files)} files"
                records.append(rec)
                continue

            size = p.stat().st_size
            rec["size_bytes"] = size
            try:
                rec["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            except Exception as e:
                rec["sha256"] = f"Error: {e}"

            ext = p.suffix.lower()
            if ext == ".parquet":
                try:
                    df = pd.read_parquet(p)
                    rec["summary"] = f"{len(df)} rows, {len(df.columns)} columns"
                    rec["columns"] = df.columns.tolist()
                    if "variant_id" not in df.columns and "variant_id_x" in df.columns:
                        rec["ambiguities"] = "Uses merged column names (e.g. variant_id_x)"
                except Exception as e:
                    rec["summary"] = f"Parquet read error: {e}"
                    rec["usable"] = False
            elif ext == ".json":
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    if isinstance(d, list):
                        rec["summary"] = f"List of {len(d)} items"
                    elif isinstance(d, dict):
                        rec["summary"] = f"Dictionary with {len(d)} top-level keys: {list(d.keys())[:5]}"
                except Exception as e:
                    rec["summary"] = f"JSON parse error: {e}"
                    rec["usable"] = False
            elif ext == ".pdb":
                try:
                    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
                    atom_lines = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
                    rec["summary"] = f"PDB format with {len(lines)} lines ({len(atom_lines)} coordinate atoms)"
                except Exception as e:
                    rec["summary"] = f"PDB read error: {e}"
            elif ext in [".md", ".xyz"]:
                rec["summary"] = f"Text document ({size} bytes)"

            records.append(rec)

        return records

    def write_markdown_report(self, records: List[Dict[str, Any]], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Q-Catalyst Project Artifact Source Inventory\n\n")
            f.write(f"- **Source Root Evaluated:** `{self.source_root}`\n")
            f.write(f"- **Total Candidate Sources Audited:** {len(records)}\n")
            f.write(f"- **Available & Usable Sources:** {sum(1 for r in records if r['usable'])}\n")
            f.write(f"- **Missing Sources:** {sum(1 for r in records if not r['exists'])}\n\n")

            f.write("## 1. Discovered Source Artifacts Table\n\n")
            f.write("| Relative Path | Format | Size (Bytes) | Evidence Class | Usable? | Summary & Findings |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|:---|\n")

            for r in records:
                path_str = f"`{r['relative_path']}`"
                size_str = f"{r['size_bytes']:,}" if r['size_bytes'] else "—"
                usable_str = "✅ YES" if r['usable'] else "❌ NO"
                f.write(f"| {path_str} | {r['file_type']} | {size_str} | `{r['evidence_class']}` | {usable_str} | {r['summary']} |\n")

            f.write("\n## 2. In-Depth Artifact Analysis & Schema Observations\n\n")

            for r in records:
                if not r["usable"]:
                    continue
                f.write(f"### `{r['relative_path']}`\n")
                f.write(f"- **Description:** {r['description']}\n")
                f.write(f"- **Evidence Classification:** `{r['evidence_class']}`\n")
                f.write(f"- **SHA-256:** `{r['sha256']}`\n")
                f.write(f"- **Summary:** {r['summary']}\n")
                if r.get("ambiguities"):
                    f.write(f"- **⚠️ Schema Ambiguity:** {r['ambiguities']}\n")
                if r.get("missing_info"):
                    f.write(f"- **Missing Information:** {r['missing_info']}\n")
                f.write("\n")

            f.write("## 3. Critical Scientific Inferences and Boundaries\n\n")
            f.write("1. **Sequence vs Structural Numbering:** In `variants.parquet`, mutations use mature protein indexing (residues 37–290 for IsPETase), corresponding to 5XJH crystallographic numbering. Residue numbers must be checked explicitly against sequence indices.\n")
            f.write("2. **Chemistry Coverage:** Only 31 variants (1 WT + 15 Arm A4 + 15 Arm A3) have PySCF DFT cluster calculations. Other variants in `ranking.parquet` lack QM data and must remain marked `NOT_AVAILABLE`.\n")
            f.write("3. **Quantum Coverage:** Only 5 candidate variants plus WT possess exact CASCI(4e, 4o) keys. VQE agreement benchmarks simulation accuracy, not experimental PET turnover.\n")
            f.write(r"4. **Gate 0 Uncertainty Validation:** Evaluated on held-out multi-site mutants and failed honestly ($\rho = -0.3889 \le 0$). Diversity fallback is active." + "\n")
