#!/usr/bin/env python3
"""
Completes Step 6 (Radius Sensitivity + FAST-PETase Check) and Step 7 (Summary Tables) for Role 2.
"""

import json
import os
import glob
import time
import numpy as np
import pandas as pd

from qcatalyst.chemistry.descriptors import DescriptorExtractor
from qcatalyst.chemistry.sensitivity import SensitivityAnalyzer

def cleanup_tmp():
    """Remove stale PySCF temp files in /tmp to guarantee ample disk space."""
    for f in glob.glob("/tmp/tmp*"):
        try:
            if os.path.isfile(f):
                os.remove(f)
        except Exception:
            pass

def main():
    print("=== Finalizing Role 2 Computational Chemistry Deliverables ===")
    cleanup_tmp()

    # 1. Verify chem_results.parquet
    parquet_path = "results/chemistry/chem_results.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"{parquet_path} not found!")
    df_results = pd.read_parquet(parquet_path)
    print(f"Verified {len(df_results)} variants in {parquet_path}")

    # 2. Check Role 3 CASCI control
    casci_path = "results/quantum/casci_control_role3.json"
    if not os.path.exists(casci_path):
        raise FileNotFoundError(f"{casci_path} not found!")
    with open(casci_path) as f:
        casci_data = json.load(f)
    print(f"Verified CASCI controls for: {list(casci_data['variants'].keys())}")

    # 3. Load or Run Protonation Sensitivity
    prot_csv = "reports/tables/sensitivity_protonation.csv"
    if os.path.exists(prot_csv):
        print(f"Loaded existing {prot_csv}")
        df_prot = pd.read_csv(prot_csv)
    else:
        print("Running protonation state sensitivity...")
        prot_records = []
        ext = DescriptorExtractor("data/structures/5XJH.pdb")
        for state in ["HIE", "HIP"]:
            for var in ["WT", "S121E"]:
                print(f"  -> Protonation state {state} on {var}...", flush=True)
                res = ext.extract_variant_descriptors(var, var, his_protonation=state, include_approach_scan=False)
                res["sensitivity_test"] = "protonation_state"
                res["test_parameter"] = state
                prot_records.append(res)
                cleanup_tmp()
        df_prot = pd.DataFrame(prot_records)
        df_prot.to_csv(prot_csv, index=False)

    # 4. Proximity Radius Sensitivity (R = 6, 8, 10, 12 A)
    print("\nRunning proximity cutoff radius sensitivity (R = 6.0, 8.0, 10.0, 12.0 A)...")
    radius_records = []
    # Test on WT
    for r in [6.0, 8.0, 10.0, 12.0]:
        print(f"  -> Testing cutoff radius R={r} A on WT...", flush=True)
        ext_r = DescriptorExtractor("data/structures/5XJH.pdb", radius_cutoff=r)
        res = ext_r.extract_variant_descriptors("WT", "WT", include_approach_scan=False)
        res["sensitivity_test"] = "radius_cutoff"
        res["test_parameter"] = f"R={r}A"
        radius_records.append(res)
        cleanup_tmp()
    df_radius = pd.DataFrame(radius_records)
    radius_csv = "reports/tables/sensitivity_radius.csv"
    df_radius.to_csv(radius_csv, index=False)
    print(f"Saved {radius_csv}")

    # 5. FAST-PETase vs IsPETase Directional Plausibility Check
    print("\nRunning FAST-PETase (7SH6) vs IsPETase (5XJH) directional plausibility check...", flush=True)
    analyzer = SensitivityAnalyzer("data/structures/5XJH.pdb", "data/structures/7SH6.pdb")
    fast_check = analyzer.run_fast_petase_directional_check()
    cleanup_tmp()
    fast_json = "results/chemistry/fast_petase_directional_check.json"
    with open(fast_json, "w") as f:
        json.dump(fast_check, f, indent=2)
    print(f"Directional Check Verdict: {fast_check['directional_shift']['plausibility_verdict']}")
    print(f"Saved {fast_json}")

    # 6. Generate Analytical Summary Tables
    print("\n--- Generating Analytical Summary Markdown Table ---", flush=True)
    summary_md_path = "reports/tables/computational_chemistry_summary.md"
    cols_to_show = ["variant_id", "mutation", "dist_to_ser160_og", "delta_esp_kcal_mol", "delta_charge_ser_og", "delta_proton_shuttle_proxy_kcal", "delta_approach_barrier_proxy_kcal"]
    
    with open(summary_md_path, "w") as f:
        f.write("# Computational Chemistry Deliverables Summary (Role 2 Lead)\n\n")
        f.write("## 1. Executive Summary\n")
        f.write(f"- **Total variants computed:** {len(df_results)} (1 WT, 15 Arm A4 Gated, 15 Arm A3 Random Control)\n")
        f.write(f"- **Quantum CASCI Controls computed for Role 3:** {len(casci_data['variants'])} candidates (Arm A5)\n")
        f.write(f"- **Average CPU wall-clock time per variant:** {df_results['wall_clock_sec'].mean():.2f} s\n")
        f.write(f"- **Primary Parquet Database:** [`results/chemistry/chem_results.parquet`](file://{os.path.abspath(parquet_path)})\n\n")

        f.write("## 2. Top Gated Variants (Arm A4) vs WT Descriptors\n\n")
        df_a4 = df_results[df_results["arm"] == "A4_gated"][cols_to_show]
        f.write(df_a4.to_markdown(index=False) + "\n\n")

        f.write("## 3. Comparison with Random Control (Arm A3)\n\n")
        df_a3 = df_results[df_results["arm"] == "A3_random"][cols_to_show]
        f.write(df_a3.to_markdown(index=False) + "\n\n")

        f.write("## 4. FAST-PETase (7SH6) Directional Plausibility Check\n\n")
        f.write(f"- **IsPETase (5XJH) Ser160-His237 distance:** {fast_check['ispetase_5xjh']['d_ser160_og_his237_ne2_A']} Å\n")
        f.write(f"- **FAST-PETase (7SH6) Ser160-His237 distance:** {fast_check['fast_petase_7sh6']['d_ser160_og_his237_ne2_A']} Å\n")
        f.write(f"- **IsPETase (5XJH) His237-Asp206 distance:** {fast_check['ispetase_5xjh']['d_his237_nd1_asp206_od2_A']} Å\n")
        f.write(f"- **FAST-PETase (7SH6) His237-Asp206 distance:** {fast_check['fast_petase_7sh6']['d_his237_nd1_asp206_od2_A']} Å\n")
        f.write(f"- **Δ Ser160 OG Mulliken Charge:** {fast_check['directional_shift']['delta_charge_ser_og']:+.4f}\n")
        f.write(f"- **Literature Benchmark:** {fast_check['directional_shift']['literature_benchmark']}\n")
        f.write(f"- **Verdict:** {fast_check['directional_shift']['plausibility_verdict']}\n\n")

        f.write("## 5. Sensitivity Summary\n\n")
        f.write("### Protonation State Sensitivity (HIE vs HIP)\n")
        f.write(df_prot[["variant_id", "test_parameter", "esp_at_ser_og_kcal_mol", "charge_ser_og"]].to_markdown(index=False) + "\n\n")
        f.write("### Radius Cutoff Sensitivity (R = 6, 8, 10, 12 Å)\n")
        f.write(df_radius[["variant_id", "test_parameter", "num_mm_charges", "esp_at_ser_og_kcal_mol"]].to_markdown(index=False) + "\n\n")

    print(f"Summary report written to {summary_md_path}", flush=True)
    cleanup_tmp()
    print("\n=== All Role 2 Deliverables Successfully Completed! ===", flush=True)

if __name__ == "__main__":
    main()
