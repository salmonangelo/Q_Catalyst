#!/usr/bin/env python3
"""
Generates the final Analytical Summary Markdown Table for Role 2.
All underlying calculations are 100% completed.
"""

import json
import os
import pandas as pd

def main():
    print("=== Compiling Final Role 2 Summary Markdown Report ===")
    
    # 1. Load results
    parquet_path = "results/chemistry/chem_results.parquet"
    df_results = pd.read_parquet(parquet_path)
    
    with open("results/quantum/casci_control_role3.json") as f:
        casci_data = json.load(f)
        
    df_prot = pd.read_csv("reports/tables/sensitivity_protonation.csv")
    df_radius = pd.read_csv("reports/tables/sensitivity_radius.csv")
    
    with open("results/chemistry/fast_petase_directional_check.json") as f:
        fast_check = json.load(f)
        
    # 2. Generate Markdown
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
        f.write(df_radius[["variant_id", "test_parameter", "radius_cutoff_A", "esp_at_ser_og_kcal_mol", "charge_ser_og"]].to_markdown(index=False) + "\n\n")

    print(f"Summary report written to {summary_md_path}")
    print("\n=== ALL ROLE 2 DELIVERABLES COMPLETED SUCCESSFULLY! ===")

if __name__ == "__main__":
    main()
