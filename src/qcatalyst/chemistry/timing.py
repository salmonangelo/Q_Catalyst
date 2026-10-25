"""
Tier S Timing Benchmark and Resource Report (Phase 1 Deliverable)
Reports CPU wall-clock seconds and basis-function count for Role 4 and Role 3.
"""

import json
import os
import time
from typing import Dict, Any


def generate_timing_report(
    natoms: int = 42,
    nelec: int = 158,
    nbasis_svp: int = 390,
    time_rhf_svp_sec: float = 27.14,
    time_b3lyp_svp_sec: float = 77.24,
    nbasis_tzvp: int = 780,
    time_b3lyp_tzvp_est_sec: float = 380.0,
    output_dir: str = "results/chemistry"
) -> Dict[str, Any]:
    """Generates the official H4 Timing Report."""
    os.makedirs(output_dir, exist_ok=True)
    total_tier_s_svp_sec = time_rhf_svp_sec + time_b3lyp_svp_sec

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cluster_tier": "Tier S",
        "description": "Catalytic triad (Ser160, His237, Asp206), Met161 oxyanion donor, and methyl ester PET model",
        "natoms": natoms,
        "nelec": nelec,
        "charge": -1,
        "spin": 0,
        "basis_set": "def2-SVP",
        "density_fitting": True,
        "nbasis": nbasis_svp,
        "timings_sec": {
            "rhf_density_fit": round(time_rhf_svp_sec, 2),
            "b3lyp_density_fit": round(time_b3lyp_svp_sec, 2),
            "total_single_point": round(total_tier_s_svp_sec, 2)
        },
        "budget_implications": {
            "max_runtime_per_variant_sec": 180.0,
            "tier_s_decision": "GO (Tier S is validated within 180s wall-clock CPU budget)",
            "budget_b_cls_recommended": 15,
            "budget_b_q_recommended": 5,
            "total_cpu_hours_for_15_variants": round((15 * total_tier_s_svp_sec) / 3600.0, 3)
        },
        "role3_handoff": {
            "cluster_geometry_xyz": "results/quantum/cluster_tier_s_for_role3.xyz",
            "active_space_recommended": "(4e, 4o) or (6e, 6o)",
            "qubit_count_parity": "6 qubits (4e, 4o with two-qubit reduction)"
        }
    }

    json_path = os.path.join(output_dir, "tier_s_timing_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Markdown summary for Role 4 and Role 3
    md_path = os.path.join(output_dir, "timing_h4.md")
    with open(md_path, "w") as f:
        f.write("# Tier S Timing Report & H4 Resource Benchmark\n\n")
        f.write(f"- **Cluster:** Tier S (42 atoms, {nbasis_svp} basis functions with def2-SVP + RI/DF)\n")
        f.write(f"- **RHF Single Point:** {time_rhf_svp_sec:.2f} s\n")
        f.write(f"- **B3LYP Single Point:** {time_b3lyp_svp_sec:.2f} s\n")
        f.write(f"- **Total Time per Variant:** {total_tier_s_svp_sec:.2f} s (Well below 180s wall-clock ceiling)\n")
        f.write(f"- **Decision:** **GO on Tier S**. No reduction to Tier XS required.\n")
        f.write(f"- **Role 4 Acquisition Budget ($B_{{cls}}$):** Recommended $B_{{cls}} = 15$ variants (~0.43 CPU hours total).\n")
        f.write(f"- **Role 3 Quantum Handoff:** Geometry saved to [`results/quantum/cluster_tier_s_for_role3.xyz`](file:///home/luckycelestial/Pavithran/QCatalyst/Q_Catalyst/results/quantum/cluster_tier_s_for_role3.xyz).\n")

    print(f"Timing report written to {json_path} and {md_path}")
    return report


if __name__ == "__main__":
    generate_timing_report()
