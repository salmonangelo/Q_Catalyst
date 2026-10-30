"""
Script 4: Execute classical active-site cluster electronic-structure calculations.
"""
import os
import sys
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.qcatalyst.chemistry.pyscf_engine import ClusterChemistryEngine


def main():
    print("=" * 60)
    print("Q-CATALYST: Step 4 - Computational Chemistry (PySCF Active-Site Cluster)")
    print("=" * 60)

    jobs_path = PROJECT_ROOT / "results" / "chemistry" / "chem_jobs.json"
    if not jobs_path.exists():
        raise FileNotFoundError(f"Missing {jobs_path}. Run select_candidates.py first.")

    with open(jobs_path, "r") as f:
        jobs = json.load(f)

    print(f"Executing cluster calculations for {len(jobs)} scheduled jobs...")
    engine = ClusterChemistryEngine(method="B3LYP-D3", basis="def2-SVP")

    results = []
    total_cpu_time = 0.0

    for job in jobs:
        res = engine.compute_cluster_descriptors(
            variant_id=job["variant_id"],
            mutations_str=job["mutations"],
            min_dist=job["min_dist_to_triad_angstrom"]
        )
        res["job_id"] = job["job_id"]
        res["selection_rank"] = job["selection_rank"]
        results.append(res)
        total_cpu_time += res["qm_runtime_seconds"]
        print(f"  [{job['job_id']}] {job['mutations']} -> d_attack={res['d_attack_angstrom']} Å, "
              f"TS_stab={res['ts_stabilization_kcal_mol']:+.2f} kcal/mol, runtime={res['qm_runtime_seconds']}s")

    df_chem = pd.DataFrame(results)
    out_path = PROJECT_ROOT / "results" / "chemistry" / "chem_results.parquet"
    df_chem.to_parquet(out_path, index=False)

    print(f"\nAll cluster calculations converged successfully!")
    print(f"Total chemistry compute time: {total_cpu_time:.1f} seconds ({total_cpu_time/3600:.4f} CPU hours)")
    print(f"Saved chemistry results to: {out_path}")
    print("Step 4 Complete!\n")


if __name__ == "__main__":
    main()
