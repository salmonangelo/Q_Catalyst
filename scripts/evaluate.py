"""
Script 6: Fuse evidence, assign confidence tiers, and evaluate benchmark arms A0 through A7.
"""
import os
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.qcatalyst.fusion.ranker import fuse_evidence_and_rank
from src.qcatalyst.evaluation.benchmark import run_benchmark_arms


def main():
    print("=" * 60)
    print("Q-CATALYST: Step 6 - Evidence Fusion, Ranking & Benchmark Evaluation")
    print("=" * 60)

    curated_path = PROJECT_ROOT / "data" / "curated" / "variants.parquet"
    pred_path = PROJECT_ROOT / "results" / "predictions" / "predictions.parquet"
    unc_path = PROJECT_ROOT / "results" / "uncertainty" / "uncertainty.parquet"
    chem_path = PROJECT_ROOT / "results" / "chemistry" / "chem_results.parquet"
    quant_path = PROJECT_ROOT / "results" / "quantum" / "quantum_results.parquet"

    df_variants = pd.read_parquet(curated_path)
    df_pred = pd.read_parquet(pred_path)
    df_unc = pd.read_parquet(unc_path)
    df_chem = pd.read_parquet(chem_path) if chem_path.exists() else None
    df_quant = pd.read_parquet(quant_path) if quant_path.exists() else None

    # Fuse evidence and rank
    df_ranked = fuse_evidence_and_rank(df_variants, df_pred, df_unc, df_chem, df_quant)

    rankings_dir = PROJECT_ROOT / "results" / "rankings"
    benchmarks_dir = PROJECT_ROOT / "results" / "benchmarks"
    rankings_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    ranking_out = rankings_dir / "ranking.parquet"
    df_ranked.to_parquet(ranking_out, index=False)
    print(f"Saved prioritised rankings to: {ranking_out}")

    # Top 5 Candidates
    print("\nTop 5 Prioritised PETase Candidates:")
    for _, row in df_ranked.head(5).iterrows():
        print(f"  Rank {row['final_rank']}: {row['mutations']} | Fused Score: {row['final_composite_score']:.2f} "
              f"| Tier: {row['confidence_tier']} | Prior Art: {row.get('prior_art_name', 'None')}")

    # Run Benchmark Arms A0 - A7
    df_bench = run_benchmark_arms(df_ranked, bootstrap_iterations=1000)
    bench_out = benchmarks_dir / "benchmark_results.parquet"
    df_bench.to_parquet(bench_out, index=False)
    print(f"\nSaved benchmark results to: {bench_out}")

    print("\nBenchmark Arm Summary (Precision@10 and Value of Chemistry):")
    for _, row in df_bench.iterrows():
        print(f"  [{row['arm_id']}] {row['arm_name']:<35} | P@10: {row['precision_at_10']:.2f} "
              f"(95% CI: [{row['ci_lower_p10']:.2f}, {row['ci_upper_p10']:.2f}]) | VoC: {row['voc_efficiency']:+.2f} / hr")

    print("\nStep 6 Complete! All pipeline stages are ready.\n")


if __name__ == "__main__":
    main()
