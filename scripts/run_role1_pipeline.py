"""Run the complete Role 1 protein-ML pipeline with real pretrained weights.

This is an end-to-end driver, not a mock/CI path. It stops immediately if a
real model checkpoint, input artifact, split assertion, or downstream stage fails.
It runs Gate 0 on activity because the pre-registered hit criterion is activity
relative to wild type; Tm predictions and uncertainty can also be requested.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for entry in (REPO_ROOT, SRC_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import pandas as pd

from scripts import run_ai
from scripts.select_candidates import build_role1_candidate_gate


def get_git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return None


def run_stage(name: str, action: Callable[[], int], records: list[dict]) -> None:
    start = time.monotonic()
    print(f"\n{'=' * 72}\nSTAGE: {name}\n{'=' * 72}", flush=True)
    try:
        rc = action()
    except Exception as exc:
        records.append({
            "stage": name, "status": "failed", "return_code": None,
            "duration_seconds": round(time.monotonic() - start, 3),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    status = "passed" if rc == 0 else "failed"
    records.append({
        "stage": name, "status": status, "return_code": int(rc),
        "duration_seconds": round(time.monotonic() - start, 3),
    })
    if rc != 0:
        raise RuntimeError(f"Stage '{name}' failed with exit code {rc}.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute Q-Catalyst Role 1 end-to-end using real ESM-1v and ESM-2 weights."
    )
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/curated/variants.parquet")
    parser.add_argument("--enzyme", default="IsPETase")
    parser.add_argument("--target", choices=["activity", "tm", "both"], default="activity")
    parser.add_argument("--max-variants", type=int, default=100000,
                        help="Upper bound on unique variants to score/embed; default processes all.")
    parser.add_argument("--skip-benchmarks", action="store_true",
                        help="Skip the activity A0-A2 benchmark only; all preceding validation still runs.")
    parser.add_argument("--run-manifest", type=Path,
                        default=REPO_ROOT / "reports" / "role1_pipeline_run.json")
    args = parser.parse_args()

    if args.max_variants < 1:
        parser.error("--max-variants must be >= 1")
    if not args.input.exists():
        parser.error(f"Input variants table does not exist: {args.input}")

    df = pd.read_parquet(args.input)
    required = {
        "variant_id", "parent_enzyme", "wildtype_sequence", "mutant_sequence",
        "mutant_raw", "is_duplicate", "is_multi_mutant", "activity_raw", "tm_raw",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        parser.error(f"Curated variants table missing required columns: {missing}")
    enzyme_df = df[df["parent_enzyme"].astype(str).str.lower() == args.enzyme.lower()]
    if enzyme_df.empty:
        parser.error(f"No records found for parent enzyme {args.enzyme!r}.")

    run_started = datetime.now(timezone.utc).isoformat()
    stage_records: list[dict] = []
    target_names = ["activity", "tm"] if args.target == "both" else [args.target]
    results = REPO_ROOT / "results"
    predictions = results / "predictions" / "predictions.parquet"
    esm1v_scores = results / "predictions" / "esm1v_scores.parquet"
    esm2_output = results / "embeddings"
    uncertainty_activity = results / "uncertainty" / "uncertainty.parquet"
    uncertainty_tm = results / "uncertainty" / "uncertainty_tm.parquet"

    print("Q-CATALYST ROLE 1 — END-TO-END REAL-WEIGHT RUN")
    print(f"Repository: {REPO_ROOT}")
    print(f"Git revision: {get_git_revision() or 'unavailable (not a Git checkout)'}")
    print(f"Parent enzyme: {args.enzyme}")
    print(f"Targets: {', '.join(target_names)}")
    print("ESM-1v: facebook/esm1v_t33_650M_UR90S_1 (mock=False)")
    print("ESM-2: facebook/esm2_t12_35M_UR50D (mock=False)")
    print("Policy: no mock fallback; missing checkpoints or artifacts fail the run.")
    print(f"Input rows for enzyme: {len(enzyme_df)}")
    print(f"Unique variant IDs: {enzyme_df['variant_id'].nunique()}")

    try:
        run_stage(
            "ESM-1v real masked-marginal scoring",
            lambda: run_ai.run_esm_scoring(
                input_path=args.input,
                output_path=esm1v_scores,
                enzyme=args.enzyme,
                max_variants=args.max_variants,
                mock=False,
            ),
            stage_records,
        )
        run_stage(
            "ESM-2 real embedding extraction",
            lambda: run_ai.run_embedding_extraction(
                input_path=args.input,
                output_dir=esm2_output,
                enzyme=args.enzyme,
                max_variants=args.max_variants,
                mock=False,
            ),
            stage_records,
        )

        for target in target_names:
            run_stage(
                f"Supervised training and leakage-free evaluation ({target})",
                lambda target=target: run_ai.run_supervised_training(
                    variants_path=args.input,
                    splits_dir=REPO_ROOT / "data" / "splits",
                    models_dir=results / "models",
                    predictions_path=predictions,
                    reports_dir=REPO_ROOT / "reports",
                    target_name=target,
                    enzyme=args.enzyme,
                    mock_embeddings=False,
                    split_version="v2",
                ),
                stage_records,
            )
            uncertainty_path = uncertainty_activity if target == "activity" else uncertainty_tm
            report_dir = REPO_ROOT / "reports" if target == "activity" else REPO_ROOT / "reports" / "role1" / "tm"
            run_stage(
                f"Uncertainty, OOD and conformal estimation ({target})",
                lambda target=target, uncertainty_path=uncertainty_path, report_dir=report_dir: run_ai.run_uncertainty_estimation(
                    variants_path=args.input,
                    output_path=uncertainty_path,
                    reports_dir=report_dir,
                    target_name=target,
                    enzyme=args.enzyme,
                    mock_embeddings=False,
                    split_version="v2",
                ),
                stage_records,
            )

        if "activity" in target_names:
            run_stage(
                "Gate 0 validation (activity)",
                lambda: run_ai.run_gate0_validation(
                    uncertainty_path=uncertainty_activity,
                    reports_dir=REPO_ROOT / "reports",
                    target_name="activity",
                    enzyme=args.enzyme,
                ),
                stage_records,
            )
            if not args.skip_benchmarks:
                run_stage(
                    "A0-A2 benchmark suite (activity)",
                    lambda: run_ai.run_benchmarking(
                        variants_path=args.input,
                        predictions_path=predictions,
                        uncertainty_path=uncertainty_activity,
                        benchmarks_dir=results / "benchmarks",
                        reports_dir=REPO_ROOT / "reports",
                        target_name="activity",
                        enzyme=args.enzyme,
                        good_threshold=1.0,
                        mock_esm=False,
                    ),
                    stage_records,
                )
            run_stage(
                "Integrated Gate 0 candidate-gate artifact",
                lambda: (
                    build_role1_candidate_gate(
                        variants_path=args.input,
                        predictions_path=predictions,
                        uncertainty_path=uncertainty_activity,
                        esm1v_scores_path=esm1v_scores,
                        gate0_verdict_path=REPO_ROOT / "reports" / "tables" / "gate0_verdict.json",
                        split_manifest_path=REPO_ROOT / "data" / "splits" / f"{args.enzyme.lower()}_activity_single_to_multi_v2.json",
                        output_path=results / "rankings" / "role1_candidate_gate.parquet",
                        decision_path=results / "rankings" / "role1_candidate_gate.json",
                        enzyme=args.enzyme,
                        target_name="activity",
                    ) and 0
                ),
                stage_records,
            )

    except Exception as exc:
        run_manifest = {
            "pipeline": "Q-Catalyst Role 1",
            "status": "failed",
            "run_started_utc": run_started,
            "run_finished_utc": datetime.now(timezone.utc).isoformat(),
            "git_revision": get_git_revision(),
            "input_path": str(args.input),
            "parent_enzyme": args.enzyme,
            "targets": target_names,
            "real_weights_required": True,
            "stages": stage_records,
            "error": f"{type(exc).__name__}: {exc}",
        }
        args.run_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.run_manifest.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        print(f"PIPELINE FAILED. Partial run manifest: {args.run_manifest}", file=sys.stderr)
        print(run_manifest["error"], file=sys.stderr)
        return 1

    run_manifest = {
        "pipeline": "Q-Catalyst Role 1",
        "status": "passed_all_requested_stages",
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": get_git_revision(),
        "input_path": str(args.input),
        "parent_enzyme": args.enzyme,
        "targets": target_names,
        "real_weights_required": True,
        "mock_fallback_allowed": False,
        "stages": stage_records,
        "outputs": {
            "esm1v_scores": str(esm1v_scores),
            "esm2_embedding_index": str(esm2_output / "embedding_index.parquet"),
            "supervised_predictions": str(predictions),
            "activity_uncertainty": str(uncertainty_activity) if "activity" in target_names else None,
            "tm_uncertainty": str(uncertainty_tm) if "tm" in target_names else None,
            "gate0_verdict": str(REPO_ROOT / "reports" / "tables" / "gate0_verdict.json") if "activity" in target_names else None,
            "candidate_gate": str(results / "rankings" / "role1_candidate_gate.parquet") if "activity" in target_names else None,
        },
        "non_claims": [
            "Predictions and zero-shot scores are not experimental evidence of improved PET degradation.",
            "A passing software pipeline does not establish predictive validity or quantum advantage.",
            "If Gate 0 fails, uncertainty must not be used as a candidate filter; the policy falls back to diversity selection.",
        ],
    }
    args.run_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.run_manifest.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print("ROLE 1 PIPELINE COMPLETE")
    print("=" * 72)
    print(f"Manifest: {args.run_manifest}")
    print(f"Stages passed: {sum(x['status'] == 'passed' for x in stage_records)}/{len(stage_records)}")
    print("Review the Gate 0 verdict and candidate-gate mode before using uncertainty downstream.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
