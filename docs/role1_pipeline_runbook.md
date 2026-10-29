# Role 1 End-to-End Runbook

This runbook describes how to execute the integrated Role 1 path using genuine pretrained weights. It is designed to stop on missing checkpoints, split drift, missing outputs, or failed stages rather than silently falling back to a mock.

## Prerequisites

- Use the project-tested Python environment and install the repository's pinned scientific dependencies.
- Ensure the curated variant table, frozen v2 split manifests, ESM-2 35M weights, and ESM-1v checkpoint `facebook/esm1v_t33_650M_UR90S_1` are available locally.
- The ESM-1v checkpoint is about 2.49 GB. The runner does not approve or trigger a full five-checkpoint ESM-1v ensemble download.
- The runner uses `mock=False` for both foundation-model paths. A missing real checkpoint must stop the run; synthetic BLOSUM62 scores and synthetic embeddings are not acceptable outputs for a scientific run.

## Run the pipeline

From the repository root:

```bash
python scripts/run_role1_pipeline.py --target activity
```

This runs, in order:

1. Frozen-split preflight (IDs, sequences, label availability, and manifest consistency).
2. Real ESM-1v masked-marginal zero-shot scoring.
3. Real ESM-2 35M embedding extraction.
4. Validation of model provenance, score finiteness, unique embedding IDs, and 480-dimensional embeddings.
5. Supervised XGBoost and Ridge training/evaluation on the v2 single-to-multi split.
6. Bootstrap ensemble, split-conformal intervals, embedding OOD distance, structural-risk signal, and composite uncertainty.
7. Gate 0 validation and the A0-A2 activity benchmark.
8. Integrated candidate-gate artifact joining the real ESM-1v scores, supervised predictions, uncertainty table, Gate 0 verdict, and frozen split manifest.

To also run the thermostability target:

```bash
python scripts/run_role1_pipeline.py --target both
```

For a fast debugging pass that skips only the A0-A2 benchmark:

```bash
python scripts/run_role1_pipeline.py --target activity --skip-benchmarks
```

Do not use `--max-variants` with a value smaller than the full candidate pool for final benchmark runs. A smoke run over a subset is not the complete pipeline.

## Key outputs

| Artifact | Purpose |
|---|---|
| `results/predictions/esm1v_scores.parquet` | Real ESM-1v mutation-effect scores with provenance |
| `results/embeddings/embedding_index.parquet` | ESM-2 embedding index and model provenance |
| `results/predictions/predictions.parquet` | XGBoost and Ridge supervised predictions |
| `results/uncertainty/uncertainty.parquet` | Activity uncertainty, intervals, OOD score, and split roles |
| `reports/tables/gate0_verdict.json` | Machine-readable Gate 0 verdict |
| `results/benchmarks/benchmark_results.parquet` | A0-A2 metrics |
| `results/rankings/role1_candidate_gate.parquet` | Joined candidate-level Role 1 data-quality and global policy decisions |
| `results/rankings/role1_candidate_gate.json` | Gate policy, candidate counts, source files, and non-claims |
| `reports/role1_pipeline_run.json` | Run manifest with Git revision, stage durations, outcomes, and error details |

A run that fails writes a manifest with `status: failed` and the failing stage. Do not interpret the presence of earlier outputs as a successful end-to-end run.

## Run tests

Run offline unit tests first:

```bash
pytest tests/test_acquisition_gate.py tests/test_gate0.py tests/test_uncertainty.py tests/test_predictor.py tests/test_benchmarks.py
```

Then run the entire offline-compatible suite, excluding tests that require local pretrained model checkpoints or a platform-specific chemistry environment where needed:

```bash
pytest --ignore=tests/test_esm1v_real.py --ignore=tests/test_embeddings_real.py --ignore=tests/test_chemistry.py
```

If the ESM-1v and ESM-2 checkpoints are cached, run the two real integration test files as well:

```bash
pytest tests/test_esm1v_real.py tests/test_embeddings_real.py
```

Record the output, commit SHA, environment manifest, and run-manifest path before marking the branch verified. Unit tests do not prove the biological hypotheses.

## Interpret Gate 0 correctly

Gate 0 passes only when the lower bound of the planned bootstrap confidence interval for the Spearman correlation between uncertainty and absolute prediction error is strictly positive (`ci_lower > 0`). An interval spanning zero is inconclusive. A fully negative interval is evidence of an inverse relationship on this retrospective sample. Both fail the pre-registered positive-association requirement and require `FALLBACK_TO_DIVERSITY_SELECTION`.

High conformal interval coverage is not sufficient by itself: report interval width, calibration size, and the distribution shift between single-site calibration variants and multi-site test variants. The held-out set is small; report uncertainty and limitations.

## Scientific boundary

These are computational prioritisation results on retrospective data. A successful software pipeline does not validate a newly proposed variant in the wet lab, establish that uncertainty generalises to future libraries, or demonstrate quantum advantage. The Role 1 candidate gate is not a replacement for Role 2's three-dimensional structural proximity / chemistry eligibility checks.
