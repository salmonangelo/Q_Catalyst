"""Script to run protein AI and uncertainty workflows in Q-Catalyst.

Usage:
    # 1. ESM-1v Mutation-Effect Scoring
    python scripts/run_ai.py --score-esm1v --max-variants 20

    # 2. ESM-2 Embedding Extraction
    python scripts/run_ai.py --extract-embeddings --max-variants 20

    # 3. Supervised Model Training & Held-Out Evaluation
    python scripts/run_ai.py --train-predictor --target activity --enzyme IsPETase

    # 4. Uncertainty Estimation & Split-Conformal Calibration
    python scripts/run_ai.py --estimate-uncertainty --target activity --enzyme IsPETase
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add src to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from qcatalyst.protein_ai.embeddings import (
    ESM2Embedder,
    ESM2EmbedderConfig,
    save_embedding_index,
)
from qcatalyst.protein_ai.esm_scoring import (
    ESMScorer,
    ESMScorerConfig,
    save_esm_predictions,
)
from qcatalyst.protein_ai.predictor import (
    SupervisedModelConfig,
    SupervisedPETasePredictor,
    create_frozen_split,
)
from qcatalyst.uncertainty.calibrate import SplitConformalCalibrator
from qcatalyst.uncertainty.estimate import (
    EnsembleConfig,
    EnsemblePredictor,
    OODDetector,
    compute_aggregate_uncertainty,
    compute_structural_risk,
)
from qcatalyst.uncertainty.gate import (
    Gate0Result,
    evaluate_gate0,
    generate_gate0_coverage_plot,
    save_gate0_artifacts,
)
from qcatalyst.evaluation.benchmarks import (
    run_benchmark_suite,
    save_benchmark_artifacts,
)


def run_esm_scoring(
    input_path: Path,
    output_path: Path,
    enzyme: str | None = None,
    max_variants: int = 20,
    mock: bool = True,
    cache_dir: Path | None = None,
) -> int:
    """Score a selected subset of variants using ESM-1v."""
    start_time = time.time()
    print("=" * 70)
    print("Q-CATALYST: ESM-1v Mutation-Effect Scoring")
    print("=" * 70)
    print(f"Input variants table: {input_path.resolve()}")
    print(f"Output predictions:   {output_path.resolve()}")
    print(f"Max variants to score: {max_variants} (home session compute limit)")
    print(f"Execution mode:        {'Mock (calibrated LLR proxy)' if mock else 'Real weights'}")
    print("Network access:        None (zero downloads)")
    print("-" * 70)

    if not input_path.exists():
        print(f"Error: Input variants file not found at {input_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(input_path, engine="pyarrow")
    print(f"Loaded {len(df)} curated variants from {input_path.name}.")

    if enzyme:
        df = df[df["parent_enzyme"].str.lower() == enzyme.lower()]
        print(f"Filtered by parent enzyme '{enzyme}': {len(df)} variants remaining.")

    if df.empty:
        print("No variants match filter criteria.", file=sys.stderr)
        return 1

    subset_df = df.head(max_variants)
    print(f"Scoring batch of {len(subset_df)} variants...")

    config = ESMScorerConfig(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        model_version="v1",
        scoring_strategy="masked_marginal",
        cache_dir=cache_dir,
        mock=mock,
        device="cpu",
    )
    scorer = ESMScorer(config)

    predictions_df = scorer.score_variants_dataframe(subset_df)

    # Save to Parquet
    reloaded = save_esm_predictions(predictions_df, output_path)
    file_size = output_path.stat().st_size

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Successfully scored {len(reloaded)} variants in {elapsed:.2f}s.")
    print(f"Saved Parquet output: {output_path} ({file_size} bytes)")
    print("=" * 70)

    print("\nSample ESM-1v Score Predictions (First 5):")
    display_cols = ["variant_id", "mutant_raw", "score", "scoring_method", "run_status"]
    print(reloaded[display_cols].head(5).to_string(index=False))
    print("\nScientific Note:")
    print("All scores are zero-shot predicted log-likelihood ratios; not experimentally validated activity.")

    return 0


def run_embedding_extraction(
    input_path: Path,
    output_dir: Path,
    model_id: str = "facebook/esm2_t12_35M_UR50D",
    pooling: str = "mean",
    enzyme: str | None = None,
    max_variants: int = 20,
    mock: bool = True,
) -> int:
    """Extract ESM-2 sequence representations for a small batch of variants."""
    start_time = time.time()
    print("=" * 70)
    print("Q-CATALYST: ESM-2 Embedding Extraction")
    print("=" * 70)
    print(f"Input variants table:  {input_path.resolve()}")
    print(f"Embeddings cache dir:  {output_dir.resolve()}")
    print(f"Model ID:              {model_id}")
    print(f"Pooling Strategy:      {pooling}")
    print(f"Max variants batch:    {max_variants} (home compute budget limit)")
    print(f"Execution mode:        {'Mock (deterministic proxy)' if mock else 'Real weights'}")
    print("Git status:            Large arrays excluded via .gitignore")
    print("-" * 70)

    if not input_path.exists():
        print(f"Error: Input variants file not found at {input_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(input_path, engine="pyarrow")
    if enzyme:
        df = df[df["parent_enzyme"].str.lower() == enzyme.lower()]

    subset_df = df.head(max_variants)
    print(f"Processing batch of {len(subset_df)} sequences...")

    config = ESM2EmbedderConfig(
        model_id=model_id,
        pooling_strategy=pooling,
        cache_dir=output_dir,
        mock=mock,
        device="cpu",
    )
    embedder = ESM2Embedder(config)

    matrix, index_df = embedder.extract_batch(
        subset_df,
        sequence_column="mutant_sequence",
        id_column="variant_id",
    )

    index_path = output_dir / "embedding_index.parquet"
    reloaded_index = save_embedding_index(index_df, index_path)

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Embedding matrix shape: {matrix.shape} (dtype: {matrix.dtype})")
    print(f"Finite value check:     {bool(matrix.size > 0 and (matrix == matrix).all())}")
    print(f"Saved embedding index:  {index_path} ({index_path.stat().st_size} bytes)")
    print(f"Extraction completed in {elapsed:.2f}s.")
    print("=" * 70)

    print("\nSample Embedding Index Entries (First 5):")
    display_cols = ["variant_id", "sequence_hash", "dimensions", "status", "is_finite"]
    print(reloaded_index[display_cols].head(5).to_string(index=False))

    return 0


def run_supervised_training(
    variants_path: Path,
    splits_dir: Path,
    models_dir: Path,
    predictions_path: Path,
    reports_dir: Path,
    target_name: str = "activity",
    enzyme: str = "IsPETase",
    random_seed: int = 42,
    mock_embeddings: bool = True,
    split_version: str = "v2",
) -> int:
    """Train XGBoost model and Ridge comparison baseline and evaluate on held-out split."""
    start_time = time.time()
    print("=" * 70)
    print(f"Q-CATALYST: Supervised PETase Prediction ({target_name.upper()})")
    print("=" * 70)
    print(f"Parent Enzyme:     {enzyme}")
    print(f"Target Column:     {'activity_raw' if target_name == 'activity' else 'tm_raw'}")
    print(f"Primary Model:     XGBoost (XGBRegressor)")
    print(f"Baseline Model:    Ridge Regression (alpha=1.0)")
    print(f"Split Strategy:    single_to_multi (train=single-site, held-out=multi-site)")
    print(f"Split Version:     {split_version} (leakage-free single-to-multi partition)")
    print(f"Random Seed:       {random_seed}")
    print("-" * 70)

    if not variants_path.exists():
        print(f"Error: Variants table not found at {variants_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(variants_path, engine="pyarrow")
    target_col = "activity_raw" if target_name == "activity" else "tm_raw"

    eligible_df = df[
        (df["parent_enzyme"].str.lower() == enzyme.lower())
        & df[target_col].notnull()
        & (~df["is_duplicate"])
    ].copy()

    # 1. Create and freeze data split
    train_df, eval_df, split_manifest = create_frozen_split(
        eligible_df, split_strategy="single_to_multi", random_seed=random_seed, version=split_version
    )

    splits_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{split_version}" if split_version != "v1" else ""
    split_file = splits_dir / f"{enzyme.lower()}_{target_name}_single_to_multi{suffix}.json"
    split_file.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    # 2. Extract / load embeddings
    emb_cache_dir = REPO_ROOT / "results" / "embeddings"
    embedder_cfg = ESM2EmbedderConfig(
        model_id="facebook/esm2_t12_35M_UR50D",
        cache_dir=emb_cache_dir,
        mock=mock_embeddings,
        device="cpu",
    )
    embedder = ESM2Embedder(embedder_cfg)

    X_train, _ = embedder.extract_batch(train_df)
    X_eval, _ = embedder.extract_batch(eval_df)

    y_train = train_df[target_col].to_numpy(dtype=np.float64)
    y_eval = eval_df[target_col].to_numpy(dtype=np.float64)

    # 3. Fit models
    model_cfg = SupervisedModelConfig(
        target_name=target_name,
        parent_enzyme=enzyme,
        split_strategy="single_to_multi",
        random_seed=random_seed,
    )
    predictor = SupervisedPETasePredictor(model_cfg)
    predictor.fit(X_train, y_train)

    # 4. Evaluate on held-out test set
    metrics_xgb, metrics_ridge = predictor.evaluate(X_eval, y_eval)

    # 5. Save model
    save_dir = models_dir / f"{enzyme.lower()}_{target_name}"
    predictor.save(save_dir)

    # 6. Generate predictions Parquet table
    pred_xgb_eval, pred_ridge_eval = predictor.predict(X_eval)
    pred_xgb_train, pred_ridge_train = predictor.predict(X_train)

    pred_records = []
    for i, (_, row) in enumerate(train_df.iterrows()):
        pred_records.append({
            "row_id": row["row_id"],
            "variant_id": row["variant_id"],
            "parent_enzyme": enzyme,
            "dataset_split": row["dataset_split"],
            "split_role": "train",
            "mutant_raw": row["mutant_raw"],
            "mutations_canonical": row["mutations_canonical"],
            "target_name": target_name,
            "target_unit": row["activity_unit"] if target_name == "activity" else row["tm_unit"],
            "y_true": float(row[target_col]),
            "y_pred_xgb": float(pred_xgb_train[i]),
            "y_pred_ridge": float(pred_ridge_train[i]),
            "residual_xgb": float(row[target_col] - pred_xgb_train[i]),
            "residual_ridge": float(row[target_col] - pred_ridge_train[i]),
            "model_id": "xgb_regressor_v1",
            "baseline_id": "ridge_regressor_v1",
            "prediction_status": "success",
            "is_prediction": True,
            "disclaimer": "Computational model prediction; not experimentally validated improvement.",
        })

    for i, (_, row) in enumerate(eval_df.iterrows()):
        pred_records.append({
            "row_id": row["row_id"],
            "variant_id": row["variant_id"],
            "parent_enzyme": enzyme,
            "dataset_split": row["dataset_split"],
            "split_role": "held_out_test",
            "mutant_raw": row["mutant_raw"],
            "mutations_canonical": row["mutations_canonical"],
            "target_name": target_name,
            "target_unit": row["activity_unit"] if target_name == "activity" else row["tm_unit"],
            "y_true": float(row[target_col]),
            "y_pred_xgb": float(pred_xgb_eval[i]),
            "y_pred_ridge": float(pred_ridge_eval[i]),
            "residual_xgb": float(row[target_col] - pred_xgb_eval[i]),
            "residual_ridge": float(row[target_col] - pred_ridge_eval[i]),
            "model_id": "xgb_regressor_v1",
            "baseline_id": "ridge_regressor_v1",
            "prediction_status": "success",
            "is_prediction": True,
            "disclaimer": "Computational model prediction; not experimentally validated improvement.",
        })

    predictions_df = pd.DataFrame(pred_records)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path.exists():
        try:
            existing_df = pd.read_parquet(predictions_path)
            non_overlap = existing_df[
                ~((existing_df["target_name"] == target_name) & (existing_df["parent_enzyme"].str.lower() == enzyme.lower()))
            ]
            predictions_df = pd.concat([non_overlap, predictions_df], ignore_index=True)
        except Exception:
            pass

    predictions_df.to_parquet(predictions_path, engine="pyarrow", index=False)

    # 7. Write metrics JSON
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = tables_dir / f"supervised_{enzyme.lower()}_{target_name}_metrics.json"

    metrics_payload = {
        "enzyme": enzyme,
        "target": target_name,
        "split_strategy": "single_to_multi",
        "n_train": len(train_df),
        "n_eval": len(eval_df),
        "xgboost": metrics_xgb.to_dict(),
        "ridge": metrics_ridge.to_dict(),
    }
    metrics_file.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Training & evaluation finished in {elapsed:.2f}s.")
    print("=" * 70)

    print(f"\nHeld-Out Test Results ({target_name.upper()}):")
    print(f"  XGBoost: MAE={metrics_xgb.mae:.4f}, RMSE={metrics_xgb.rmse:.4f}, "
          f"Spearman rho={metrics_xgb.spearman_rho:.4f} (p={metrics_xgb.spearman_pvalue:.2e})")
    print(f"  Ridge:   MAE={metrics_ridge.mae:.4f}, RMSE={metrics_ridge.rmse:.4f}, "
          f"Spearman rho={metrics_ridge.spearman_rho:.4f} (p={metrics_ridge.spearman_pvalue:.2e})")

    return 0


def run_uncertainty_estimation(
    variants_path: Path,
    output_path: Path,
    reports_dir: Path,
    target_name: str = "activity",
    enzyme: str = "IsPETase",
    random_seed: int = 42,
    n_ensemble: int = 10,
    alpha: float = 0.10,
    mock_embeddings: bool = True,
    split_version: str = "v2",
) -> int:
    """Execute Feature 6 uncertainty estimation pipeline with split-conformal calibration."""
    start_time = time.time()
    print("=" * 70)
    print(f"Q-CATALYST: Feature 6 - Uncertainty Estimation & Conformal Calibration")
    print("=" * 70)
    print(f"Parent Enzyme:        {enzyme}")
    print(f"Target Column:        {'activity_raw' if target_name == 'activity' else 'tm_raw'}")
    print(f"Ensemble Members:     {n_ensemble} bootstrap XGBoost predictors")
    print(f"Target Coverage:      {int((1.0 - alpha) * 100)}% (alpha = {alpha})")
    print(f"Output Table:         {output_path.resolve()}")
    print(f"Split Version:        {split_version} (leakage-free single-to-multi partition)")
    print("Data Leakage Rule:    Calibration done strictly on train split; test set preserved.")
    print("-" * 70)

    if not variants_path.exists():
        print(f"Error: Variants table not found at {variants_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(variants_path, engine="pyarrow")
    target_col = "activity_raw" if target_name == "activity" else "tm_raw"

    eligible_df = df[
        (df["parent_enzyme"].str.lower() == enzyme.lower())
        & df[target_col].notnull()
        & (~df["is_duplicate"])
    ].copy()

    # 1. Split into single-site (training pool) and multi-site (held-out evaluation)
    train_pool_df, test_df, _ = create_frozen_split(
        eligible_df, split_strategy="single_to_multi", random_seed=random_seed, version=split_version
    )

    # Further partition train_pool into model-train (75%) and calibration (25%)
    # Guaranteed leakage-free: test set is untouched during calibration
    shuffled_train_pool = train_pool_df.sample(frac=1.0, random_state=random_seed).copy()
    n_cal = max(5, int(len(shuffled_train_pool) * 0.25))
    cal_df = shuffled_train_pool.iloc[:n_cal].copy()
    fit_df = shuffled_train_pool.iloc[n_cal:].copy()

    print(f"Partitioned single-site pool (N={len(train_pool_df)}):")
    print(f"  Model-fit split:   {len(fit_df)} variants")
    print(f"  Calibration split: {len(cal_df)} variants (independent holdout)")
    print(f"Held-out test set:   {len(test_df)} variants")

    # 2. Extract embeddings
    emb_cache_dir = REPO_ROOT / "results" / "embeddings"
    embedder = ESM2Embedder(ESM2EmbedderConfig(cache_dir=emb_cache_dir, mock=mock_embeddings))

    X_fit, _ = embedder.extract_batch(fit_df)
    X_cal, _ = embedder.extract_batch(cal_df)
    X_test, _ = embedder.extract_batch(test_df)

    y_fit = fit_df[target_col].to_numpy(dtype=np.float64)
    y_cal = cal_df[target_col].to_numpy(dtype=np.float64)
    y_test = test_df[target_col].to_numpy(dtype=np.float64)

    # 3. Train ensemble on fit_df
    print(f"\nTraining {n_ensemble} bootstrap XGBoost ensemble members...")
    ensemble_cfg = EnsembleConfig(n_members=n_ensemble, random_seed=random_seed)
    ensemble = EnsemblePredictor(ensemble_cfg)
    ensemble.fit(X_fit, y_fit)

    # 4. Calibrate conformal intervals on cal_df
    print("Calibrating split-conformal prediction intervals...")
    pred_cal_mean, _, _ = ensemble.predict(X_cal)
    calibrator = SplitConformalCalibrator(alpha=alpha)
    calibrator.calibrate(y_cal, pred_cal_mean)
    print(f"  Conformal quantile threshold q = {calibrator.threshold_q:.4f}")

    # 5. Fit OOD detector on X_fit
    ood_detector = OODDetector()
    ood_detector.fit(X_fit)

    # 6. Evaluate all splits
    splits_data = [
        (fit_df, X_fit, y_fit, "model_train"),
        (cal_df, X_cal, y_cal, "calibration"),
        (test_df, X_test, y_test, "held_out_test"),
    ]

    uncertainty_records = []
    test_coverage_val = 0.0

    for split_df, X_split, y_split, role in splits_data:
        pred_mean, pred_std, _ = ensemble.predict(X_split)
        conf_res = calibrator.predict_intervals(pred_mean)
        ood_scores, ood_statuses = ood_detector.score(X_split)

        if role == "held_out_test":
            cov, n_cov, n_tot = calibrator.evaluate_coverage(
                y_split, conf_res.lower_bounds, conf_res.upper_bounds
            )
            test_coverage_val = cov
            print(f"Held-out test interval coverage: {cov * 100:.1f}% ({n_cov}/{n_tot} variants covered)")

        for i, (_, row) in enumerate(split_df.iterrows()):
            # Structural risk calculation
            struct_risk, struct_status = compute_structural_risk(
                row["structure_5xjh_positions"],
                row["structure_mapping_status"],
            )

            # Aggregate uncertainty U(x)
            u_agg, agg_status = compute_aggregate_uncertainty(
                u_ensemble=float(pred_std[i]),
                u_conformal=float(conf_res.interval_widths[i]),
                u_ood=float(ood_scores[i]),
            )

            is_covered = (
                bool((y_split[i] >= conf_res.lower_bounds[i]) and (y_split[i] <= conf_res.upper_bounds[i]))
                if np.isfinite(y_split[i]) and np.isfinite(conf_res.lower_bounds[i])
                else False
            )

            flags = []
            if ood_statuses[i] == "ood_warning":
                flags.append("ood_warning")
            if struct_status != "available":
                flags.append("missing_structure")
            if not flags:
                flags.append("clean")

            uncertainty_records.append({
                "row_id": row["row_id"],
                "variant_id": row["variant_id"],
                "parent_enzyme": enzyme,
                "dataset_split": row["dataset_split"],
                "split_role": role,
                "target_name": target_name,
                "y_true": float(y_split[i]),
                "y_pred_mean": float(pred_mean[i]),
                "u_ensemble_std": float(pred_std[i]),
                "conformal_lower": float(conf_res.lower_bounds[i]),
                "conformal_upper": float(conf_res.upper_bounds[i]),
                "u_conformal_width": float(conf_res.interval_widths[i]),
                "conformal_covered": is_covered,
                "u_ood_distance": float(ood_scores[i]),
                "ood_status": ood_statuses[i],
                "u_structural_risk": struct_risk,
                "structural_signal_status": struct_status,
                "aggregate_uncertainty": u_agg,
                "aggregate_status": agg_status,
                "calibration_status": conf_res.calibration_status,
                "gate0_status": "pending_gate0",
                "uncertainty_flags": ",".join(flags),
                "disclaimer": "Computational uncertainty estimate; not experimentally validated error.",
            })

    uncertainty_df = pd.DataFrame(uncertainty_records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    uncertainty_df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Saved uncertainty table: {output_path} ({output_path.stat().st_size} bytes, {len(uncertainty_df)} rows)")

    # 7. Write uncertainty summary report
    report_lines = [
        "# Q-Catalyst Uncertainty Estimation & Conformal Calibration Report",
        "",
        "## 1. Overview and Setup",
        f"- **Parent Enzyme:** `{enzyme}`",
        f"- **Target Target:** `{target_name}`",
        f"- **Ensemble Architecture:** {n_ensemble} bootstrap XGBoost predictors",
        f"- **Calibration Scheme:** Split-conformal regression on independent single-site holdout (N={len(cal_df)})",
        f"- **Target Coverage:** {int((1.0 - alpha) * 100)}% (alpha = {alpha})",
        f"- **Conformal Quantile Threshold ($q$):** {calibrator.threshold_q:.4f}",
        f"- **Held-Out Empirical Coverage:** **{test_coverage_val * 100:.1f}%**",
        "",
        "## 2. Uncertainty Signals Summary (Held-Out Test Set)",
        "| Signal | Mean | Std | Min | Max | Missingness Policy |",
        "|---|---|---|---|---|---|",
    ]

    test_recs = uncertainty_df[uncertainty_df["split_role"] == "held_out_test"]
    for col, name in [
        ("u_ensemble_std", "Ensemble Std (Epistemic)"),
        ("u_conformal_width", "Conformal Interval Width"),
        ("u_ood_distance", "ESM-2 OOD Distance"),
        ("u_structural_risk", "Structural Proximity Risk"),
        ("aggregate_uncertainty", "Aggregate Uncertainty U(x)"),
    ]:
        series = test_recs[col].dropna()
        if not series.empty:
            report_lines.append(
                f"| `{name}` | {series.mean():.4f} | {series.std():.4f} | {series.min():.4f} | {series.max():.4f} | Strictly recorded / non-zero |"
            )
        else:
            report_lines.append(f"| `{name}` | N/A | N/A | N/A | N/A | Missing / Unmapped |")

    report_lines.extend([
        "",
        "## 3. Data Leakage & Calibration Safeguards",
        "1. **Strict Partitioning:** Conformal calibration used only data from the training pool (single-site mutants); the evaluation set was not observed during threshold calculation.",
        "2. **No Zero-Imputation:** Missing structural positions are explicitly marked `None` with `structural_signal_status='missing_unmapped'`, preventing false certainty.",
        "3. **Epistemic Disagreement:** Bootstrap ensemble standard deviation captures model uncertainty in combinatorial multi-mutant space.",
    ])

    report_text = "\n".join(report_lines)
    report_file = reports_dir / "uncertainty_report.md"
    report_file.write_text(report_text, encoding="utf-8")
    print(f"Saved summary report: {report_file} ({report_file.stat().st_size} bytes)")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Uncertainty pipeline completed in {elapsed:.2f}s.")
    print("=" * 70)

    return 0


def run_gate0_validation(
    uncertainty_path: Path,
    reports_dir: Path,
    target_name: str = "activity",
    enzyme: str = "IsPETase",
    n_bootstrap: int = 2000,
    random_seed: int = 42,
) -> int:
    """Run Feature 7 Gate 0 Uncertainty Validation on held-out predictions."""
    start_time = time.time()
    print("=" * 70)
    print("Q-CATALYST: Feature 7 - Gate 0 Uncertainty Validation")
    print("=" * 70)
    print(f"Parent Enzyme:        {enzyme}")
    print(f"Target Evaluated:     {target_name}")
    print(f"Uncertainty Input:    {uncertainty_path.resolve()}")
    print(f"Bootstrap Iterations: {n_bootstrap}")
    print(f"Decision Rule:        Spearman 95% CI lower bound > 0.0 to PASS")
    print("-" * 70)

    if not uncertainty_path.exists():
        print(f"Error: Uncertainty table not found at {uncertainty_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(uncertainty_path)
    # Filter for parent enzyme and target if present
    sub_df = df[
        (df["parent_enzyme"].str.lower() == enzyme.lower())
        & (df["target_name"].str.lower() == target_name.lower())
    ].copy()

    if sub_df.empty:
        sub_df = df.copy()

    # Gate 0 evaluation strictly on held-out variants
    result = evaluate_gate0(
        sub_df,
        target_name=target_name,
        enzyme=enzyme,
        n_bootstrap=n_bootstrap,
        seed=random_seed,
    )

    # Output file paths
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    plot_path = figures_dir / "gate0_coverage_plot.png"
    verdict_json_path = tables_dir / "gate0_verdict.json"
    summary_csv_path = tables_dir / "gate0_summary.csv"
    report_md_path = reports_dir / "gate0_validation_report.md"

    # Generate publication-grade figure
    generate_gate0_coverage_plot(
        sub_df,
        output_path=plot_path,
        target_name=target_name,
        enzyme=enzyme,
        result=result,
    )

    # Save artifacts
    save_gate0_artifacts(
        result=result,
        verdict_json_path=verdict_json_path,
        summary_csv_path=summary_csv_path,
        report_md_path=report_md_path,
    )

    # Update gate0_status in the uncertainty parquet for provenance
    try:
        held_out_mask = df["split_role"] == "held_out_test"
        df.loc[held_out_mask, "gate0_status"] = f"gate0_{result.verdict.lower()}"
        df.to_parquet(uncertainty_path, engine="pyarrow", index=False)
    except Exception:
        pass

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Gate 0 Validation Finished in {elapsed:.2f}s.")
    print("=" * 70)
    print(f"Evaluated Held-Out Variants: {result.n_held_out}")
    print(f"Spearman rho:                {result.spearman_rho:.4f} (p = {result.spearman_pvalue:.2e})")
    print(f"Bootstrap 95% CI:            [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print(f"CI Includes Zero:            {'YES (Failed planned criterion)' if result.ci_includes_zero else 'NO (Statistically robust correlation)'}")
    auc_display = f"{result.auroc:.4f}" if result.auroc is not None else "Undefined"
    print(f"Worst Quartile AUROC:        {auc_display} ({result.auroc_status})")
    print("-" * 70)
    print(f"GATE 0 VERDICT:              {result.verdict}")
    print(f"ROLE 4 DIRECTIVE:            {result.recommended_action}")
    print(f"Machine-readable verdict:    {verdict_json_path} ({verdict_json_path.stat().st_size} bytes)")
    print(f"Calibration/Coverage plot:   {plot_path} ({plot_path.stat().st_size} bytes)")
    print(f"Validation report:           {report_md_path} ({report_md_path.stat().st_size} bytes)")
    print("=" * 70)

    return 0


def run_benchmarking(
    variants_path: Path,
    predictions_path: Path,
    uncertainty_path: Path,
    benchmarks_dir: Path,
    reports_dir: Path,
    target_name: str = "activity",
    enzyme: str = "IsPETase",
    good_threshold: float = 1.0,
    random_seed: int = 42,
    n_bootstrap: int = 1000,
    mock_esm: bool = True,
) -> int:
    """Run Feature 8 benchmarking suite across arms A0, A1, A2 on held-out split."""
    start_time = time.time()
    print("=" * 70)
    print("Q-CATALYST: Feature 8 - Role 1 Benchmarking (Arms A0, A1, A2)")
    print("=" * 70)
    print(f"Parent Enzyme:          {enzyme}")
    print(f"Target Evaluated:       {target_name}")
    print(f"Good Variant Threshold: y_true > {good_threshold}")
    print(f"Bootstrap Iterations:   {n_bootstrap}")
    print(f"Random Seed:            {random_seed}")
    print("-" * 70)

    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = benchmarks_dir / "benchmark_results.parquet"
    tables_dir = reports_dir / "tables"
    csv_path = tables_dir / "benchmark_summary.csv"
    report_path = reports_dir / "benchmark_report.md"

    df_results = run_benchmark_suite(
        variants_path=variants_path,
        predictions_path=predictions_path,
        uncertainty_path=uncertainty_path,
        target_name=target_name,
        enzyme=enzyme,
        good_threshold=good_threshold,
        random_seed=random_seed,
        n_bootstrap=n_bootstrap,
        mock_esm=mock_esm,
    )

    save_benchmark_artifacts(
        benchmark_df=df_results,
        parquet_path=parquet_path,
        csv_path=csv_path,
        report_path=report_path,
    )

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Benchmarking completed in {elapsed:.2f}s.")
    print("=" * 70)
    print(f"Saved Parquet Output: {parquet_path} ({parquet_path.stat().st_size} bytes)")
    print(f"Saved Summary CSV:    {csv_path} ({csv_path.stat().st_size} bytes)")
    print(f"Saved Report:         {report_path} ({report_path.stat().st_size} bytes)")
    print("-" * 70)
    print("\nSummary Results (Precision & Enrichment):")
    eval_df = df_results[df_results["status"] == "evaluated"]
    p10_df = eval_df[eval_df["metric_name"] == "precision@10"]
    for _, r in p10_df.iterrows():
        ef10 = eval_df[(eval_df["arm_id"] == r["arm_id"]) & (eval_df["metric_name"] == "enrichment_factor@10")]["metric_value"].iloc[0]
        sp = eval_df[(eval_df["arm_id"] == r["arm_id"]) & (eval_df["metric_name"] == "spearman_rho")]["metric_value"].iloc[0]
        print(f"  [{r['arm_id']}] {r['arm_name']:42s} | P@10={r['metric_value']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}] | EF@10={ef10:.2f}x | Spearman={sp:.3f}")
    print("=" * 70)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q-Catalyst Protein AI Workflows (ESM Scoring, Embeddings, Predictors, Uncertainty, Gate 0, Benchmarking)"
    )
    # Task selectors
    parser.add_argument("--score-esm1v", action="store_true", help="Run ESM-1v mutation scoring")
    parser.add_argument("--extract-embeddings", action="store_true", help="Run ESM-2 embedding extraction")
    parser.add_argument("--train-predictor", action="store_true", help="Train supervised XGBoost and Ridge models")
    parser.add_argument("--estimate-uncertainty", action="store_true", help="Run uncertainty estimation and calibration")
    parser.add_argument("--validate-gate0", action="store_true", help="Run Gate 0 uncertainty validation and generate verdict")
    parser.add_argument("--run-benchmarks", action="store_true", help="Run benchmarking suite across arms A0, A1, A2")

    # Options
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "curated" / "variants.parquet",
        help="Path to curated variants table",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=REPO_ROOT / "data" / "splits",
        help="Directory to save frozen splits",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "results" / "models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "predictions" / "predictions.parquet",
        help="Path to output predictions table",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "embeddings",
        help="Directory to cache embeddings",
    )
    parser.add_argument(
        "--uncertainty-output",
        type=Path,
        default=REPO_ROOT / "results" / "uncertainty" / "uncertainty.parquet",
        help="Path to uncertainty parquet table",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "reports",
        help="Directory for reports and tables",
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=REPO_ROOT / "results" / "benchmarks",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="activity",
        choices=["activity", "tm"],
        help="Target to predict: activity or tm",
    )
    parser.add_argument(
        "--enzyme",
        type=str,
        default="IsPETase",
        help="Parent enzyme for training/eval",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=20,
        help="Max sequences to score or embed (default: 20)",
    )
    parser.add_argument(
        "--use-cached-model",
        action="store_true",
        help="Attempt to load real weights from cache_dir instead of mock proxy",
    )
    parser.add_argument(
        "--split-version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Split version: v2 (leakage-free, default) or v1 (historical contaminated)",
    )

    args = parser.parse_args()
    mock = not args.use_cached_model

    if args.run_benchmarks:
        code = run_benchmarking(
            variants_path=args.input,
            predictions_path=args.output,
            uncertainty_path=args.uncertainty_output,
            benchmarks_dir=args.benchmarks_dir,
            reports_dir=args.reports_dir,
            target_name=args.target,
            enzyme=args.enzyme,
            mock_esm=mock,
        )
    elif args.validate_gate0:
        code = run_gate0_validation(
            uncertainty_path=args.uncertainty_output,
            reports_dir=args.reports_dir,
            target_name=args.target,
            enzyme=args.enzyme,
        )
    elif args.estimate_uncertainty:
        code = run_uncertainty_estimation(
            variants_path=args.input,
            output_path=args.uncertainty_output,
            reports_dir=args.reports_dir,
            target_name=args.target,
            enzyme=args.enzyme,
            mock_embeddings=mock,
            split_version=args.split_version,
        )
    elif args.train_predictor:
        code = run_supervised_training(
            variants_path=args.input,
            splits_dir=args.splits_dir,
            models_dir=args.models_dir,
            predictions_path=args.output,
            reports_dir=args.reports_dir,
            target_name=args.target,
            enzyme=args.enzyme,
            mock_embeddings=mock,
            split_version=args.split_version,
        )
    elif args.extract_embeddings:
        code = run_embedding_extraction(
            input_path=args.input,
            output_dir=args.output_dir,
            enzyme=args.enzyme,
            max_variants=args.max_variants,
            mock=mock,
        )
    else:
        code = run_esm_scoring(
            input_path=args.input,
            output_path=args.output,
            enzyme=args.enzyme,
            max_variants=args.max_variants,
            mock=mock,
        )

    sys.exit(code)


if __name__ == "__main__":
    main()
