# Role 1 to Role 4 Technical Handoff Document

**From:** Role 1 (Protein ML Lead)  
**To:** Role 4 (Integration, UI and Pitch Lead)  
**Date:** October 2026  
**Status:** Feature 1–8 Implementation Complete & Verified

---

## 1. Executive Summary

Role 1 has completed the protein AI, uncertainty quantification, Gate 0 validation, and baseline benchmarking pipeline for Q-Catalyst. All outputs use reproducible seeds, leakage-safe frozen splits, and strict schema contracts.

### Key Milestones Delivered:
1. **Curated Variants Table:** `data/curated/variants.parquet` (430 enzyme variants, canonical 5XJH structural alignment, quality flags).
2. **ESM-1v Zero-Shot Scoring:** `results/predictions/esm1v_scores.parquet` (masked-marginal log-likelihood ratios).
3. **ESM-2 Embeddings:** `results/embeddings/embedding_index.parquet` (480-dimensional mean-pooled representations with SHA-256 caching).
4. **Supervised Predictions:** `results/predictions/predictions.parquet` (XGBoost primary model vs Ridge baseline comparator on separate activity and $T_m$ targets).
5. **Uncertainty Quantification:** `results/uncertainty/uncertainty.parquet` (10-member bootstrap ensemble disagreement, split-conformal intervals, ESM-2 OOD distance, 5XJH catalytic triad structural risk).
6. **Gate 0 Validation:** `reports/tables/gate0_verdict.json` (**VERDICT: PASS**, $\rho = 0.2632$, 95% CI $[0.0848, 0.4217] > 0.0$).
7. **Benchmarking Arms A0–A2:** `results/benchmarks/benchmark_results.parquet` (Precision@5/10/20, EF@10, NDCG@10, Spearman/Kendall with bootstrap 95% CIs).

---

## 2. File Locations & Data Artifacts

| Artifact | File Path | Format | Rows | Description |
|---|---|---|---|---|
| **Curated Variants** | `data/curated/variants.parquet` | Parquet | 430 | Canonical variants with structure-mapped positions |
| **Model Predictions** | `results/predictions/predictions.parquet` | Parquet | 466 | XGBoost & Ridge predictions on train & test |
| **Uncertainty Table** | `results/uncertainty/uncertainty.parquet` | Parquet | 233 | Split-conformal intervals & composite uncertainty $U(x)$ |
| **Gate 0 Verdict** | `reports/tables/gate0_verdict.json` | JSON | 1 | Machine-readable validation verdict & statistics |
| **Gate 0 Summary** | `reports/tables/gate0_summary.csv` | CSV | 4 | Component-by-component diagnostic breakdown |
| **Gate 0 Plot** | `reports/figures/gate0_coverage_plot.png` | PNG (300 DPI) | — | 4-panel calibration, error correlation & scorecard |
| **Benchmark Results** | `results/benchmarks/benchmark_results.parquet` | Parquet | 38 | Arms A0, A1, A2 evaluated with bootstrap CIs |
| **Benchmark Summary** | `reports/tables/benchmark_summary.csv` | CSV | 38 | Tabular benchmark metrics across arms |

---

## 3. Authoritative Schema Contracts

### 3.1 Supervised Predictions Schema (`predictions.parquet`)
```python
{
    "row_id": "str (e.g. 'petgym_ispetase_mutant_12')",
    "variant_id": "str (stable unique ID: {parent}_{canonical_mutations})",
    "parent_enzyme": "str ('IsPETase', 'BhrPETase', 'CaPETase')",
    "dataset_split": "str ('single-site-mutant', 'low-to-high-mutant')",
    "split_role": "str ('train', 'held_out_test')",
    "mutant_raw": "str (original mutation notation from study)",
    "mutations_canonical": "str (sorted canonical notation, e.g. 'S121E:D186H')",
    "target_name": "str ('activity' or 'tm')",
    "target_unit": "str ('relative_fold_change' or 'deg_C')",
    "y_true": "float (ground truth experimental label)",
    "y_pred_xgb": "float (primary supervised XGBoost prediction)",
    "y_pred_ridge": "float (linear baseline comparison prediction)",
    "residual_xgb": "float (y_true - y_pred_xgb)",
    "residual_ridge": "float (y_true - y_pred_ridge)",
    "model_id": "str ('xgb_regressor_v1')",
    "baseline_id": "str ('ridge_regressor_v1')",
    "prediction_status": "str ('success')",
    "is_prediction": "bool (True)",
    "disclaimer": "str"
}
```

### 3.2 Uncertainty Quantification Schema (`uncertainty.parquet`)
```python
{
    "row_id": "str",
    "variant_id": "str",
    "parent_enzyme": "str",
    "dataset_split": "str",
    "split_role": "str ('model_train', 'calibration', 'held_out_test')",
    "target_name": "str ('activity' or 'tm')",
    "y_true": "float",
    "y_pred_mean": "float (mean ensemble prediction)",
    "u_ensemble_std": "float (epistemic disagreement across 10 bootstrap XGBoost members)",
    "conformal_lower": "float (lower bound of 90% split-conformal prediction interval)",
    "conformal_upper": "float (upper bound of 90% split-conformal prediction interval)",
    "u_conformal_width": "float (conformal interval width = conformal_upper - conformal_lower)",
    "conformal_covered": "bool (whether y_true is within [conformal_lower, conformal_upper])",
    "u_ood_distance": "float (normalized ESM-2 embedding Mahalanobis-style distance)",
    "ood_status": "str ('in_distribution' or 'ood_warning')",
    "u_structural_risk": "float or None (exponential proximity risk to 5XJH catalytic triad Ser160/His237/Asp206)",
    "structural_signal_status": "str ('available' or 'missing_unmapped')",
    "aggregate_uncertainty": "float in [0, 1] (composite uncertainty U(x))",
    "aggregate_status": "str ('valid', 'incomplete_signals', or 'all_signals_missing')",
    "calibration_status": "str ('calibrated')",
    "gate0_status": "str ('gate0_pass')",
    "uncertainty_flags": "str ('clean', 'ood_warning', 'missing_structure')",
    "disclaimer": "str"
}
```

### 3.3 Benchmark Results Schema (`benchmark_results.parquet`)
```python
{
    "arm_id": "str ('A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7')",
    "arm_name": "str",
    "target_name": "str ('activity')",
    "parent_enzyme": "str ('IsPETase')",
    "split_id": "str ('single_to_multi')",
    "split_role": "str ('held_out_test')",
    "dataset_split": "str ('low-to-high-mutant')",
    "n_samples": "int (149)",
    "base_rate": "float (0.1208 = 18/149 hits)",
    "good_variant_threshold": "float (1.0)",
    "metric_name": "str ('precision@5', 'precision@10', 'precision@20', 'enrichment_factor@10', 'ndcg@10', 'spearman_rho', 'kendall_tau')",
    "metric_value": "float (point estimate)",
    "ci_lower": "float (bootstrap 95% CI lower bound)",
    "ci_upper": "float (bootstrap 95% CI upper bound)",
    "ci_level": "float (0.95)",
    "n_bootstrap": "int (1000)",
    "random_seed": "int (42)",
    "compute_cost_cpu_seconds": "float",
    "status": "str ('evaluated' or 'pending_role2_role3_results')",
    "provenance": "str",
    "disclaimer": "str"
}
```

---

## 4. Gate 0 Verdict & Role 4 Directives

* **Gate 0 Result:** **`PASS`**
* **Key Metric:** Spearman rank correlation between composite uncertainty $U(x)$ and absolute prediction error $|y_{\text{true}} - \hat{y}|$:
  * Point estimate: $\rho = +0.2632$ ($p = 1.18 \times 10^{-3}$)
  * Bootstrap 95% CI ($B=2000$): $[+0.0848,\ +0.4217]$ (strictly excludes zero)
* **Empirical Split-Conformal Coverage:** $99.3\%$ (148/149 held-out variants covered at 90% nominal confidence)
* **Role 4 Directive:** **`PROCEED_WITH_UNCERTAINTY_GATING`**
  * You are authorized to use uncertainty-guided candidate selection in the acquisition function.
  * Role 4 does **not** need to fall back to diversity-only selection.

---

## 5. Guidance for Role 4 Acquisition & UI Integration

### 5.1 Acquisition Function Formulation
Role 4 should implement the pre-registered acquisition function:
$$a(x) = \lambda U(x) + \eta P(x) + \mu D(x) - \nu C(x)$$

* **Recommended Normalized Weights:**
  * $\lambda = 0.40$: Uncertainty score $U(x) \in [0, 1]$ (`aggregate_uncertainty`).
  * $\eta = 0.30$: Predicted activity $P(x)$ (standardized from `y_pred_mean`).
  * $\mu = 0.20$: Sequence/embedding diversity $D(x)$ (e.g. minimum distance to already selected candidates).
  * $\nu = 0.10$: Computational cost $C(x)$ (Tier S vs Tier M PySCF cluster runtime estimate).

### 5.2 Structural Proximity Filter
Before passing candidates to classical chemistry (Role 2):
* Filter candidates by minimum distance $R$ to the catalytic triad (IsPETase Ser160 Oγ or modelled substrate ester carbon):
  * Primary inclusion cutoff: $R \le 8\text{ \AA}$ (filter options: $6, 8, 10, 12\text{ \AA}$).
  * Candidates outside the cutoff retain their AI-only rank and are marked `not_eligible_for_chemistry`.

### 5.3 Loading Artifacts in Python / Streamlit
```python
import pandas as pd
import json

# 1. Load predictions & uncertainties
u_df = pd.read_parquet("results/uncertainty/uncertainty.parquet")
test_variants = u_df[u_df["split_role"] == "held_out_test"]

# 2. Load Gate 0 Verdict
with open("reports/tables/gate0_verdict.json", "r") as f:
    gate0_verdict = json.load(f)

# 3. Load Benchmark Comparisons
benchmarks = pd.read_parquet("results/benchmarks/benchmark_results.parquet")
evaluated_benchmarks = benchmarks[benchmarks["status"] == "evaluated"]

# Display in Streamlit or Acquisition Engine
print(f"Gate 0 Status: {gate0_verdict['verdict']} -> {gate0_verdict['recommended_action']}")
print(f"Top Candidate P@10 (A2): {benchmarks[(benchmarks['arm_id']=='A2') & (benchmarks['metric_name']=='precision@10')]['metric_value'].iloc[0]}")
```

---

## 6. Scientific Integrity & Non-Claims Boundary
All predictions and scores are computational prioritisations. Role 4 must include the standard project disclaimer on all evidence cards and dashboard views:
> **Disclaimer:** Computational model prioritisation based on retrospective data; not experimentally verified increase in PET plastic degradation.
