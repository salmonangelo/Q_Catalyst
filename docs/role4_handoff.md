# Role 1 to Role 4 Technical Handoff Document

**From:** Role 1 (Protein ML Lead)  
**To:** Role 4 (Integration, UI and Pitch Lead)  
**Date:** October 2026 (Updated for Real Pretrained Models & Leakage-Free Split v2)  
**Status:** Prior Role 1 artifacts exist; feature branch adds integration hardening and an end-to-end runner. Fresh branch verification is pending.

---

## 1. Executive Summary

Prior execution reports describe real ESM-2 35M and ESM-1v 650M outputs, supervised predictions, uncertainty estimates, Gate 0, and A0–A2 benchmarks. The feature branch now adds a full runner and fail-safe policy bridge; those new integrated changes must be executed and reviewed before this handoff is treated as a fresh end-to-end verification.

### Critical Rigour Milestone: Split Correction (v1 vs v2)
During validation, Role 1 audited the experimental splits and discovered that the historical PET-Gym dataset split (`low-to-high-mutant`, $N=149$) contained all $84$ single mutants plus $65$ combinatorial multi-mutants. Evaluating models trained on single mutants against the old $N=149$ split created **56.4% data leakage** (84/149 test variants were in the training set).
Role 1 immediately created versioned, leakage-free splits (`v2`):
- **Training pool:** Exactly $N=84$ single-site mutants.
- **Held-out evaluation pool:** Exactly $N=65$ combinatorial multi-mutants ($0$ ID or sequence overlap).
- **Historical split preservation:** Old v1 artifacts are archived in `*_v1_historical.parquet` for reproducibility.

### Key Milestones Delivered:
1. **Curated Variants Table:** `data/curated/variants.parquet` (430 enzyme variants, canonical 5XJH structural alignment, quality flags).
2. **Real ESM-1v Zero-Shot Scoring:** `results/predictions/esm1v_scores.parquet` (genuine masked-marginal log-likelihood ratios from official `facebook/esm1v_t33_650M_UR90S_1` 2.49 GB checkpoint, evaluated on CPU in eval mode).
3. **Real ESM-2 Embeddings:** `results/embeddings/embedding_index.parquet` and `results/embeddings/real_facebook_esm2_t12_35M_UR50D/` (genuine 480-dimensional mean-pooled representations with attention-mask residue-only pooling and SHA-256 caching for all 149 unique variants).
4. **Supervised Predictions on Split v2:** `results/predictions/predictions.parquet` (XGBoost primary model vs Ridge baseline comparator on separate activity and $T_m$ targets).
5. **Uncertainty Quantification on Split v2:** `results/uncertainty/uncertainty.parquet` (10-member bootstrap ensemble disagreement, split-conformal intervals, normalized ESM-2 OOD distance, 5XJH catalytic triad structural risk).
6. **Gate 0 Validation:** The previous artifact reports `FAIL`, $\rho = -0.3889$, 95% CI $[-0.5985, -0.1352]$ on the 65-variant held-out pool. The CI is entirely negative and therefore does **not** span zero; it fails because the planned criterion requires a strictly positive lower bound. The required action is `FALLBACK_TO_DIVERSITY_SELECTION`.
7. **Benchmarking Arms A0–A2:** `results/benchmarks/benchmark_results.parquet` (Precision@5/10/20, EF@10, NDCG@10, Spearman/Kendall with 1000 bootstrap CIs comparing A0 Random, A1 Real ESM-1v, and A2 Supervised XGBoost).

---

## 2. File Locations & Data Artifacts

| Artifact | File Path | Format | Rows | Description |
|---|---|---|---|---|
| **Curated Variants** | `data/curated/variants.parquet` | Parquet | 430 | Canonical variants with structure-mapped positions |
| **Real ESM-2 Embeddings** | `results/embeddings/real_facebook_esm2_t12_35M_UR50D/` | NPZ/Parquet | 149 | 480-d mean-pooled sequence embeddings |
| **Real ESM-1v Predictions** | `results/predictions/esm1v_scores.parquet` | Parquet | 65 | Real masked-marginal zero-shot LLR scores |
| **Model Predictions** | `results/predictions/predictions.parquet` | Parquet | 298 | XGBoost & Ridge predictions on train & held-out test (v2) |
| **Uncertainty Table** | `results/uncertainty/uncertainty.parquet` | Parquet | 149 | Split-conformal intervals & composite uncertainty $U(x)$ (v2) |
| **Gate 0 Verdict** | `reports/tables/gate0_verdict.json` | JSON | 1 | Machine-readable validation verdict & statistics |
| **Gate 0 Summary** | `reports/tables/gate0_summary.csv` | CSV | 4 | Component-by-component diagnostic breakdown |
| **Gate 0 Plot** | `reports/figures/gate0_coverage_plot.png` | PNG (300 DPI) | — | 4-panel calibration, error correlation & scorecard |
| **Benchmark Results** | `results/benchmarks/benchmark_results.parquet` | Parquet | 33 | Arms A0, A1, A2 evaluated with bootstrap CIs on held-out v2 |
| **Benchmark Summary** | `reports/tables/benchmark_summary.csv` | CSV | 33 | Tabular benchmark metrics across arms |

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
    "u_ood_distance": "float (normalized ESM-2 embedding distance)",
    "ood_status": "str ('in_distribution' or 'ood_warning')",
    "u_structural_risk": "float or None (proximity risk to 5XJH catalytic triad Ser160/His237/Asp206)",
    "structural_signal_status": "str ('available' or 'missing_unmapped')",
    "aggregate_uncertainty": "float in [0, 1] (composite uncertainty U(x))",
    "aggregate_status": "str ('valid', 'incomplete_signals', or 'all_signals_missing')",
    "calibration_status": "str ('calibrated')",
    "gate0_status": "str ('gate0_fail')",
    "uncertainty_flags": "str ('clean', 'ood_warning', 'missing_structure')",
    "disclaimer": "str"
}
```

---

## 4. Gate 0 Verdict & Scientific Story for the Hackathon Pitch

* **Gate 0 Result:** **`FAIL`**
* **Key Metric:** Spearman rank correlation between composite uncertainty $U(x)$ and absolute prediction error $|y_{\text{true}} - \hat{y}|$:
  * Point estimate on genuine held-out multi-mutants ($N=65$): $\rho = -0.3889$ ($p = 1.34 \times 10^{-3}$)
  * Bootstrap 95% CI ($B=2000$): $[-0.5985,\ -0.1352]$ (strictly negative)
* **Previous-run empirical split-conformal coverage:** $100\%$ (65/65 held-out variants covered at $q=0.7825$; symmetric interval width about 1.565). This is observed coverage on a small retrospective pool, not proof of useful or prospective calibration.
* **Role 4 Directive:** **`FALLBACK_TO_DIVERSITY_SELECTION`**

### Interpretation for the project review:
1. The historical v1 split included single-site variants from the training pool in its evaluation pool; results from that split are not valid held-out estimates.
2. On the corrected v2 pool, the reported uncertainty/error correlation is negative. This is a warning about the current uncertainty signal under mutational-depth shift; it does not show that the signal is calibrated.
3. The pre-registered fallback is therefore important: `FALLBACK_TO_DIVERSITY_SELECTION` prevents the system from using an unvalidated uncertainty score to exclude candidates. This is evidence of a useful guardrail, not proof of production robustness.

---

## 5. Guidance for Role 4 Acquisition & UI Integration

### 5.1 Active Acquisition Function Under Gate 0 Fallback
Since Gate 0 triggered `FALLBACK_TO_DIVERSITY_SELECTION`, Role 4 should configure the acquisition function with diversity prioritization:
$$a(x) = \mu D(x) + \eta P(x) - \nu C(x)$$

* **Recommended Fallback Weights:**
  * $\mu = 0.50$: Sequence/embedding diversity $D(x)$ (Euclidean/cosine distance in real ESM-2 480-d latent space from previously selected mutants).
  * $\eta = 0.35$: Predicted activity $P(x)$ (standardized from `y_pred_mean`).
  * $\nu = 0.15$: Computational cost $C(x)$ (Tier S vs Tier M PySCF cluster runtime estimate).
  * $\lambda = 0.00$: Uncertainty gating weight suppressed per Gate 0 directive.

### 5.2 Comparative Prioritisation Benchmarks (Split v2 Held-Out $N=65$)
| Arm ID | Strategy Name | Precision@5 | Precision@10 | Precision@20 | EF@10 | NDCG@10 | Spearman $\rho$ |
|---|---|---|---|---|---|---|---|
| **A0** | Random Floor Baseline | 0.00 | 0.00 | 0.00 | 0.00x | 0.405 | -0.005 |
| **A1** | Real ESM-1v Zero-Shot Ranking | 0.00 | 0.00 | 0.00 | 0.00x | 0.324 | -0.387 |
| **A2** | Supervised XGBoost + Structural | **0.40** | **0.20** | **0.15** | **1.86x** | **0.406** | **0.107** |

*These numbers are from a previous artifact snapshot. The integration branch changes the A0 ranking and A2 prediction source; regenerate the benchmark before presenting any A0–A2 metrics as results from the current branch.*

---

## 6. Scientific Integrity & Non-Claims Boundary
All predictions, zero-shot scores, and rankings are computational prioritisations based on retrospective PET-Gym data. Role 4 must include the standard project disclaimer on all evidence cards and UI views:
> **Disclaimer:** Computational model prioritisation based on retrospective data; not experimentally verified increase in PET plastic degradation.
