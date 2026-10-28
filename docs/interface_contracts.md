# Q-Catalyst Authoritative Interface Contracts

**Version:** 1.0.0  
**Effective Date:** October 2026  
**Scope:** Shared data structures, file locations, and serialization schemas across all roles.

---

## 1. Overview of File Contracts

| File Path | Owner | Consumers | Format | Purpose |
|---|---|---|---|---|
| `data/curated/variants.parquet` | Role 1 | Roles 2, 4 | Parquet | Canonical variants with structure-mapped positions |
| `results/predictions/predictions.parquet` | Role 1 | Role 4 | Parquet | Supervised model predictions (XGBoost, Ridge) |
| `results/uncertainty/uncertainty.parquet` | Role 1 | Role 4 | Parquet | Split-conformal bounds and composite uncertainty $U(x)$ |
| `results/benchmarks/benchmark_results.parquet` | Role 1 | Role 4 | Parquet | Prioritisation benchmark metrics across arms A0–A7 |
| `reports/tables/gate0_verdict.json` | Role 1 | Role 4 | JSON | Gate 0 decision verdict and validation statistics |
| `chem_jobs.json` | Role 4 | Role 2 | JSON | Ordered list of candidate variants for classical chemistry |
| `results/chemistry/chem_results.parquet` | Role 2 | Roles 3, 4 | Parquet | PySCF descriptors, cluster energies, and CASCI inputs |
| `results/quantum/quantum_results.parquet` | Role 3 | Role 4 | Parquet | Active-space CASCI vs VQE energies and error metrics |
| `results/ranking/ranking.parquet` | Role 4 | Dashboard | Parquet | Final fused multi-objective ranking and confidence tiers |

---

## 2. Detailed Schemas

### 2.1 `variants.parquet`
- `row_id` (str): Unique record ID (`petgym_{enzyme}_{mutant_type}_{idx}`).
- `variant_id` (str): Unique variant ID (`{parent}_{canonical_mutations}`).
- `parent_enzyme` (str): Wild-type enzyme identifier (`IsPETase`, `BhrPETase`, `CaPETase`).
- `dataset_split` (str): Dataset split category (`single-site-mutant`, `low-to-high-mutant`).
- `study_id` (str): Originating research publication / dataset identifier.
- `mutant_raw` (str): Original notation in source publication.
- `mutations_canonical` (str): Colon-separated, position-sorted mutations.
- `num_mutations` (int): Count of point mutations relative to wild-type.
- `is_multi_mutant` (bool): True if `num_mutations > 1`.
- `wildtype_sequence` (str): Full wild-type amino-acid sequence.
- `mutant_sequence` (str): Complete mutated sequence.
- `structure_5xjh_positions` (str): Mapped residues in 5XJH crystal structure numbering.
- `structure_mapping_status` (str): `exact_1to1` or `unmapped`.
- `activity_raw` (float or null): Experimental activity (relative fold change for IsPETase).
- `activity_unit` (str): Unit of activity measurement.
- `tm_raw` (float or null): Experimental melting temperature in °C.
- `tm_unit` (str): Unit of thermostability measurement (`deg_C`).
- `is_duplicate` (bool): Flag indicating duplicate sequence in dataset.
- `is_outlier` (bool): Flag indicating unphysical outlier.
- `data_quality_flags` (str): Comma-separated quality flags (`clean`, `duplicate`, `outlier`).

### 2.2 `predictions.parquet`
- `row_id` (str), `variant_id` (str), `parent_enzyme` (str), `dataset_split` (str).
- `split_role` (str): `train` or `held_out_test`.
- `target_name` (str): `activity` or `tm`.
- `y_true` (float): Ground truth label.
- `y_pred_xgb` (float): XGBoost predicted value.
- `y_pred_ridge` (float): Ridge regression predicted value.
- `residual_xgb` (float): Residual ($y_{\text{true}} - \hat{y}_{\text{xgb}}$).
- `residual_ridge` (float): Residual ($y_{\text{true}} - \hat{y}_{\text{ridge}}$).
- `model_id` (str), `baseline_id` (str), `prediction_status` (str), `is_prediction` (bool).
- `disclaimer` (str): Non-claims integrity statement.

### 2.3 `uncertainty.parquet`
- `row_id` (str), `variant_id` (str), `split_role` (str), `target_name` (str), `y_true` (float).
- `y_pred_mean` (float): Ensemble mean prediction.
- `u_ensemble_std` (float): Disagreement across bootstrap ensemble members.
- `conformal_lower` (float): Lower bound of 90% split-conformal interval.
- `conformal_upper` (float): Upper bound of 90% split-conformal interval.
- `u_conformal_width` (float): Width of conformal interval.
- `conformal_covered` (bool): True if $y_{\text{true}} \in [y_{\text{lower}}, y_{\text{upper}}]$.
- `u_ood_distance` (float): ESM-2 embedding distance to training distribution.
- `ood_status` (str): `in_distribution` or `ood_warning`.
- `u_structural_risk` (float or null): Active-site proximity risk (5XJH Ser160/His237/Asp206).
- `structural_signal_status` (str): `available` or `missing_unmapped`.
- `aggregate_uncertainty` (float in $[0, 1]$): Composite uncertainty score $U(x)$.
- `gate0_status` (str): `gate0_pass` or `gate0_fail`.

### 2.4 `benchmark_results.parquet`
- `arm_id` (str): `A0`, `A1`, `A2`, `A3`, `A4`, `A5`, `A6`, `A7`.
- `arm_name` (str): Human-readable strategy name.
- `target_name` (str), `parent_enzyme` (str), `split_id` (str), `split_role` (str), `dataset_split` (str).
- `n_samples` (int), `base_rate` (float), `good_variant_threshold` (float).
- `metric_name` (str): `precision@5`, `precision@10`, `precision@20`, `enrichment_factor@10`, `ndcg@10`, `spearman_rho`, `kendall_tau`.
- `metric_value` (float): Metric point estimate.
- `ci_lower` (float), `ci_upper` (float): Bootstrap 95% confidence interval bounds.
- `ci_level` (float: 0.95), `n_bootstrap` (int: 1000), `random_seed` (int: 42).
- `compute_cost_cpu_seconds` (float).
- `status` (str): `evaluated` (Arms A0–A2) or `pending_role2_role3_results` (Arms A3–A7).
- `provenance` (str), `disclaimer` (str).
