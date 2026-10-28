# Q-Catalyst Supervised PETase Prediction Report

## 1. Executive Summary

This report documents the evaluation of the primary supervised learning model (**XGBoost**) against the simple linear baseline comparator (**Ridge Regression**) on the canonical `IsPETase` benchmark dataset from PET-Gym.

- **Feature Representations:** 480-dimensional sequence embeddings from ESM-2 (`facebook/esm2_t12_35M_UR50D`, mean-pooled).
- **Target Separation:** Activity (`activity_raw`, relative fold-change) and Melting Temperature (`tm_raw`, in °C) are strictly trained and evaluated as separate models.
- **Split Strategy:** Canonical single-to-multi generalization split:
  - **Training Set:** `single-site-mutant` (84 variants).
  - **Held-Out Evaluation Set:** `low-to-high-mutant` (149 combinatorial multi-site variants).
  - **Data Leakage Prevention:** Feature scaling (`StandardScaler`) was strictly fitted only on the training set.
  - **Frozen Split Record:** Saved at `data/splits/ispetase_activity_single_to_multi.json` and `data/splits/ispetase_tm_single_to_multi.json`.

---

## 2. Held-Out Evaluation Results (Generalization to Multi-Site Mutants)

### Task A: Catalytic Activity (`activity_raw`)
*Evaluation on 149 held-out multi-mutant IsPETase variants:*

| Metric | XGBoost (Primary) | Ridge (Baseline Comparator) | Improvement |
|---|---|---|---|
| **MAE** | **0.2257** | 0.5203 | **-56.6% error** |
| **RMSE** | **0.3210** | 0.8130 | **-60.5% error** |
| **Spearman $\rho$ (Rank)** | **0.5535** ($p = 2.45 \times 10^{-13}$) | 0.2493 ($p = 2.17 \times 10^{-3}$) | **+122.0% rank correlation** |
| **Pearson $r$ (Linear)** | **0.6120** ($p = 1.08 \times 10^{-16}$) | 0.2312 ($p = 4.58 \times 10^{-3}$) | **+164.7% linear correlation** |
| **$R^2$** | **0.3644** | -3.0783 | Substantial non-linear fit |

### Task B: Thermostability / Melting Temperature (`tm_raw`, °C)
*Evaluation on 149 held-out multi-mutant IsPETase variants:*

| Metric | XGBoost (Primary) | Ridge (Baseline Comparator) | Improvement |
|---|---|---|---|
| **MAE (°C)** | **10.3317** | 11.4134 | **-9.5% error** |
| **RMSE (°C)** | **15.9019** | 18.1770 | **-12.5% error** |
| **Spearman $\rho$ (Rank)** | **0.3697** ($p = 3.49 \times 10^{-6}$) | 0.1760 ($p = 3.18 \times 10^{-2}$) | **+110.1% rank correlation** |
| **Pearson $r$ (Linear)** | **0.3496** ($p = 1.18 \times 10^{-5}$) | 0.1691 ($p = 3.93 \times 10^{-2}$) | **+106.7% linear correlation** |

---

## 3. Scientific Findings & Non-Claims

1. **Non-Linear Advantage:**
   XGBoost demonstrates substantial ranking and error superiority over Ridge regression when generalizing from single-site to multi-site mutants ($\rho = 0.55$ vs $0.25$ for activity).
2. **Target Incomparability:**
   Activity measurements across other enzymes (`BhrPETase`, `CaPETase`) use different assay methods and rates and must not be pooled without calibrated experimental conversion.
3. **Computational Boundary:**
   All model outputs are computational prioritisation scores. They **must not be treated or reported as experimentally validated increases in PET degradation**.

---

## 4. Artifact & Model Provenance

- **Saved Models:**
  - `results/models/ispetase_activity/` (`xgb_model.json`, `ridge_model.json`, `scaler.json`, `config.json`)
  - `results/models/ispetase_tm/` (`xgb_model.json`, `ridge_model.json`, `scaler.json`, `config.json`)
- **Canonical Predictions:** `results/predictions/predictions.parquet` (466 records across train and held-out sets).
- **Machine-Readable Metrics:**
  - `reports/tables/supervised_ispetase_activity_metrics.json`
  - `reports/tables/supervised_ispetase_tm_metrics.json`
- **Frozen Split Assignments:**
  - `data/splits/ispetase_activity_single_to_multi.json`
  - `data/splits/ispetase_tm_single_to_multi.json`
