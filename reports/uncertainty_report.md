# Q-Catalyst Uncertainty Estimation & Conformal Calibration Report

## 1. Overview and Setup
- **Parent Enzyme:** `IsPETase`
- **Target Target:** `activity`
- **Ensemble Architecture:** 10 bootstrap XGBoost predictors
- **Calibration Scheme:** Split-conformal regression on independent single-site holdout (N=21)
- **Target Coverage:** 90% (alpha = 0.1)
- **Conformal Quantile Threshold ($q$):** 0.7825
- **Held-Out Empirical Coverage:** **100.0%**

## 2. Uncertainty Signals Summary (Held-Out Test Set)
| Signal | Mean | Std | Min | Max | Missingness Policy |
|---|---|---|---|---|---|
| `Ensemble Std (Epistemic)` | 0.0779 | 0.0164 | 0.0475 | 0.1504 | Strictly recorded / non-zero |
| `Conformal Interval Width` | 1.5650 | 0.0000 | 1.5650 | 1.5650 | Strictly recorded / non-zero |
| `ESM-2 OOD Distance` | 1.5109 | 0.3290 | 0.5020 | 2.0246 | Strictly recorded / non-zero |
| `Structural Proximity Risk` | 0.8512 | 0.1536 | 0.5134 | 0.9200 | Strictly recorded / non-zero |
| `Aggregate Uncertainty U(x)` | 0.6134 | 0.0311 | 0.5067 | 0.6568 | Strictly recorded / non-zero |

## 3. Data Leakage & Calibration Safeguards
1. **Strict Partitioning:** Conformal calibration used only data from the training pool (single-site mutants); the evaluation set was not observed during threshold calculation.
2. **No Zero-Imputation:** Missing structural positions are explicitly marked `None` with `structural_signal_status='missing_unmapped'`, preventing false certainty.
3. **Epistemic Disagreement:** Bootstrap ensemble standard deviation captures model uncertainty in combinatorial multi-mutant space.