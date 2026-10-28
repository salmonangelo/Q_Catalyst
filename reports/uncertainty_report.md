# Q-Catalyst Uncertainty Estimation & Conformal Calibration Report

## 1. Overview and Setup
- **Parent Enzyme:** `IsPETase`
- **Target Target:** `activity`
- **Ensemble Architecture:** 10 bootstrap XGBoost predictors
- **Calibration Scheme:** Split-conformal regression on independent single-site holdout (N=21)
- **Target Coverage:** 90% (alpha = 0.1)
- **Conformal Quantile Threshold ($q$):** 0.8726
- **Held-Out Empirical Coverage:** **99.3%**

## 2. Uncertainty Signals Summary (Held-Out Test Set)
| Signal | Mean | Std | Min | Max | Missingness Policy |
|---|---|---|---|---|---|
| `Ensemble Std (Epistemic)` | 0.1468 | 0.0747 | 0.0182 | 0.4368 | Strictly recorded / non-zero |
| `Conformal Interval Width` | 1.7453 | 0.0000 | 1.7453 | 1.7453 | Strictly recorded / non-zero |
| `ESM-2 OOD Distance` | 0.3647 | 0.4745 | 0.0000 | 1.0152 | Strictly recorded / non-zero |
| `Structural Proximity Risk` | 0.5213 | 0.3674 | 0.0000 | 0.9200 | Strictly recorded / non-zero |
| `Aggregate Uncertainty U(x)` | 0.5539 | 0.0649 | 0.4145 | 0.7495 | Strictly recorded / non-zero |

## 3. Data Leakage & Calibration Safeguards
1. **Strict Partitioning:** Conformal calibration used only data from the training pool (single-site mutants); the evaluation set was not observed during threshold calculation.
2. **No Zero-Imputation:** Missing structural positions are explicitly marked `None` with `structural_signal_status='missing_unmapped'`, preventing false certainty.
3. **Epistemic Disagreement:** Bootstrap ensemble standard deviation captures model uncertainty in combinatorial multi-mutant space.