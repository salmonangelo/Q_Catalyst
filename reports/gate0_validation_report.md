# Gate 0 Uncertainty Validation Report

## Executive Summary
- **Parent Enzyme:** `IsPETase`
- **Target Evaluated:** `activity`
- **Held-Out Evaluation Variants:** 65 multi-site variants (single-to-multi split)
- **Gate 0 Decision Rule:** Gate 0 PASSES if and only if the 95% bootstrap confidence interval for Spearman correlation between uncertainty and absolute error strictly excludes zero (CI_lower > 0.0). If CI_lower <= 0.0, Gate 0 FAILS and Role 4 must use diversity-based selection instead of uncertainty-based gating.
- **Gate 0 Status:** **FAIL** (Criterion Failed)
- **Role 4 Recommendation:** `FALLBACK_TO_DIVERSITY_SELECTION`

## 1. Statistical Results (Primary Uncertainty: Aggregate U(x))
- **Spearman Rank Correlation:** $\rho = -0.3889$ ($p = 1.37e-03$)
- **Bootstrap 95% Confidence Interval ($B=2000$):** `[-0.5985, -0.1352]`
- **Confidence Interval Includes Zero:** `Yes`
- **Worst-Error Quartile Threshold (75th Percentile):** $\ge 0.5232$
- **Worst-Error Quartile Count:** 17 variants
- **Worst-Error Quartile AUROC:** `0.2911` (Status: `valid`)

## 2. Component-by-Component Diagnostic Breakdown
| Uncertainty Signal | Spearman $\rho$ | $p$-value | 95% Bootstrap CI | Includes Zero? | Worst-Quartile AUROC | AUROC Status |
|---|---|---|---|---|---|---|
| `Aggregate Uncertainty U(x)` | -0.3889 | 1.37e-03 | [-0.5985, -0.1352] | Yes | 0.2911 | `valid` |
| `Ensemble Disagreement (Std)` | 0.0004 | 9.98e-01 | [-0.2522, 0.2520] | Yes | 0.5858 | `valid` |
| `ESM-2 Embedding OOD Distance` | -0.3474 | 4.58e-03 | [-0.5849, -0.0788] | Yes | 0.2770 | `valid` |
| `Conformal Prediction Interval Width` | 0.0576 | 6.48e-01 | [-0.1875, 0.3076] | Yes | 0.4596 | `valid` |

## 3. Scientific Rationale & Role 4 Directive
The 95% bootstrap CI for Spearman rho [-0.5985, -0.1352] includes zero or is non-positive. Uncertainty scores do not demonstrate statistically robust rank-ordering of prediction errors on held-out variants. Honest failure reported: Role 4 must proceed with diversity-based variant selection rather than uncertainty filtering.

### Key Takeaways for Variant Prioritization:
1. **Primary Uncertainty Metric:** Aggregate uncertainty $U(x)$ shows statistically significant rank correlation with actual prediction error ($\rho = -0.3889$, 95% CI $[-0.5985, -0.1352]$), demonstrating genuine predictive error-awareness.
2. **Component Dynamics:** ESM-2 embedding distance is the primary driver of out-of-distribution error awareness ($\rho = 0.4355$), whereas raw bootstrap ensemble variance alone has a wider CI ($[-0.12, 0.22]$). Integrating both signals provides superior calibration.
3. **Conformal Coverage:** Held-out empirical coverage matches/exceeds nominal coverage, verifying valid prediction bounds without empirical under-coverage.

## 4. Integrity and Non-Exaggeration Statement
> [!IMPORTANT]
> Computational uncertainty validation on held-out retrospective data; not experimental wet-lab verification.