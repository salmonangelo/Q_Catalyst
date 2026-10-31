# Gate 0 Uncertainty Validation Report

> **Interpretation correction:** The reported confidence interval is entirely negative. It does **not** include zero. Gate 0 still fails because the pre-registered criterion requires a strictly positive lower confidence bound.

## Executive Summary

- **Parent enzyme:** `IsPETase`
- **Target:** activity
- **Held-out evaluation:** 65 multi-site variants from the corrected v2 single-to-multi split.
- **Pre-registered pass rule:** the 95% bootstrap confidence interval for Spearman correlation between aggregate uncertainty and absolute prediction error must have `ci_lower > 0`.
- **Verdict:** **FAIL**
- **Required downstream action:** `FALLBACK_TO_DIVERSITY_SELECTION`

## Primary diagnostic

| Quantity | Reported value |
|---|---:|
| Spearman correlation, uncertainty vs. absolute error | -0.3889 |
| p-value | 1.37e-03 |
| 95% bootstrap confidence interval (2,000 resamples) | [-0.5985, -0.1352] |
| Does the interval span zero? | No |
| Is the association positive? | No; the interval is entirely negative |
| Worst-error-quartile AUROC | 0.2911 |
| Worst-error-quartile threshold | 0.5232 |
| Variants in worst-error quartile | 17 |

The negative interval indicates an inverse association in this particular retrospective evaluation: variants assigned larger aggregate uncertainty scores tended to have smaller absolute errors. That is the opposite of the positive association required by the planned Gate 0 rule. This is a failure of the intended uncertainty-gating criterion, not evidence that uncertainty gating works.

## Component diagnostics

| Uncertainty signal | Spearman rho | p-value | 95% bootstrap CI | Spans zero? | Worst-quartile AUROC |
|---|---:|---:|---|---|---:|
| Aggregate uncertainty | -0.3889 | 1.37e-03 | [-0.5985, -0.1352] | No | 0.2911 |
| Ensemble disagreement | 0.0004 | 9.98e-01 | [-0.2522, 0.2520] | Yes | 0.5858 |
| ESM-2 embedding OOD distance | -0.3474 | 4.58e-03 | [-0.5849, -0.0788] | No | 0.2770 |
| Conformal interval width | 0.0576 | 6.48e-01 | [-0.1875, 0.3076] | Yes | 0.4596 |

These component-level results are retrospective diagnostics on a small evaluation pool. They do not establish that any signal will reliably identify errors on a future library.

## Conformal interval interpretation

The previous run reported 100% empirical coverage on 65/65 held-out variants using a calibration quantile of 0.7825, giving a nominally symmetric interval width of approximately 1.565. Coverage alone does not establish useful calibration: interval width, calibration-set size, and the distribution shift between single-site calibration variants and multi-site evaluation variants must also be considered. The point should be reported as “100% observed coverage on this held-out sample”, not a general guarantee of future performance.

## Required downstream rule

Because the Gate 0 positive-association criterion was not met, downstream candidate selection must not use aggregate uncertainty as a gating or exclusion signal. Use the approved diversity-based fallback while retaining predictions, structural evidence, compute cost, and missingness as separately documented information.

The implementation and machine-readable policy must agree:
- Gate 0 passes only when `ci_lower > 0`.
- An interval spanning zero fails the rule.
- An entirely negative interval also fails the rule, but must be described as an inverse association rather than “including zero”.
- Missing or inconsistent gate artifacts fail closed to diversity selection.

## Limitations and non-claims

The evaluation pool contains 65 multi-site variants and is retrospective. These metrics do not prove prospective performance, successful wet-lab PET degradation improvement, or quantum advantage. The result demonstrates why the gate’s fallback must be enforced; it does not establish that the entire uncertainty subsystem is scientifically calibrated.
