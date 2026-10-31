# Q-Catalyst Role 1 Benchmarking Report

## Executive Summary
- **Parent Enzyme:** `IsPETase`
- **Target Target:** `activity`
- **Held-Out Evaluation Split:** `low-to-high-mutant` ($N=65$ variants)
- **Good Variant Standard:** $y_{\text{true}} > 1.0$ (pre-registered in `docs/scientific_decisions.md`)
- **Library Hit Base Rate:** **10.77%** (7/65 variants)

## 1. Comparative Prioritisation Performance (Arms A0–A2)
| Arm ID | Strategy Name | Precision@5 | Precision@10 | Precision@20 | EF@10 | NDCG@10 | Spearman $\rho$ | Kendall $\tau$ |
|---|---|---|---|---|---|---|---|---|
| **A0** | Random Ranking Baseline Floor | **0.00** | **0.00** | **0.00** | **0.00x** | 0.405 | -0.005 | -0.005 |
| **A1** | ESM-1v Zero-Shot Ranking | **0.00** | **0.00** | **0.00** | **0.00x** | 0.324 | -0.387 | -0.261 |
| **A2** | Supervised XGBoost + Structural Features | **0.40** | **0.20** | **0.15** | **1.86x** | 0.406 | 0.107 | 0.061 |

## 2. Metric Details with 95% Bootstrap Confidence Intervals ($B=1000$)
| Arm | Metric | Point Estimate | 95% CI Lower | 95% CI Upper | Status |
|---|---|---|---|---|---|
| `A0` | `precision@5` | **0.0000** | 0.0000 | 0.0000 | evaluated |
| `A0` | `precision@10` | **0.0000** | 0.0000 | 0.0000 | evaluated |
| `A0` | `precision@20` | **0.0000** | 0.0000 | 0.0500 | evaluated |
| `A0` | `enrichment_factor@5` | **0.0000** | 0.0000 | 0.0000 | evaluated |
| `A0` | `enrichment_factor@10` | **0.0000** | 0.0000 | 0.0000 | evaluated |
| `A0` | `enrichment_factor@20` | **0.0000** | 0.0000 | 0.5429 | evaluated |
| `A0` | `ndcg@5` | **0.3805** | 0.1078 | 0.5923 | evaluated |
| `A0` | `ndcg@10` | **0.4045** | 0.1968 | 0.5990 | evaluated |
| `A0` | `ndcg@20` | **0.4121** | 0.2460 | 0.5688 | evaluated |
| `A0` | `spearman_rho` | **-0.0051** | -0.2383 | 0.2300 | evaluated |
| `A0` | `kendall_tau` | **-0.0048** | -0.1598 | 0.1573 | evaluated |
| `A1` | `precision@5` | **0.0000** | 0.0000 | 0.0000 | evaluated |
| `A1` | `precision@10` | **0.0000** | 0.0000 | 0.0000 | evaluated |
| `A1` | `precision@20` | **0.0000** | 0.0000 | 0.0000 | evaluated |
| `A1` | `enrichment_factor@5` | **0.0000** | 0.0000 | 0.0000 | evaluated |
| `A1` | `enrichment_factor@10` | **0.0000** | 0.0000 | 0.0000 | evaluated |
| `A1` | `enrichment_factor@20` | **0.0000** | 0.0000 | 0.0000 | evaluated |
| `A1` | `ndcg@5` | **0.4334** | 0.1775 | 0.5555 | evaluated |
| `A1` | `ndcg@10` | **0.3236** | 0.1653 | 0.4865 | evaluated |
| `A1` | `ndcg@20` | **0.3233** | 0.1961 | 0.4354 | evaluated |
| `A1` | `spearman_rho` | **-0.3867** | -0.5803 | -0.1693 | evaluated |
| `A1` | `kendall_tau` | **-0.2605** | -0.4070 | -0.1085 | evaluated |
| `A2` | `precision@5` | **0.4000** | 0.0000 | 0.8000 | evaluated |
| `A2` | `precision@10` | **0.2000** | 0.0000 | 0.5000 | evaluated |
| `A2` | `precision@20` | **0.1500** | 0.0500 | 0.3500 | evaluated |
| `A2` | `enrichment_factor@5` | **3.7143** | 0.0000 | 8.6667 | evaluated |
| `A2` | `enrichment_factor@10` | **1.8571** | 0.0000 | 4.8750 | evaluated |
| `A2` | `enrichment_factor@20` | **1.3929** | 0.3611 | 2.8539 | evaluated |
| `A2` | `ndcg@5` | **0.4781** | 0.1743 | 0.7256 | evaluated |
| `A2` | `ndcg@10` | **0.4062** | 0.1874 | 0.6609 | evaluated |
| `A2` | `ndcg@20` | **0.4124** | 0.2544 | 0.6060 | evaluated |
| `A2` | `spearman_rho` | **0.1067** | -0.1573 | 0.3641 | evaluated |
| `A2` | `kendall_tau` | **0.0608** | -0.1265 | 0.2459 | evaluated |

## 3. Downstream Arm Status (A3–A7 Policy)
> [!NOTE]
> Per the team's scientific charter and non-invention policy, arms A3–A7 (classical chemistry and VQE active-space quantum calculations) are marked as `pending_role2_role3_results`.
> Role 1 will evaluate these arms using this exact metric harness as soon as real results from Roles 2 and 3 are deposited into `results/chemistry/chem_results.parquet` and `results/quantum/quantum_results.parquet`.

## 4. Integrity and Non-Claims Statement
> [!IMPORTANT]
> All metrics are computed on retrospective held-out multi-mutant data from PET-Gym. No computational score constitutes wet-lab experimental proof of degradation increase.