# Q-Catalyst Role 1 Benchmarking Report

## Executive Summary
- **Parent Enzyme:** `IsPETase`
- **Target Target:** `activity`
- **Held-Out Evaluation Split:** `low-to-high-mutant` ($N=149$ variants)
- **Good Variant Standard:** $y_{\text{true}} > 1.0$ (pre-registered in `docs/scientific_decisions.md`)
- **Library Hit Base Rate:** **12.08%** (18/149 variants)

## 1. Comparative Prioritisation Performance (Arms A0–A2)
| Arm ID | Strategy Name | Precision@5 | Precision@10 | Precision@20 | EF@10 | NDCG@10 | Spearman $\rho$ | Kendall $\tau$ |
|---|---|---|---|---|---|---|---|---|
| **A0** | Random Ranking Baseline Floor | **0.40** | **0.20** | **0.20** | **1.66x** | 0.446 | -0.094 | -0.067 |
| **A1** | ESM-1v Zero-Shot Ranking | **0.60** | **0.30** | **0.15** | **2.48x** | 0.686 | 0.074 | 0.035 |
| **A2** | Supervised XGBoost + Structural Features | **0.60** | **0.40** | **0.20** | **3.31x** | 0.694 | 0.362 | 0.273 |

## 2. Metric Details with 95% Bootstrap Confidence Intervals ($B=1000$)
| Arm | Metric | Point Estimate | 95% CI Lower | 95% CI Upper | Status |
|---|---|---|---|---|---|
| `A0` | `precision@5` | **0.4000** | 0.0000 | 0.6050 | evaluated |
| `A0` | `precision@10` | **0.2000** | 0.0000 | 0.5000 | evaluated |
| `A0` | `precision@20` | **0.2000** | 0.0500 | 0.3500 | evaluated |
| `A0` | `enrichment_factor@5` | **3.3111** | 0.0000 | 5.9600 | evaluated |
| `A0` | `enrichment_factor@10` | **1.6556** | 0.0000 | 4.3824 | evaluated |
| `A0` | `enrichment_factor@20` | **1.6556** | 0.3386 | 2.7938 | evaluated |
| `A0` | `ndcg@5` | **0.4682** | 0.1393 | 0.7070 | evaluated |
| `A0` | `ndcg@10` | **0.4463** | 0.2408 | 0.6590 | evaluated |
| `A0` | `ndcg@20` | **0.4281** | 0.2732 | 0.5810 | evaluated |
| `A0` | `spearman_rho` | **-0.0943** | -0.2672 | 0.0709 | evaluated |
| `A0` | `kendall_tau` | **-0.0669** | -0.1848 | 0.0469 | evaluated |
| `A1` | `precision@5` | **0.6000** | 0.0000 | 1.0000 | evaluated |
| `A1` | `precision@10` | **0.3000** | 0.0000 | 0.7000 | evaluated |
| `A1` | `precision@20` | **0.1500** | 0.0500 | 0.3500 | evaluated |
| `A1` | `enrichment_factor@5` | **4.9667** | 0.0000 | 8.9400 | evaluated |
| `A1` | `enrichment_factor@10` | **2.4833** | 0.0000 | 5.2604 | evaluated |
| `A1` | `enrichment_factor@20` | **1.2417** | 0.4133 | 2.9822 | evaluated |
| `A1` | `ndcg@5` | **0.6887** | 0.4133 | 0.9240 | evaluated |
| `A1` | `ndcg@10` | **0.6860** | 0.3984 | 0.8111 | evaluated |
| `A1` | `ndcg@20` | **0.5338** | 0.4175 | 0.7150 | evaluated |
| `A1` | `spearman_rho` | **0.0737** | -0.0913 | 0.2402 | evaluated |
| `A1` | `kendall_tau` | **0.0345** | -0.0860 | 0.1566 | evaluated |
| `A2` | `precision@5` | **0.6000** | 0.0000 | 1.0000 | evaluated |
| `A2` | `precision@10` | **0.4000** | 0.1000 | 0.7000 | evaluated |
| `A2` | `precision@20` | **0.2000** | 0.0500 | 0.4012 | evaluated |
| `A2` | `enrichment_factor@5` | **4.9667** | 0.0000 | 8.7647 | evaluated |
| `A2` | `enrichment_factor@10` | **3.3111** | 0.7087 | 5.7944 | evaluated |
| `A2` | `enrichment_factor@20` | **1.6556** | 0.4382 | 3.3525 | evaluated |
| `A2` | `ndcg@5` | **0.6932** | 0.2324 | 0.9385 | evaluated |
| `A2` | `ndcg@10` | **0.6941** | 0.3783 | 0.8664 | evaluated |
| `A2` | `ndcg@20` | **0.6570** | 0.4398 | 0.8073 | evaluated |
| `A2` | `spearman_rho` | **0.3619** | 0.1986 | 0.5084 | evaluated |
| `A2` | `kendall_tau` | **0.2731** | 0.1568 | 0.3794 | evaluated |

## 3. Downstream Arm Status (A3–A7 Policy)
> [!NOTE]
> Per the team's scientific charter and non-invention policy, arms A3–A7 (classical chemistry and VQE active-space quantum calculations) are marked as `pending_role2_role3_results`.
> Role 1 will evaluate these arms using this exact metric harness as soon as real results from Roles 2 and 3 are deposited into `results/chemistry/chem_results.parquet` and `results/quantum/quantum_results.parquet`.

## 4. Integrity and Non-Claims Statement
> [!IMPORTANT]
> All metrics are computed on retrospective held-out multi-mutant data from PET-Gym. No computational score constitutes wet-lab experimental proof of degradation increase.