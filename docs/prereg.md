# Pre-Registration Plan: Evaluation Protocol & Hypotheses

**Commit Reference:** Pre-execution baseline freeze  
**Date:** October 2026  
**Primary Objective:** Determine whether uncertainty-guided computational chemistry improves PETase variant prioritisation enough to justify computational costs over AI baselines.

---

## 1. Pre-Registered Hypotheses

### Hypothesis 1 (Gate 0 — Uncertainty Informativeness)
- **Null Hypothesis ($H_{1,0}$):** The aggregate predictive uncertainty score $U(x)$ has no monotonic relationship with absolute prediction error on held-out test data ($\rho \le 0$).
- **Alternative Hypothesis ($H_{1,1}$):** Predictive uncertainty is positively correlated with prediction error ($\rho > 0$, and the 95% bootstrap confidence interval strictly excludes 0).
- **Decision Rule:** If $H_{1,0}$ cannot be rejected, Gate 0 FAILS. The candidate acquisition gate automatically abstains from uncertainty weighting and shifts to the pre-specified Stratified Diversity Fallback.

### Hypothesis 2 (Value of Chemistry — VoC)
- **Null Hypothesis ($H_{2,0}$):** Incorporating quantum/classical active-site cluster descriptors into AI ranking yields no improvement in top-10 precision over the supervised AI baseline ($P@10(\text{A4}) \le P@10(\text{A2})$).
- **Alternative Hypothesis ($H_{2,1}$):** Uncertainty-gated chemistry (Arm A4) achieves statistically significant Precision@10 improvement with positive Value of Chemistry:
  $$\text{VoC} = \frac{P@10(\text{A4}) - P@10(\text{A2})}{\text{Chemistry CPU Hours}} > 0$$

---

## 2. Benchmark Arms Specification

| Arm ID | Description | Selection Policy | Evidence Sources |
|---|---|---|---|
| **A0** | Random Floor | Uniform random selection from eligible pool | None (Null Baseline) |
| **A1** | Zero-Shot ESM-1v | Ranked by ESM pseudo-log-likelihood ratio | Sequence fitness alone |
| **A2** | Supervised AI Baseline | Supervised ensemble regressor + sequence/structure features | Sequence + structure features |
| **A3** | AI + Random Chemistry | Supervised AI + Random batch of 12 chemistry evaluations | AI + Unbiased chemistry |
| **A4** | AI + Uncertainty Gate | Acquisition score $a(x) = \lambda U + \eta P + \mu D - \nu C$ | AI + Uncertainty-gated chemistry |
| **A5** | AI + CASCI Reference | AI + Exact CASCI active-space validation | AI + Exact chemistry ground truth |
| **A6** | AI + Noiseless VQE | AI + Ideal statevector VQE simulation | AI + Quantum simulator |
| **A7** | AI + Noisy VQE | AI + Realistic depolarizing noise VQE simulation | AI + Noisy quantum hardware proxy |

---

## 3. Evaluation Protocol
- **Evaluation Split:** Test partition (15% held-out, identity-aware split to prevent data leakage).
- **Primary Metric:** Precision@10 (fraction of top 10 ranked candidates exceeding $1.5\times$ wild-type activity or $\Delta T_m \ge +2.0^\circ\text{C}$).
- **Complementary Metrics:** Precision@5, Precision@20, Spearman rank correlation ($\rho$), VoC efficiency.
- **Statistical Significance:** Paired non-parametric bootstrap resampling ($B = 1000$ iterations) with 95% percentile confidence intervals.
