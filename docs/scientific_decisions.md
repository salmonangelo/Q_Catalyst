# Q-Catalyst Scientific Decision Document: Variant Prioritisation & Evaluation Standards

**Author:** Role 1 (Protein ML Lead)  
**Status:** Approved & Pre-registered  
**Target Enzymes:** PET Hydrolases (*Ideonella sakaiensis* PETase / IsPETase, BhrPETase)  
**Reference PDB:** 5XJH (mature sequence residues 37–290)

---

## 1. Definition of a "Good Variant" (Hit Definition)

In computational protein engineering and laboratory enzyme screening, candidate variants must be evaluated against a reproducible, experimentally grounded standard of improvement.

### 1.1 Primary Activity Criterion: Relative Fold Improvement Over Wild-Type
For catalytic degradation of PET plastics (`activity_raw`), activity values are measured relative to the wild-type parent enzyme under identical assay conditions:
$$\text{Relative Activity} = \frac{\text{Activity}(\text{Mutant})}{\text{Activity}(\text{Wild-Type})}$$

* **Binary Hit Standard:**
  $$\text{is\_good\_variant} \iff y_{\text{true}} > 1.0$$
  A variant is classified as a genuine "good variant" if and only if its experimental degradation activity strictly exceeds the wild-type enzyme under identical pH, temperature, and substrate conditions.

* **Held-Out Library Base Rate (Prevalence):**
  * **Held-Out Evaluation Pool:** 149 combinatorial multi-site mutants (`low-to-high-mutant` split from PET-Gym).
  * **Hit Count ($y_{\text{true}} > 1.0$):** **18 variants**
  * **Prevalence / Base Rate:**
    $$\text{Base Rate} = \frac{18}{149} \approx 12.08\%$$
  This 12.08% prevalence represents the true screening hit rate against which random selection (A0) and model-guided prioritisation (A1, A2) must be compared.

### 1.2 Secondary Thermostability Criterion
For melting temperature / thermostability ($T_m$ / $\Delta T_m$):
$$\Delta T_m = T_m(\text{Mutant}) - T_m(\text{Wild-Type})$$
* **Hit Standard:** $\Delta T_m > 0.0^\circ\text{C}$ (enhanced thermal resistance without losing catalytic fold).

---

## 2. Evaluation Metrics Formalization

To avoid misleading reporting, all evaluation arms (A0–A2) are assessed on identical held-out test splits using standardized metrics with non-parametric percentile bootstrap confidence intervals ($B = 1000, 95\% \text{ CI}$):

### 2.1 Precision@K ($K \in \{5, 10, 20\}$)
Fraction of the top-$K$ algorithmically ranked variants that satisfy the hit standard:
$$\text{Precision@K} = \frac{1}{K} \sum_{i=1}^K \mathbb{I}(y_{\text{true}}[\pi(i)] > 1.0)$$
where $\pi(i)$ is the index of the $i$-th highest scoring candidate.

### 2.2 Enrichment Factor at K (EF@K)
Fold-enrichment of hits in the top-$K$ selection compared to random screening:
$$\text{EF@K} = \frac{\text{Precision@K}}{\text{Base Rate}} = \frac{\text{Precision@K}}{18 / 149}$$

### 2.3 Normalized Discounted Cumulative Gain (NDCG@K)
Measures ranking quality with position-discounted rewards:
$$\text{DCG@K} = \sum_{i=1}^K \frac{\max(0, y_{\text{true}}[\pi(i)])}{\log_2(i + 1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
where $\text{IDCG@K}$ is the maximum possible DCG achieved by sorting strictly by true experimental activity.

### 2.4 Rank Correlations (Spearman $\rho$ and Kendall $\tau$)
* **Spearman $\rho$:** Monotonic rank correlation between algorithm scores and experimental labels across the entire held-out variant set ($N=149$).
* **Kendall $\tau$:** Pairwise concordance rank correlation measuring the fraction of correctly ordered pairs.

---

## 3. Benchmarking Arms Scope & Policy

| Arm ID | Arm Description | Source / Inputs | Status |
|---|---|---|---|
| **A0** | Random ranking baseline floor | Uniform random permutation (averaged across seeds) | Evaluated |
| **A1** | ESM-1v zero-shot effect ranking | Zero-shot $\Delta\text{LLR}$ score from ESM-1v | Evaluated |
| **A2** | Supervised XGBoost + Structural Features | Supervised model on ESM-2 representations + catalytic triad proximity | Evaluated |
| **A3** | AI + Random Classical Chemistry | Role 2 PySCF cluster calculations | Pending Role 2 |
| **A4** | AI + Uncertainty-Gated Chemistry | Role 1 Gate + Role 2 PySCF | Pending Role 2/4 |
| **A5** | AI + Exact CASCI Quantum Control | Role 3 classical active-space exact solver | Pending Role 3 |
| **A6** | AI + Noiseless VQE | Role 3 simulator VQE | Pending Role 3 |
| **A7** | AI + Noisy VQE | Role 3 noisy simulator VQE | Pending Role 3 |

> [!IMPORTANT]
> **Scientific Integrity Rule:** Arms A3–A7 are marked as pending. Results for downstream chemistry and quantum arms will **not** be invented or synthesized by Role 1; they will be evaluated only when real outputs from Roles 2 and 3 are delivered.
