# Q-Catalyst — Role 1 Deliverables Checklist & Traceability Matrix

**Role:** Role 1 (Protein ML Lead)  
**Scope:** Data curation, protein AI, uncertainty quantification, Gate 0 validation, and baseline benchmarking (Arms A0–A2).  
**Effective Date:** October 2026 (Updated for Real Model Weights & Leakage-Free Split Integration)

---

## 1. Role 1 Responsibilities & Deliverables Mapping

| # | Deliverable / Task | Implementation File | Test Suite | Output Artifact | Acceptance Criteria | Current Status |
|---|---|---|---|---|---|---|
| **D1** | **Raw Data Audit & Licensing** | `src/qcatalyst/data/audit.py`, `scripts/prepare_data.py` | `tests/test_data_audit.py` | `reports/data_audit_summary.md`, `reports/tables/petgym_file_manifest.csv` | Verify file existence, SHA-256 hashes, licenses, non-comparability of unpooled assay units | **Completed** |
| **D2** | **Mutation Parsing & Canonical Curation** | `src/qcatalyst/data/curation.py`, `mutation.py` | `tests/test_mutation.py`, `tests/test_data_contracts.py` | `data/curated/variants.parquet`, `reports/data_quality_report.md` | Standardize sequence representations; map parent mutations to mature 5XJH crystal numbering; flag duplicates/outliers | **Completed** |
| **D3** | **Pretrained ESM-1v Zero-Shot Scoring** | `src/qcatalyst/protein_ai/esm_scoring.py`, `scripts/run_ai.py` | `tests/test_protein_ai.py`, `scratch/step2_esm1v_smoke_test.py` | `results/predictions/esm1v_scores.parquet`, `results/predictions/real_esm1v/` | Official masked-marginal $\Delta\text{LLR}$ scoring using genuine `facebook/esm1v_t33_650M_UR90S_1` checkpoint; 1-based indexing; deterministic eval | **In Progress (Weights Approved)** |
| **D4** | **Pretrained ESM-2 Sequence Embeddings** | `src/qcatalyst/protein_ai/embeddings.py` | `tests/test_embeddings.py`, `tests/test_embeddings_real.py` | `results/embeddings/real_facebook_esm2_t12_35M_UR50D/`, `embedding_index.parquet` | Genuine 480-d sequence embeddings from `facebook/esm2_t12_35M_UR50D`; attention-masked residue-only mean pooling; SHA-256 cache | **Completed** |
| **D5** | **Supervised Predictors (XGBoost & Ridge)** | `src/qcatalyst/protein_ai/predictor.py` | `tests/test_predictor.py` | `results/models/`, `results/predictions/predictions.parquet`, `reports/supervised_prediction_report.md` | Train XGBoost & Ridge on real ESM-2 embeddings against genuine experimental labels; evaluate on leakage-free single-to-multi split ($N=65$) | **Partially Completed (Requires Retraining on Leakage-Free Split)** |
| **D6** | **Uncertainty Quantification & Conformalization** | `src/qcatalyst/uncertainty/estimate.py`, `calibrate.py` | `tests/test_uncertainty.py` | `results/uncertainty/uncertainty.parquet`, `reports/uncertainty_report.md` | 10-member bootstrap ensemble std, 90% split-conformal intervals, normalized ESM-2 OOD distance, 5XJH catalytic triad structural risk, composite $U(x)$ | **Partially Completed (Requires Evaluation on Leakage-Free Split)** |
| **D7** | **Gate 0 Uncertainty Validation** | `src/qcatalyst/uncertainty/gate.py` | `tests/test_gate0.py` | `reports/tables/gate0_verdict.json`, `reports/figures/gate0_coverage_plot.png`, `reports/gate0_validation_report.md` | Spearman correlation $\rho(U(x), \|y - \hat{y}\|) > 0$ with 95% bootstrap CI; empirical coverage $\ge 85\%$ on genuine held-out evaluation variants | **Partially Completed (Requires Evaluation on Leakage-Free Split)** |
| **D8** | **Benchmarking Suite (Arms A0–A2)** | `src/qcatalyst/evaluation/benchmarks.py` | `tests/test_benchmarks.py` | `results/benchmarks/benchmark_results.parquet`, `reports/benchmark_report.md` | Standardized metrics (P@5/10/20, EF@10, NDCG@10, Spearman $\rho$, Kendall $\tau$) with 1000 bootstrap CIs comparing A0 (Random), A1 (ESM-1v), A2 (XGBoost) | **Partially Completed (Requires Real ESM-1v & Leakage-Free Split)** |

---

## 2. Key Verification Rules & Guardrails
1. **Model Weight Separation:** ESM-2 35M embeddings (D4) and ESM-1v zero-shot mutation scores (D3) must remain strictly segregated in code and disk artifacts.
2. **Leakage-Free Partition:** Single-to-multi evaluation task must strictly segregate single-site training mutants ($N=84$) from multi-site evaluation mutants ($N=65$) with exactly zero ID or sequence overlap.
3. **No Synthetic Fallback Masking:** Offline unit tests use an explicit `mock=True` parameter with calibrated BLOSUM62 proxy; production runs with `mock=False` must use real model weights and never label synthetic values as genuine.
4. **Non-Claims Boundary:** All computational predictions and zero-shot scores must carry the authoritative project disclaimer: *"Computational model prioritisation based on retrospective data; not experimentally verified increase in PET plastic degradation."*

