# Q-Catalyst Project Artifact Source Inventory

- **Source Root Evaluated:** `/home/luckycelestial/Pavithran/QCatalyst/Q_Catalyst`
- **Total Candidate Sources Audited:** 20
- **Available & Usable Sources:** 20
- **Missing Sources:** 0

## 1. Discovered Source Artifacts Table

| Relative Path | Format | Size (Bytes) | Evidence Class | Usable? | Summary & Findings |
|:---|:---:|:---:|:---:|:---:|:---|
| `results/rankings/role1_candidate_gate.json` | JSON | 1,710 | `PREDICTED` | ✅ YES | Dictionary with 14 top-level keys: ['pipeline_component', 'parent_enzyme', 'target_name', 'split_version', 'n_train'] |
| `results/rankings/role1_candidate_gate.parquet` | Parquet | 55,628 | `PREDICTED` | ✅ YES | 65 rows, 65 columns |
| `results/rankings/ranking.parquet` | Parquet | 30,880 | `PREDICTED / REPORTED` | ✅ YES | 26 rows, 38 columns |
| `results/predictions/predictions.parquet` | Parquet | 26,351 | `PREDICTED` | ✅ YES | 298 rows, 19 columns |
| `results/predictions/esm1v_scores.parquet` | Parquet | 18,719 | `PREDICTED` | ✅ YES | 65 rows, 18 columns |
| `results/uncertainty/uncertainty.parquet` | Parquet | 25,026 | `PREDICTED` | ✅ YES | 149 rows, 23 columns |
| `results/uncertainty/gate0_report.json` | JSON | 306 | `COMPUTED` | ✅ YES | Dictionary with 8 top-level keys: ['gate0_passed', 'spearman_rho', 'p_value', 'ci_lower_95', 'ci_upper_95'] |
| `results/chemistry/chem_results.parquet` | Parquet | 35,631 | `COMPUTED` | ✅ YES | 31 rows, 48 columns |
| `results/chemistry/chem_jobs.json` | JSON | 4,799 | `REPORTED` | ✅ YES | List of 31 items |
| `results/chemistry/tier_s_cluster.xyz` | XYZ | 1,950 | `COMPUTED` | ✅ YES | Text document (1950 bytes) |
| `results/chemistry/fast_petase_directional_check.json` | JSON | 782 | `COMPUTED` | ✅ YES | Dictionary with 4 top-level keys: ['timestamp', 'ispetase_5xjh', 'fast_petase_7sh6', 'directional_shift'] |
| `results/quantum/casci_control_role3.json` | JSON | 1,399 | `COMPUTED` | ✅ YES | Dictionary with 5 top-level keys: ['description', 'active_space', 'basis', 'method', 'variants'] |
| `results/quantum/quantum_results.parquet` | Parquet | 10,383 | `COMPUTED` | ✅ YES | 20 rows, 15 columns |
| `data/curated/variants.parquet` | Parquet | 44,913 | `EXPERIMENTAL / REPORTED` | ✅ YES | 430 rows, 24 columns |
| `data/structures/5XJH.pdb` | PDB | 203,472 | `EXPERIMENTAL` | ✅ YES | PDB format with 2512 lines (2073 coordinate atoms) |
| `data/structures/7SH6.pdb` | PDB | 226,638 | `EXPERIMENTAL` | ✅ YES | PDB format with 2798 lines (2348 coordinate atoms) |
| `data/structures/variants` | Directory (PDBs) | 4,477,340 | `COMPUTED` | ✅ YES | Directory containing 26 PDB structure files |
| `reports/gate0_validation_report.md` | Markdown | 2,941 | `REPORTED` | ✅ YES | Text document (2941 bytes) |
| `reports/benchmark_report.md` | Markdown | 4,087 | `REPORTED` | ✅ YES | Text document (4087 bytes) |
| `reports/final_report.md` | Markdown | 20,618 | `REPORTED` | ✅ YES | Text document (20618 bytes) |

## 2. In-Depth Artifact Analysis & Schema Observations

### `results/rankings/role1_candidate_gate.json`
- **Description:** Role 1 Candidate Acquisition Gate Summary
- **Evidence Classification:** `PREDICTED`
- **SHA-256:** `7cd4b6d2012e182d9b525cb677920bcb085efa47e140fdfce74db5179f2f49e6`
- **Summary:** Dictionary with 14 top-level keys: ['pipeline_component', 'parent_enzyme', 'target_name', 'split_version', 'n_train']

### `results/rankings/role1_candidate_gate.parquet`
- **Description:** Held-out candidates with gate annotation & uncertainty
- **Evidence Classification:** `PREDICTED`
- **SHA-256:** `461c63b380372ef77d58a324d35698bb0be7946abc40a70c064871012cac1c83`
- **Summary:** 65 rows, 65 columns

### `results/rankings/ranking.parquet`
- **Description:** Final 26 prioritized candidates with confidence tiers
- **Evidence Classification:** `PREDICTED / REPORTED`
- **SHA-256:** `67af3d2ca5669e1323e7f79d07b10cbd1d93815fec00048d6233fc3962ece131`
- **Summary:** 26 rows, 38 columns

### `results/predictions/predictions.parquet`
- **Description:** Supervised XGBoost & Ridge predictions on Split v2
- **Evidence Classification:** `PREDICTED`
- **SHA-256:** `6a388544ae39849dd1735b0a8f37405941b385c18633ae378f3f9addcc574c2b`
- **Summary:** 298 rows, 19 columns

### `results/predictions/esm1v_scores.parquet`
- **Description:** Real ESM-1v 650M masked-marginal zero-shot LLR scores
- **Evidence Classification:** `PREDICTED`
- **SHA-256:** `3cc7d0742ec5865bd732039af26072ab62e21e047be478d6daeec86013b790d0`
- **Summary:** 65 rows, 18 columns

### `results/uncertainty/uncertainty.parquet`
- **Description:** Multi-signal uncertainty and split-conformal intervals
- **Evidence Classification:** `PREDICTED`
- **SHA-256:** `8e62de4ad8bab925432199c62f8b41f06dc8ae6ad1eca16989ea350e34b46007`
- **Summary:** 149 rows, 23 columns

### `results/uncertainty/gate0_report.json`
- **Description:** Gate 0 bootstrap validation verdict and statistical breakdown
- **Evidence Classification:** `COMPUTED`
- **SHA-256:** `d5ea189f3c163cdcfd4e0e8e325607a5bbc81d7ef3d1aadf38908fb998f98533`
- **Summary:** Dictionary with 8 top-level keys: ['gate0_passed', 'spearman_rho', 'p_value', 'ci_lower_95', 'ci_upper_95']

### `results/chemistry/chem_results.parquet`
- **Description:** 31 Tier S cluster QM/MM DFT/RHF calculations & descriptors
- **Evidence Classification:** `COMPUTED`
- **SHA-256:** `1d4d1c705f3d3c32caaf5185f88a43055a954c77a91382f57b0a000506bd2636`
- **Summary:** 31 rows, 48 columns

### `results/chemistry/chem_jobs.json`
- **Description:** List of 31 budgeted chemistry variant targets
- **Evidence Classification:** `REPORTED`
- **SHA-256:** `9072f0572a17532da5b471fbcd80143d79f3830cef0b911a31601a5050b7a345`
- **Summary:** List of 31 items

### `results/chemistry/tier_s_cluster.xyz`
- **Description:** 42-atom catalytic cluster Cartesian geometry
- **Evidence Classification:** `COMPUTED`
- **SHA-256:** `b9a85fedc2215e541583fedb623ae147c77f9e4f3ffd6e3b1011d216f2478db4`
- **Summary:** Text document (1950 bytes)

### `results/chemistry/fast_petase_directional_check.json`
- **Description:** 5XJH vs 7SH6 catalytic triad directional comparison
- **Evidence Classification:** `COMPUTED`
- **SHA-256:** `34f9dbcf1697e86c43ac6cca3ff4f56f2542b08de9136bf2dd4f9e0f4615fa44`
- **Summary:** Dictionary with 4 top-level keys: ['timestamp', 'ispetase_5xjh', 'fast_petase_7sh6', 'directional_shift']

### `results/quantum/casci_control_role3.json`
- **Description:** Exact classical CASCI(4e, 4o) ground truth for Arm A5
- **Evidence Classification:** `COMPUTED`
- **SHA-256:** `4850194407ae89c6c0707c3d998b934270c6292e1c2f27ab94dc16c21df4363b`
- **Summary:** Dictionary with 5 top-level keys: ['description', 'active_space', 'basis', 'method', 'variants']

### `results/quantum/quantum_results.parquet`
- **Description:** Simulated VQE vs CASCI parity, noise sweeps & resources
- **Evidence Classification:** `COMPUTED`
- **SHA-256:** `216e8c5c2091d9c03687f3016789b6ec0304a6f1487a61972a6bec0216f5c8d2`
- **Summary:** 20 rows, 15 columns

### `data/curated/variants.parquet`
- **Description:** 430 curated PETase variants aligned to 5XJH
- **Evidence Classification:** `EXPERIMENTAL / REPORTED`
- **SHA-256:** `3045fb834742235d91cf2095598204c991ba69be7023f67c157a0672c7358cee`
- **Summary:** 430 rows, 24 columns

### `data/structures/5XJH.pdb`
- **Description:** High-resolution (1.5Å) crystal structure of native IsPETase
- **Evidence Classification:** `EXPERIMENTAL`
- **SHA-256:** `a0410e1407aa9ccf5fa16396a3fe329c46504fc4d6182fb438607ec23c697680`
- **Summary:** PDB format with 2512 lines (2073 coordinate atoms)

### `data/structures/7SH6.pdb`
- **Description:** Crystal structure of engineered FAST-PETase
- **Evidence Classification:** `EXPERIMENTAL`
- **SHA-256:** `629690d5c50a83e9f28549c1f15ffca5c3c6c261b5e2168ba471622f1d6c702f`
- **Summary:** PDB format with 2798 lines (2348 coordinate atoms)

### `data/structures/variants`
- **Description:** 26 mutant structural models for prioritized candidates
- **Evidence Classification:** `COMPUTED`
- **SHA-256:** `26 files`
- **Summary:** Directory containing 26 PDB structure files

### `reports/gate0_validation_report.md`
- **Description:** Detailed narrative of Gate 0 uncertainty failure and fallback
- **Evidence Classification:** `REPORTED`
- **SHA-256:** `9d2fe2d5bfa4b2982e080c4042f309e8740b373520ce28a81407d55ddf1f7c8f`
- **Summary:** Text document (2941 bytes)

### `reports/benchmark_report.md`
- **Description:** Prioritization performance across Arms A0, A1, A2
- **Evidence Classification:** `REPORTED`
- **SHA-256:** `9ea4cb814fa4b86f1fc2627f4c167fcd6340381699f3d4ff05b4e3df6577fbf9`
- **Summary:** Text document (4087 bytes)

### `reports/final_report.md`
- **Description:** Role 2 Computational Chemistry Lead final report
- **Evidence Classification:** `REPORTED`
- **SHA-256:** `c431a9aa7d393841ffc5b111251a066c8475e57d78d3d5ac8bb75bd72183a70d`
- **Summary:** Text document (20618 bytes)

## 3. Critical Scientific Inferences and Boundaries

1. **Sequence vs Structural Numbering:** In `variants.parquet`, mutations use mature protein indexing (residues 37–290 for IsPETase), corresponding to 5XJH crystallographic numbering. Residue numbers must be checked explicitly against sequence indices.
2. **Chemistry Coverage:** Only 31 variants (1 WT + 15 Arm A4 + 15 Arm A3) have PySCF DFT cluster calculations. Other variants in `ranking.parquet` lack QM data and must remain marked `NOT_AVAILABLE`.
3. **Quantum Coverage:** Only 5 candidate variants plus WT possess exact CASCI(4e, 4o) keys. VQE agreement benchmarks simulation accuracy, not experimental PET turnover.
4. **Gate 0 Uncertainty Validation:** Evaluated on held-out multi-site mutants and failed honestly ($ho = -0.3889 \le 0$). Diversity fallback is active.
