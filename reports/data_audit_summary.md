# Q-Catalyst Raw Data Audit & Manifest Report

## 1. Executive Summary
- **Total Inventoried Files:** 51
- **File Formats:** `.csv`: 14, `.fasta`: 14, `.pdb`: 13, ``: 3, `.py`: 3, `.md`: 2, `.toml`: 1, `.lock`: 1
- **Audited CSV Datasets:** 14 (14 valid, 0 error/empty)
- **Total Measurement Records:** 1113
- **Network Access:** None (100% local inspection, raw data strictly unmodified).

## 2. Dataset Inventory by Enzyme & Target Type

| Enzyme | Dataset Type | Target | Rows | Columns | Score Range | Anomalies / Flags |
|---|---|---|---|---|---|---|
| `pazy_collection` | `pazy` | `cross_enzyme_survey` | 307 | `protein_id;protein_name;verified_activity;organism_id;organism_name;phylum;substrates;EMBL-GenBank-DDBJ_CDS;EMBL-GenBank-DDBJ_CDS;MGnify;PDB;RefSeq_Protein;UniParc;UniProtKB;UniRef100;UniRef50;UniRef90;literature_dois;sequence_length` | [nan, nan] | `none` |
| `BhrPETase` | `low-to-high-mutant` | `Act` | 95 | `mutant;score` | [0.06, 33.11] | `1_duplicate_mutants` |
| `BhrPETase` | `low-to-high-mutant` | `Tm` | 95 | `mutant;score` | [71.5, 842.0] | `1_duplicate_mutants,unphysical_high_tm(max=842.0)` |
| `IsPETase` | `low-to-high-mutant` | `Act` | 149 | `mutant;score` | [0.0, 1.35] | `none` |
| `IsPETase` | `low-to-high-mutant` | `Tm` | 149 | `mutant;score` | [37.5, 77.0] | `none` |
| `MipaPETase` | `low-to-high-mutant` | `Tm` | 28 | `mutant;score` | [21.0, 92.4] | `none` |
| `BhrPETase` | `single-site-mutant` | `Act` | 33 | `mutant;score` | [0.06, 17.56] | `none` |
| `BhrPETase` | `single-site-mutant` | `Tm` | 33 | `mutant;score` | [79.5, 97.5] | `none` |
| `CaPETase` | `single-site-mutant` | `Act` | 15 | `mutant;score` | [0.15, 1.0] | `none` |
| `CaPETase` | `single-site-mutant` | `Tm` | 15 | `mutant;score` | [60.7, 70.1] | `none` |
| `IsPETase` | `single-site-mutant` | `Act` | 84 | `mutant;score` | [0.0, 1.2] | `none` |
| `IsPETase` | `single-site-mutant` | `Tm` | 84 | `mutant;score` | [37.5, 55.0] | `none` |
| `KubuPETase` | `single-site-mutant` | `Tm` | 12 | `mutant;score` | [79.2, 90.4] | `none` |
| `MipaPETase` | `single-site-mutant` | `Tm` | 14 | `mutant;score` | [58.5, 75.1] | `none` |

## 3. Key Findings & Data Quality Flags
1. **Zero FASTA Mismatches:** All PET-Gym mutations align 100% with the parent sequence without out-of-bounds indices.
2. **Duplicates Detected:**
   - `low-to-high-mutant/mutant/BhrPETase_Act`: 1 duplicate row (`H218S:F222I:H191D`, score 6.19).
   - `low-to-high-mutant/mutant/BhrPETase_Tm`: 1 duplicate row (`H218S:F222I:H191D`, score 88.0 °C).
3. **Physical Range Outlier:**
   - `low-to-high-mutant/mutant/BhrPETase_Tm`: Row 62 (`H218S:F222I:F243T`) has `score = 842.0` °C.
     - Flagged as `unphysical_high_tm(max=842.0)`. Likely a decimal omission for 84.2 °C.
4. **Target Separation:** Activity (`Act`) and melting temperature (`Tm`) labels differ in measurement types and scales and must remain strictly separate modeling targets.

## 4. Manifest Integrity
- All inventoried files have SHA256 checksums recorded in `reports/tables/petgym_file_manifest.csv`.
- Re-running this audit does not modify or corrupt any raw data.