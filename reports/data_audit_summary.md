# Q-Catalyst Raw Data Audit & Manifest Report

## 1. Executive Summary
- **Total Inventoried Files:** 3
- **File Formats:** ``: 1, `.csv`: 1, `.fasta`: 1
- **Audited CSV Datasets:** 1 (1 valid, 0 error/empty)
- **Total Measurement Records:** 307
- **Network Access:** None (100% local inspection, raw data strictly unmodified).

## 2. Dataset Inventory by Enzyme & Target Type

| Enzyme | Dataset Type | Target | Rows | Columns | Score Range | Anomalies / Flags |
|---|---|---|---|---|---|---|
| `pazy_collection` | `pazy` | `cross_enzyme_survey` | 307 | `protein_id;protein_name;verified_activity;organism_id;organism_name;phylum;substrates;EMBL-GenBank-DDBJ_CDS;EMBL-GenBank-DDBJ_CDS;MGnify;PDB;RefSeq_Protein;UniParc;UniProtKB;UniRef100;UniRef50;UniRef90;literature_dois;sequence_length` | N/A | `none` |

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