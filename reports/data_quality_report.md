# Q-Catalyst Data Quality & Curation Report

## 1. Curation Overview
- **Total Processed Rows:** 430
- **Unique Biological Variant IDs:** 299
- **Single-Site Mutants:** 289
- **Multi-Site Mutants:** 141
- **Variants with Activity:** 376
- **Variants with Tm:** 430
- **Variants with Both Act & Tm:** 376

## 2. Records by Parent Enzyme
| Parent Enzyme | Count | Has Activity | Has Tm | Split Distribution |
|---|---|---|---|---|
| `BhrPETase` | 128 | True | True | low-to-high-mutant: 95, single-site-mutant: 33 |
| `CaPETase` | 15 | True | True | single-site-mutant: 15 |
| `IsPETase` | 233 | True | True | low-to-high-mutant: 149, single-site-mutant: 84 |
| `KubuPETase` | 12 | False | True | single-site-mutant: 12 |
| `MipaPETase` | 42 | False | True | low-to-high-mutant: 28, single-site-mutant: 14 |

## 3. Structure Mapping to Reference 5XJH
- **Reference Structure:** [5XJH.pdb](file:///c:/Users/Asus/Desktop/QHacks/qcatalyst/data/structures/5XJH.pdb) (IsPETase mature domain, residues 37..290).
- **Numbering Rule:** Parent residue numbering is preserved in `parent_mutation_positions`. Mapped positions on 5XJH are kept strictly separate in `structure_5xjh_positions`.
- **Mapping Breakdown:**
  - `exact_1to1`: 233 variants
  - `aligned`: 190 variants
  - `partial_aligned`: 5 variants
  - `unmapped`: 2 variants

## 4. Raw Data Anomalies & Unresolved Data Issues
1. **Duplicate Records (1 detected):**
   - In `low-to-high-mutant/BhrPETase`, mutant `H218S:F222I:H191D` occurs twice with identical scores.
   - Handled by preserving both rows with unique `row_id`, adding `#dup2` to `variant_id`, and setting `is_duplicate=True`.
2. **Physical Outliers (1 detected):**
   - In `low-to-high-mutant/BhrPETase_Tm`, mutant `H218S:F222I:F243T` has raw $T_m = 842.0^\circ\text{C}$.
   - Flagged with `is_outlier=True` and `unphysical_high_tm(842.0)`. Preserved as raw value for transparency without corrupting downstream models.
3. **Incomparable Activity Labels:**
   - Raw activity measurements across `IsPETase`, `BhrPETase`, and `CaPETase` originate from different assays and units.
   - `activity_unit` explicitly identifies each source (`relative_fold_change_ispetase`, `activity_bhr_unpooled`, `activity_ca_unpooled`) and prevents uncalibrated pooling.

## 5. Output Parquet Schema
| Field | Type | Description |
|---|---|---|
| `row_id` | `string` | Unique dataset row identifier (`{split}_{enzyme}_{index}`) |
| `variant_id` | `string` | Stable variant identifier (`{enzyme}_{canonical_mut}`) |
| `parent_enzyme` | `string` | Parent wild-type enzyme name |
| `dataset_split` | `string` | `single-site-mutant` vs `low-to-high-mutant` |
| `study_id` | `string` | Source study identifier (`PET-Gym`) |
| `source_files` | `string` | Relative paths to source CSV files |
| `source_row_index` | `int64` | Zero-based row index in source CSV |
| `mutant_raw` | `string` | Raw unnormalized mutation string from source CSV |
| `mutations_canonical` | `string` | Canonical position-sorted colon-separated substitutions |
| `num_mutations` | `int64` | Mutation count (1 for single, >= 2 for multi) |
| `is_multi_mutant` | `bool` | True if multi-site substitution |
| `wildtype_sequence` | `string` | Parent full-length FASTA sequence |
| `mutant_sequence` | `string` | Generated variant sequence with substitutions applied |
| `sequence_length` | `int64` | Amino acid length of variant sequence |
| `parent_mutation_positions` | `string` | Colon-separated 1-based positions on parent sequence |
| `structure_5xjh_positions` | `string` | Colon-separated 1-based mapped positions on 5XJH structure |
| `structure_mapping_status` | `string` | Mapping status (`exact_1to1`, `aligned`, `partial_aligned`) |
| `activity_raw` | `float64` | Preserved raw activity score (or NaN) |
| `activity_unit` | `string` | Unpooled measurement unit or `none` |
| `tm_raw` | `float64` | Preserved raw melting temperature (or NaN) |
| `tm_unit` | `string` | `celsius` or `none` |
| `is_duplicate` | `bool` | True if duplicated in source dataset |
| `is_outlier` | `bool` | True if unphysical value detected |
| `data_quality_flags` | `string` | Comma-separated quality flags (`clean`, `duplicate`, etc.) |