"""Dataset curation module: converts raw PET-Gym records to canonical variants.parquet.

Standardizes mutations, computes full mutant sequences, verifies sequence bounds,
maps structure numbering to 5XJH with strict separation from parent numbering,
preserves original source measurements and missingness, and flags raw data anomalies.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from qcatalyst.data.mutation import (
    apply_mutations,
    canonicalize_mutations,
    map_mutation_positions_to_5xjh,
    parse_mutations,
    validate_mutations,
)


def load_parent_fastas(petgym_dir: Path) -> dict[str, str]:
    """Load wild-type FASTA sequence for each enzyme from PET-Gym files."""
    fastas: dict[str, str] = {}
    known_enzymes = ["IsPETase", "BhrPETase", "CaPETase", "KubuPETase", "MipaPETase"]

    for fasta_path in petgym_dir.glob("**/*.fasta"):
        stem = fasta_path.stem
        for enzyme in known_enzymes:
            if enzyme.lower() in stem.lower() and enzyme not in fastas:
                with open(fasta_path, "r", encoding="utf-8") as f:
                    seq = "".join(line.strip() for line in f if not line.startswith(">"))
                fastas[enzyme] = seq
                break

    if len(fastas) < len(known_enzymes):
        missing = set(known_enzymes) - set(fastas.keys())
        raise FileNotFoundError(f"Missing parent FASTA sequences for: {missing}")

    return fastas


def curate_petgym_dataset(petgym_dir: Path) -> pd.DataFrame:
    """Read all PET-Gym CSVs and assemble canonical curated variants DataFrame.

    Handles both single-site and low-to-high mutant sets, preserves original
    measurements, flags duplicates and outliers, and produces complete mutant sequences.
    """
    parent_fastas = load_parent_fastas(petgym_dir)
    ispetase_seq = parent_fastas["IsPETase"]

    # Activity unit mapping to maintain non-pooled separation
    ACTIVITY_UNITS = {
        "IsPETase": "relative_fold_change_ispetase",
        "BhrPETase": "activity_bhr_unpooled",
        "CaPETase": "activity_ca_unpooled",
    }

    # Datasets to process: (split_dir, enzyme, has_act, has_tm)
    dataset_configs = [
        # Single-site mutants
        ("single-site-mutant", "IsPETase", True, True),
        ("single-site-mutant", "BhrPETase", True, True),
        ("single-site-mutant", "CaPETase", True, True),
        ("single-site-mutant", "KubuPETase", False, True),
        ("single-site-mutant", "MipaPETase", False, True),
        # Low-to-high multi-mutants
        ("low-to-high-mutant/mutant", "IsPETase", True, True),
        ("low-to-high-mutant/mutant", "BhrPETase", True, True),
        ("low-to-high-mutant/mutant", "MipaPETase", False, True),
    ]

    records: list[dict[str, Any]] = []

    for split_path_str, enzyme, has_act, has_tm in dataset_configs:
        dataset_split = "single-site-mutant" if "single-site" in split_path_str else "low-to-high-mutant"
        wt_seq = parent_fastas[enzyme]

        act_csv = petgym_dir / split_path_str / f"{enzyme}_Act" / f"{enzyme}_Act.csv" if has_act else None
        tm_csv = petgym_dir / split_path_str / f"{enzyme}_Tm" / f"{enzyme}_Tm.csv" if has_tm else None

        df_act = pd.read_csv(act_csv) if act_csv and act_csv.exists() else None
        df_tm = pd.read_csv(tm_csv) if tm_csv and tm_csv.exists() else None

        # Build list of row records
        # If both exist, the row count is identical (except in Bhr low-to-high where both have same duplicate)
        primary_df = df_act if df_act is not None else df_tm
        if primary_df is None:
            continue

        source_files = []
        if act_csv and act_csv.exists():
            source_files.append(str(act_csv.relative_to(petgym_dir).as_posix()))
        if tm_csv and tm_csv.exists():
            source_files.append(str(tm_csv.relative_to(petgym_dir).as_posix()))
        source_files_str = ";".join(source_files)

        mutants_seen_in_dataset: dict[str, int] = {}

        for idx, row in primary_df.iterrows():
            mutant_raw = str(row["mutant"]).strip()

            # Track duplicate mutants within this dataset split
            occurrence = mutants_seen_in_dataset.get(mutant_raw, 0) + 1
            mutants_seen_in_dataset[mutant_raw] = occurrence
            is_dup = occurrence > 1

            # Extract scores
            act_val = float(df_act.iloc[idx]["score"]) if df_act is not None else None
            tm_val = float(df_tm.iloc[idx]["score"]) if df_tm is not None else None

            # Parse mutations
            parsed_muts = parse_mutations(mutant_raw)
            canonical_mut = canonicalize_mutations(parsed_muts)
            num_mutations = len(parsed_muts)
            is_multi = num_mutations > 1

            # Validate against parent sequence
            is_valid, validation_errors = validate_mutations(wt_seq, parsed_muts)
            if not is_valid:
                raise ValueError(
                    f"Validation failed for {enzyme} mutant '{mutant_raw}': {validation_errors}"
                )

            # Apply mutations to produce variant sequence
            mutant_seq = apply_mutations(wt_seq, parsed_muts, strict=False)

            # Extract parent positions (1-based relative to parent sequence)
            parent_positions = ":".join(str(m[1]) for m in sorted(parsed_muts, key=lambda m: m[1]))

            # Map to 5XJH structure numbering
            struct_pos_str, struct_status = map_mutation_positions_to_5xjh(
                enzyme, parsed_muts, wt_seq, ispetase_seq
            )

            # Quality flags & outlier detection
            flags: list[str] = []
            is_outlier = False

            if is_dup:
                flags.append("duplicate")
            if tm_val is not None and tm_val > 100.0:
                is_outlier = True
                flags.append(f"unphysical_high_tm({tm_val})")
            if not flags:
                flags.append("clean")

            # Stable variant ID
            base_vid = f"{enzyme}_{canonical_mut}"
            variant_id = f"{base_vid}#dup{occurrence}" if is_dup else base_vid
            row_id = f"{dataset_split}_{enzyme}_{idx + 1:04d}"

            records.append({
                "row_id": row_id,
                "variant_id": variant_id,
                "parent_enzyme": enzyme,
                "dataset_split": dataset_split,
                "study_id": "PET-Gym",
                "source_files": source_files_str,
                "source_row_index": idx,
                "mutant_raw": mutant_raw,
                "mutations_canonical": canonical_mut,
                "num_mutations": num_mutations,
                "is_multi_mutant": is_multi,
                "wildtype_sequence": wt_seq,
                "mutant_sequence": mutant_seq,
                "sequence_length": len(mutant_seq),
                "parent_mutation_positions": parent_positions,
                "structure_5xjh_positions": struct_pos_str,
                "structure_mapping_status": struct_status,
                "activity_raw": act_val,
                "activity_unit": ACTIVITY_UNITS.get(enzyme, "none") if act_val is not None else "none",
                "tm_raw": tm_val,
                "tm_unit": "celsius" if tm_val is not None else "none",
                "is_duplicate": is_dup,
                "is_outlier": is_outlier,
                "data_quality_flags": ",".join(flags),
            })

    return pd.DataFrame(records)


def build_and_save_variants_table(
    raw_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Build and serialize canonical variants table to Parquet."""
    petgym_dir = raw_dir / "petflow" / "petflow" / "PET-Gym"
    if not petgym_dir.exists():
        raise FileNotFoundError(f"PET-Gym directory not found at: {petgym_dir}")

    df = curate_petgym_dataset(petgym_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)

    # Roundtrip assertion test
    reloaded = pd.read_parquet(output_path, engine="pyarrow")
    if len(reloaded) != len(df):
        raise IOError(f"Parquet roundtrip failed: wrote {len(df)} rows, reloaded {len(reloaded)} rows")

    return reloaded


def generate_data_quality_report(df: pd.DataFrame, report_path: Path) -> str:
    """Generate markdown data quality report for curated variants."""
    total_variants = len(df)
    unique_vids = df["variant_id"].nunique()
    multi_count = int(df["is_multi_mutant"].sum())
    single_count = total_variants - multi_count

    enzyme_counts = df["parent_enzyme"].value_counts().to_dict()
    split_counts = df["dataset_split"].value_counts().to_dict()
    act_counts = df["activity_raw"].notnull().sum()
    tm_counts = df["tm_raw"].notnull().sum()
    both_counts = (df["activity_raw"].notnull() & df["tm_raw"].notnull()).sum()

    dup_count = int(df["is_duplicate"].sum())
    outlier_count = int(df["is_outlier"].sum())

    struct_status_counts = df["structure_mapping_status"].value_counts().to_dict()

    report_lines = [
        "# Q-Catalyst Data Quality & Curation Report",
        "",
        "## 1. Curation Overview",
        f"- **Total Processed Rows:** {total_variants}",
        f"- **Unique Biological Variant IDs:** {unique_vids}",
        f"- **Single-Site Mutants:** {single_count}",
        f"- **Multi-Site Mutants:** {multi_count}",
        f"- **Variants with Activity:** {act_counts}",
        f"- **Variants with Tm:** {tm_counts}",
        f"- **Variants with Both Act & Tm:** {both_counts}",
        "",
        "## 2. Records by Parent Enzyme",
        "| Parent Enzyme | Count | Has Activity | Has Tm | Split Distribution |",
        "|---|---|---|---|---|",
    ]

    for enzyme in sorted(df["parent_enzyme"].unique()):
        sub = df[df["parent_enzyme"] == enzyme]
        has_act = sub["activity_raw"].notnull().any()
        has_tm = sub["tm_raw"].notnull().any()
        splits = ", ".join(f"{k}: {v}" for k, v in sub["dataset_split"].value_counts().items())
        report_lines.append(
            f"| `{enzyme}` | {len(sub)} | {has_act} | {has_tm} | {splits} |"
        )

    report_lines.extend([
        "",
        "## 3. Structure Mapping to Reference 5XJH",
        "- **Reference Structure:** [5XJH.pdb](file:///c:/Users/Asus/Desktop/QHacks/qcatalyst/data/structures/5XJH.pdb) (IsPETase mature domain, residues 37..290).",
        "- **Numbering Rule:** Parent residue numbering is preserved in `parent_mutation_positions`. Mapped positions on 5XJH are kept strictly separate in `structure_5xjh_positions`.",
        "- **Mapping Breakdown:**",
    ])
    for status, cnt in struct_status_counts.items():
        report_lines.append(f"  - `{status}`: {cnt} variants")

    report_lines.extend([
        "",
        "## 4. Raw Data Anomalies & Unresolved Data Issues",
        f"1. **Duplicate Records ({dup_count} detected):**",
        "   - In `low-to-high-mutant/BhrPETase`, mutant `H218S:F222I:H191D` occurs twice with identical scores.",
        "   - Handled by preserving both rows with unique `row_id`, adding `#dup2` to `variant_id`, and setting `is_duplicate=True`.",
        f"2. **Physical Outliers ({outlier_count} detected):**",
        "   - In `low-to-high-mutant/BhrPETase_Tm`, mutant `H218S:F222I:F243T` has raw $T_m = 842.0^\\circ\\text{C}$.",
        "   - Flagged with `is_outlier=True` and `unphysical_high_tm(842.0)`. Preserved as raw value for transparency without corrupting downstream models.",
        "3. **Incomparable Activity Labels:**",
        "   - Raw activity measurements across `IsPETase`, `BhrPETase`, and `CaPETase` originate from different assays and units.",
        "   - `activity_unit` explicitly identifies each source (`relative_fold_change_ispetase`, `activity_bhr_unpooled`, `activity_ca_unpooled`) and prevents uncalibrated pooling.",
        "",
        "## 5. Output Parquet Schema",
        "| Field | Type | Description |",
        "|---|---|---|",
        "| `row_id` | `string` | Unique dataset row identifier (`{split}_{enzyme}_{index}`) |",
        "| `variant_id` | `string` | Stable variant identifier (`{enzyme}_{canonical_mut}`) |",
        "| `parent_enzyme` | `string` | Parent wild-type enzyme name |",
        "| `dataset_split` | `string` | `single-site-mutant` vs `low-to-high-mutant` |",
        "| `study_id` | `string` | Source study identifier (`PET-Gym`) |",
        "| `source_files` | `string` | Relative paths to source CSV files |",
        "| `source_row_index` | `int64` | Zero-based row index in source CSV |",
        "| `mutant_raw` | `string` | Raw unnormalized mutation string from source CSV |",
        "| `mutations_canonical` | `string` | Canonical position-sorted colon-separated substitutions |",
        "| `num_mutations` | `int64` | Mutation count (1 for single, >= 2 for multi) |",
        "| `is_multi_mutant` | `bool` | True if multi-site substitution |",
        "| `wildtype_sequence` | `string` | Parent full-length FASTA sequence |",
        "| `mutant_sequence` | `string` | Generated variant sequence with substitutions applied |",
        "| `sequence_length` | `int64` | Amino acid length of variant sequence |",
        "| `parent_mutation_positions` | `string` | Colon-separated 1-based positions on parent sequence |",
        "| `structure_5xjh_positions` | `string` | Colon-separated 1-based mapped positions on 5XJH structure |",
        "| `structure_mapping_status` | `string` | Mapping status (`exact_1to1`, `aligned`, `partial_aligned`) |",
        "| `activity_raw` | `float64` | Preserved raw activity score (or NaN) |",
        "| `activity_unit` | `string` | Unpooled measurement unit or `none` |",
        "| `tm_raw` | `float64` | Preserved raw melting temperature (or NaN) |",
        "| `tm_unit` | `string` | `celsius` or `none` |",
        "| `is_duplicate` | `bool` | True if duplicated in source dataset |",
        "| `is_outlier` | `bool` | True if unphysical value detected |",
        "| `data_quality_flags` | `string` | Comma-separated quality flags (`clean`, `duplicate`, etc.) |",
    ])

    report_text = "\n".join(report_lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    return report_text
