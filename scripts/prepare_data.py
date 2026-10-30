"""Script for data preparation, auditing, and canonical curation in Q-Catalyst.

Usage:
    python scripts/prepare_data.py --audit-only
    python scripts/prepare_data.py --curate
    python scripts/prepare_data.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to sys.path so qcatalyst can be imported without installation
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qcatalyst.data.audit import (
    audit_all_csvs,
    generate_readable_summary,
    scan_file_manifest,
)
from qcatalyst.data.curation import (
    build_and_save_variants_table,
    generate_data_quality_report,
)


def run_audit(raw_dir: Path, reports_dir: Path) -> int:
    """Execute full audit of raw datasets and generate manifest and summary reports."""
    start_time = time.time()
    print("=" * 70)
    print("Q-CATALYST: Data Audit & Manifest Generator")
    print("=" * 70)
    print(f"Scanning raw directory: {raw_dir.resolve()}")
    print(f"Writing reports to:     {reports_dir.resolve()}")
    print("Network access:         None (100% local inspection)")
    print("-" * 70)

    if not raw_dir.exists():
        print(f"Error: Raw directory does not exist: {raw_dir}", file=sys.stderr)
        return 1

    # 1. Scan file manifest
    print("1. Scanning file manifest...")
    manifest_df = scan_file_manifest(raw_dir)
    print(f"   Found {len(manifest_df)} files across raw directories.")

    # 2. Filter CSV files for deep audit
    csv_rows = manifest_df[manifest_df["extension"] == ".csv"]
    print(f"2. Auditing {len(csv_rows)} CSV files...")
    csv_paths = [raw_dir / row["file_path"] for _, row in csv_rows.iterrows()]
    audit_df = audit_all_csvs(csv_paths, base_dir=raw_dir)

    # 3. Save table outputs
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    manifest_out = tables_dir / "petgym_file_manifest.csv"
    audit_out = tables_dir / "petgym_csv_audit.csv"
    summary_out = reports_dir / "data_audit_summary.md"

    manifest_df.to_csv(manifest_out, index=False)
    audit_df.to_csv(audit_out, index=False)

    print(f"   Saved manifest table: {manifest_out} ({manifest_out.stat().st_size} bytes)")
    print(f"   Saved audit table:    {audit_out} ({audit_out.stat().st_size} bytes)")

    # 4. Generate readable summary
    summary_text = generate_readable_summary(manifest_df, audit_df)
    summary_out.write_text(summary_text, encoding="utf-8")
    print(f"   Saved summary report: {summary_out} ({summary_out.stat().st_size} bytes)")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Audit completed successfully in {elapsed:.2f}s.")
    print("=" * 70)
    return 0


def run_curation(raw_dir: Path, curated_dir: Path, reports_dir: Path) -> int:
    """Execute Feature 2 canonical curation pipeline into data/curated/variants.parquet."""
    start_time = time.time()
    print("=" * 70)
    print("Q-CATALYST: Feature 2 - Mutation Parsing & Sequence Curation")
    print("=" * 70)
    print(f"Source raw directory:    {raw_dir.resolve()}")
    print(f"Destination curated dir: {curated_dir.resolve()}")
    print(f"Reports directory:       {reports_dir.resolve()}")
    print("Network access:          None (100% local processing)")
    print("-" * 70)

    # Run audit first
    audit_code = run_audit(raw_dir, reports_dir)
    if audit_code != 0:
        return audit_code

    print("\nStarting canonical dataset preparation...")
    variants_parquet = curated_dir / "variants.parquet"
    variants_df = build_and_save_variants_table(raw_dir, variants_parquet)

    print(f"-> Curated {len(variants_df)} variant records.")
    print(f"-> Saved canonical table: {variants_parquet} ({variants_parquet.stat().st_size} bytes)")

    quality_report_path = reports_dir / "data_quality_report.md"
    generate_data_quality_report(variants_df, quality_report_path)
    print(f"-> Saved data quality report: {quality_report_path} ({quality_report_path.stat().st_size} bytes)")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Feature 2 completed successfully in {elapsed:.2f}s.")
    print("=" * 70)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q-Catalyst Data Preparation, Audit, and Curation Pipeline"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="Path to raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--curated-dir",
        type=Path,
        default=REPO_ROOT / "data" / "curated",
        help="Path to curated data directory (default: data/curated)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "reports",
        help="Path to reports directory (default: reports)",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run only data audit without building canonical tables",
    )
    parser.add_argument(
        "--curate",
        action="store_true",
        default=True,
        help="Run both audit and canonical curation (default: True)",
    )

    args = parser.parse_args()

    if args.audit_only:
        code = run_audit(args.raw_dir, args.reports_dir)
    else:
        code = run_curation(args.raw_dir, args.curated_dir, args.reports_dir)
    sys.exit(code)


if __name__ == "__main__":
    main()
