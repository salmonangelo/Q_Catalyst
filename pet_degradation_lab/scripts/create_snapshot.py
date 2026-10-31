#!/usr/bin/env python3
"""
CLI script to create an immutable, read-only scientific snapshot of Q-Catalyst outputs.
Supports --dry-run, --source-root, --snapshot-id, and --include.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add pet_degradation_lab root to sys.path
LAB_ROOT = Path(__file__).resolve().parent.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from ingestion.snapshot_adapter import ProjectSnapshotAdapter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an immutable, read-only snapshot of Q-Catalyst scientific outputs."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=LAB_ROOT.parent,
        help="Root path to existing Q-Catalyst repository (default: .. relative to pet_degradation_lab)."
    )
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        default=LAB_ROOT / "data" / "project_snapshots",
        help="Directory where snapshots are stored."
    )
    parser.add_argument(
        "--snapshot-id",
        type=str,
        default=None,
        help="Custom snapshot identifier (default: snap_YYYYMMDD_HHMMSS)."
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=None,
        help="Specific file patterns to include (overrides default allowlist)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate snapshot creation without copying files or creating directories."
    )
    args = parser.parse_args()

    adapter = ProjectSnapshotAdapter(
        source_root=args.source_root,
        snapshots_dir=args.snapshots_dir,
    )

    if args.dry_run:
        print("=== EXECUTING SNAPSHOT DRY RUN ===")
        plan = adapter.create_snapshot(
            snapshot_id=args.snapshot_id,
            include_patterns=args.include,
            dry_run=True,
        )
        print(f"Snapshot ID:        {plan['snapshot_id']}")
        print(f"Target Directory:   {plan['target_dir']}")
        print(f"Destination Exists: {plan['destination_exists']}")
        print(f"Files to Copy:      {plan['total_files']} files ({plan['total_bytes'] / (1024*1024):.2f} MB)")
        if plan["missing_sources"]:
            print(f"Missing Sources:    {plan['missing_sources']}")
        print("\nFiles planned for copy:")
        for f in plan["files_to_copy"][:15]:
            print(f"  + {f['source_rel']} ({f['size_bytes']:,} B)")
        if len(plan["files_to_copy"]) > 15:
            print(f"  ... and {len(plan['files_to_copy']) - 15} more files.")
        print("\n[DRY RUN COMPLETE: Zero files modified or copied]")
        return 0

    print("=== CREATING IMMUTABLE SNAPSHOT ===")
    try:
        res = adapter.create_snapshot(
            snapshot_id=args.snapshot_id,
            include_patterns=args.include,
            dry_run=False,
        )
        print(f"Status:             {res['status']}")
        print(f"Snapshot ID:        {res['snapshot_id']}")
        print(f"Target Directory:   {res['target_dir']}")
        print(f"Total Files Copied: {res['total_files']} ({res['total_bytes'] / (1024*1024):.2f} MB)")
        print(f"Manifests Written:  {res.get('manifests_written')}")
        print("\n[SNAPSHOT CREATION SUCCESSFUL]")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Failed to create snapshot: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
