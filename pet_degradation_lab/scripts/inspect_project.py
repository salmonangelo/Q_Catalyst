#!/usr/bin/env python3
"""
CLI script to inspect existing Q-Catalyst repository artifacts
and generate pet_degradation_lab/reports/source_inventory.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add pet_degradation_lab root to sys.path
LAB_ROOT = Path(__file__).resolve().parent.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from ingestion.inventory import SourceInventoryScanner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect existing Q-Catalyst project outputs and produce source_inventory.md."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=LAB_ROOT.parent,
        help="Root directory of the Q-Catalyst repository (default: .. relative to pet_degradation_lab)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LAB_ROOT / "reports" / "source_inventory.md",
        help="Path where source_inventory.md should be written."
    )
    args = parser.parse_args()

    print(f"Scanning Q-Catalyst repository at: {args.source_root.resolve()}")
    scanner = SourceInventoryScanner(args.source_root)
    records = scanner.scan()

    usable_count = sum(1 for r in records if r["usable"])
    print(f"Audited {len(records)} candidate sources: {usable_count} usable, {len(records) - usable_count} missing.")

    scanner.write_markdown_report(records, args.output)
    print(f"Source inventory written to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
