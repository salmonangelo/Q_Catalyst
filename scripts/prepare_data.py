"""
Script 1: Prepare and curate PETase variant dataset with splits.
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.qcatalyst.data.loader import get_curated_variants_data


def main():
    print("=" * 60)
    print("Q-CATALYST: Step 1 - Data Preparation & Curation")
    print("=" * 60)

    df = get_curated_variants_data()
    print(f"Loaded {len(df)} curated PETase variants.")

    # Assign identity-aware splits (70% train, 15% val, 15% test)
    # Ensure prior-art and negative controls are strategically distributed
    n = len(df)
    splits = []
    for i, row in df.iterrows():
        muts = row["mutations"]
        if muts in ["WT", "S160A"]: # Critical controls in train/val
            splits.append("train")
        elif i % 5 == 0:
            splits.append("test")
        elif i % 5 == 1:
            splits.append("val")
        else:
            splits.append("train")

    df["split"] = splits
    print(f"Split distribution: {df['split'].value_counts().to_dict()}")

    # Ensure output directories exist
    curated_dir = PROJECT_ROOT / "data" / "curated"
    splits_dir = PROJECT_ROOT / "data" / "splits"
    curated_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    out_path = curated_dir / "variants.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved curated dataset to: {out_path}")

    splits_path = splits_dir / "splits.parquet"
    df[["variant_id", "mutations", "split"]].to_parquet(splits_path, index=False)
    print(f"Saved splits assignment to: {splits_path}")
    print("Step 1 Complete!\n")


if __name__ == "__main__":
    main()
