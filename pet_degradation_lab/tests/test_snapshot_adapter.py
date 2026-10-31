"""Unit tests verifying ProjectSnapshotAdapter isolation, immutability, and checksum validation."""
import hashlib
import json
import tempfile
from pathlib import Path
import pytest

from ingestion.snapshot_adapter import ProjectSnapshotAdapter


@pytest.fixture
def temp_source_and_snapshot_dirs():
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as snap_dir:
        src_path = Path(src_dir)
        snap_path = Path(snap_dir)

        # Create dummy source files
        f1 = src_path / "ranking.parquet"
        f1.write_bytes(b"SYNTHETIC_PARQUET_CONTENT_FOR_TESTS")

        f2 = src_path / "5XJH.pdb"
        f2.write_text("ATOM      1  CA  SER A 160       0.0 0.0 0.0\nEND\n")

        yield src_path, snap_path


def test_snapshot_adapter_dry_run(temp_source_and_snapshot_dirs):
    src_dir, snap_dir = temp_source_and_snapshot_dirs
    adapter = ProjectSnapshotAdapter(
        source_root=src_dir,
        snapshots_dir=snap_dir,
        allowlist=["ranking.parquet", "5XJH.pdb"]
    )

    plan = adapter.create_snapshot(snapshot_id="test_dry_snap", dry_run=True)
    assert plan["status"] == "DRY_RUN_SUCCESS"
    assert plan["total_files"] == 2
    assert not (snap_dir / "test_dry_snap").exists()


def test_snapshot_adapter_creates_immutable_snapshot(temp_source_and_snapshot_dirs):
    src_dir, snap_dir = temp_source_and_snapshot_dirs
    adapter = ProjectSnapshotAdapter(
        source_root=src_dir,
        snapshots_dir=snap_dir,
        allowlist=["ranking.parquet", "5XJH.pdb"]
    )

    # 1. First snapshot creation succeeds
    res = adapter.create_snapshot(snapshot_id="snap_immutable_test", dry_run=False)
    assert res["status"] == "SNAPSHOT_CREATED"
    target_dir = snap_dir / "snap_immutable_test"
    assert target_dir.exists()

    # Verify checksums match exactly
    src_f1_hash = hashlib.sha256((src_dir / "ranking.parquet").read_bytes()).hexdigest()
    dst_f1_hash = hashlib.sha256((target_dir / "raw" / "ranking.parquet").read_bytes()).hexdigest()
    assert src_f1_hash == dst_f1_hash

    # Verify manifest exists
    assert (target_dir / "snapshot_manifest.json").exists()

    # 2. Second creation with SAME snapshot_id MUST raise FileExistsError
    with pytest.raises(FileExistsError, match="Refusing to overwrite existing snapshot directory"):
        adapter.create_snapshot(snapshot_id="snap_immutable_test", dry_run=False)


def test_snapshot_adapter_never_modifies_source_files(temp_source_and_snapshot_dirs):
    src_dir, snap_dir = temp_source_and_snapshot_dirs
    src_file = src_dir / "5XJH.pdb"
    mtime_before = src_file.stat().st_mtime_ns
    bytes_before = src_file.read_bytes()

    adapter = ProjectSnapshotAdapter(
        source_root=src_dir,
        snapshots_dir=snap_dir,
        allowlist=["5XJH.pdb"]
    )
    adapter.create_snapshot(snapshot_id="snap_mtime_test", dry_run=False)

    mtime_after = src_file.stat().st_mtime_ns
    bytes_after = src_file.read_bytes()

    assert mtime_before == mtime_after
    assert bytes_before == bytes_after
