"""Unit tests verifying residue numbering conversions, PDB parsing, and catalytic landmark mapping."""
from pathlib import Path
import pytest

from ingestion.structure_importer import StructureImporter

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_structure_importer_parses_minimal_pdb():
    pdb_path = FIXTURES_DIR / "synthetic_minimal.pdb"
    assert pdb_path.exists()

    # Pass fixture parent as synthetic snapshot dir
    importer = StructureImporter(snapshot_dir=FIXTURES_DIR.parent)
    record = importer.parse_pdb_file(pdb_path, variant_id="SYNTH_WT")

    assert record.variant_id == "SYNTH_WT"
    assert record.file_format == "PDB"
    assert record.chain_id == "A"
    assert record.has_water_molecules is True
    assert record.resolved_residue_count == 3  # SER 160, ASP 206, HIS 237

    # Check residue numbering differentiation
    res_160 = next(r for r in record.residue_mappings if r.pdb_residue_number == 160)
    assert res_160.pdb_residue_number == 160
    assert res_160.amino_acid == "SER"
    assert res_160.is_catalytic_triad is True

    res_237 = next(r for r in record.residue_mappings if r.pdb_residue_number == 237)
    assert res_237.is_catalytic_triad is True

    res_206 = next(r for r in record.residue_mappings if r.pdb_residue_number == 206)
    assert res_206.is_catalytic_triad is True


def test_structure_importer_flags_missing_pdb():
    importer = StructureImporter(snapshot_dir=FIXTURES_DIR.parent)
    with pytest.raises(FileNotFoundError):
        importer.parse_pdb_file(FIXTURES_DIR / "non_existent.pdb", variant_id="NON_EXIST")
