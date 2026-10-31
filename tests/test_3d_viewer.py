import re
import json
from pathlib import Path
from src.qcatalyst.structure.proximity import calculate_variant_min_distance, is_eligible_for_chemistry

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_mutation_residue_extraction():
    variants_to_test = {
        "VAR-ENG-004": ("S121E,D186H,R280A,N246D,W159H,F229Y,I168R,S214H", 8, [121, 159, 168, 186, 214, 229, 246, 280]),
        "VAR-CAN-022": ("S121E,D186H,W159H,F229Y", 4, [121, 159, 186, 229]),
        "VAR-CAN-012": ("S121E,D186H", 2, [121, 186]),
        "VAR-ENG-001": ("N233K,S121E,D186H,R224Q,R280A", 5, [121, 186, 224, 233, 280]),
        "VAR-WT-000": ("WT", 0, []),
    }

    for vid, (active_muts, expected_count, expected_resis) in variants_to_test.items():
        mut_residue_nums = []
        if active_muts and active_muts != "WT":
            for m in active_muts.split(","):
                m = m.strip()
                num_match = re.search(r'\d+', m)
                if num_match:
                    mut_residue_nums.append(int(num_match.group()))
        mut_residue_nums = sorted(list(set(mut_residue_nums)))
        assert len(mut_residue_nums) == expected_count
        assert mut_residue_nums == expected_resis

def test_variant_pdb_files_exist_and_contain_waters():
    variants_dir = PROJECT_ROOT / "data" / "structures" / "variants"
    vids = ["VAR-ENG-004", "VAR-CAN-022", "VAR-ENG-001", "VAR-WT-000"]
    for vid in vids:
        pdb_path = variants_dir / f"{vid}.pdb"
        assert pdb_path.exists(), f"Missing PDB for {vid}"
        text = pdb_path.read_text(encoding="utf-8")
        assert len(text) > 100000
        hoh_lines = [l for l in text.splitlines() if "HOH" in l and l.startswith("HETATM")]
        assert len(hoh_lines) > 50, f"Expected water molecules in {vid}, found {len(hoh_lines)}"

def test_viewer3d_capabilities():
    import pandas as pd
    from src.qcatalyst.dashboard.viewer3d import (
        CATALYTIC_TRIAD_RESIS,
        SUBSTRATE_BINDING_RESIS,
        parse_pdb_ca_and_functional,
        get_mutation_inspection_data,
        get_confirmed_qm_interactions,
        generate_3dmol_html
    )

    # 1. Verify Catalytic Triad & Substrate Binding definitions
    assert CATALYTIC_TRIAD_RESIS == [160, 206, 237]
    assert SUBSTRATE_BINDING_RESIS == [87, 161, 185, 208]

    # 2. Parse 5XJH.pdb
    pdb_path = PROJECT_ROOT / "data" / "structures" / "5XJH.pdb"
    pdb_content = pdb_path.read_text(encoding="utf-8")
    ca_map, atom_map, resolved_resis = parse_pdb_ca_and_functional(pdb_content)

    for r in CATALYTIC_TRIAD_RESIS + SUBSTRATE_BINDING_RESIS:
        assert r in resolved_resis, f"Residue {r} missing from 5XJH.pdb"
        assert r in ca_map, f"CA coordinate missing for residue {r}"

    # 3. Test Mutation Inspection Data for VAR-ENG-004
    mut_str = "S121E,D186H,R280A,N246D,W159H,F229Y,I168R,S214H"
    insp = get_mutation_inspection_data(mut_str, ca_map)
    assert len(insp) == 8
    assert 121 in insp
    assert insp[121]["orig_name"] == "Serine"
    assert insp[121]["orig_letter"] == "S"
    assert insp[121]["mut_name"] == "Glutamic acid"
    assert insp[121]["mut_letter"] == "E"
    assert "Ser160" in insp[121]["location_class"]
    assert "Pro120" in insp[121]["nearby_residues"]
    assert "Ser160" in insp[159]["nearby_residues"]

    # 4. Strict Interaction Analysis (Requirement D)
    chem_df = pd.read_parquet(PROJECT_ROOT / "results" / "chemistry" / "chem_results.parquet")
    
    # 4a. Tested on variant present in chem_results (VAR-ENG-004)
    lines_eng, ok_eng, msg_eng = get_confirmed_qm_interactions("VAR-ENG-004", chem_df, atom_map)
    assert ok_eng is True
    assert len(lines_eng) >= 3  # H-Bond His-Ser, H-Bond Asp-His, Attack Vector, Salt Bridge
    assert any("His237" in l["label"] for l in lines_eng)
    assert any("Asp206" in l["label"] for l in lines_eng)
    assert any("Attack" in l["label"] for l in lines_eng)

    # 4b. Tested on non-existent or un-computed variant (must NOT draw any lines)
    lines_none, ok_none, msg_none = get_confirmed_qm_interactions("VAR-NOT-RUN", chem_df, atom_map)
    assert ok_none is False
    assert len(lines_none) == 0
    assert "Strict" in msg_none or "No PySCF QM" in msg_none

    # 5. HTML Generation test
    html = generate_3dmol_html(
        viewer_id="test_view",
        models=[{"pdb": pdb_content[:5000], "role": "variant", "mut_resis": [121, 186]}],
        triad_resis=CATALYTIC_TRIAD_RESIS,
        substrate_resis=SUBSTRATE_BINDING_RESIS,
        interaction_lines=lines_eng,
        mutation_info=insp,
        dark_mode=True
    )
    assert "3Dmol" in html
    assert "Catalytic Triad" in html
    assert "Substrate Binding" in html
    assert "Mutation Inspector" in html

    # 6. Verify New Visual Context Capabilities:
    # Requirement 1: Interactive Hover Labels (Tooltips)
    assert "hover_tooltip_test_view" in html
    assert "setHoverable" in html
    assert "Catalytic Nucleophile" in html
    
    # Requirement 2: The "Active Site" Highlight (Translucent Red Surface)
    assert "addSurface" in html
    assert "#EF4444" in html
    assert "opacity: 0.3" in html or "0.30" in html

    # Requirement 3: The Distance Line & Measurement
    assert "drawDistanceLineToTriad_test_view" in html
    assert "ins_dist_test_view" in html
    assert "#FBBF24" in html

    # Requirement 4: Fixed Labels for Main Triad (Ser160, Asp206, His237)
    assert "addFixedTriadLabels_test_view" in html
    assert "Ser160" in html
    assert "Asp206" in html
    assert "His237" in html
    assert "#FFFFFF" in html


def test_dynamic_pdb_loader_and_missing_metadata():
    from src.qcatalyst.dashboard.viewer3d import (
        CATALYTIC_TRIAD_RESIS,
        SUBSTRATE_BINDING_RESIS,
        parse_pdb_ca_and_functional,
        generate_3dmol_html
    )

    variants_dir = PROJECT_ROOT / "data" / "structures" / "variants"
    test_pdb = variants_dir / "TEST-NEW-001.pdb"
    assert test_pdb.exists(), "TEST-NEW-001.pdb must exist in data/structures/variants/"

    # Test dynamic file scanning logic
    known_vids = {"VAR-ENG-001", "VAR-ENG-002", "VAR-WT-000"}
    custom_pdbs = sorted([p for p in variants_dir.glob("*.pdb") if p.stem not in known_vids and p.stem not in ["5XJH", "7SH6"]])
    custom_names = [p.name for p in custom_pdbs]
    assert "TEST-NEW-001.pdb" in custom_names

    # Test reading and parsing of custom PDB
    content = test_pdb.read_text(encoding="utf-8", errors="ignore")
    ca_map, atom_map, resis = parse_pdb_ca_and_functional(content)
    assert len(ca_map) > 200, f"Expected >200 residues, got {len(ca_map)}"

    # Test 3Dmol viewer generation with missing metadata (raw structure)
    triad = [r for r in CATALYTIC_TRIAD_RESIS if r in resis]
    sub = [r for r in SUBSTRATE_BINDING_RESIS if r in resis]
    raw_html = generate_3dmol_html(
        viewer_id="test_raw_view",
        models=[{"pdb": content[:10000], "role": "variant", "mut_resis": []}],
        triad_resis=triad,
        substrate_resis=sub,
        interaction_lines=[],
        mutation_info={},
        dark_mode=False,
        title="TEST-NEW-001 — Uploaded Custom Structure",
        subtitle=f"Raw PDB Structure ({len(ca_map)} Resolved Residues) | Generic Backbone & Catalytic Sites",
        show_inspector_panel=True
    )
    assert "TEST-NEW-001" in raw_html
    assert "Raw PDB Structure" in raw_html
    assert "Structural Inspector: Raw Variant Structure" in raw_html
    assert "3Dmol" in raw_html



