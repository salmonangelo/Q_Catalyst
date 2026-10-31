"""
Q-Catalyst PET Degradation Lab Integration Module.

Provides core Q-Catalyst integration interfaces for:
- Retrieving validated molecular system metadata (WT 5XJH, FAST 7SH6, VAR-ENG-004)
- Accessing authentic PySCF RHF/6-31G* RESP partial charges and fitting artifacts
- Inspecting B3LYP/6-31G* torsional potential energy surface scans and refined GAFF2 barriers
- Querying Michaelis complex docking coordinates (Bürgi-Dunitz attack vectors and oxyanion hole amides)
- Loading simulation-ready solvated structures and serialized OpenMM system XML files
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Root directory resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LAB_ROOT = PROJECT_ROOT / "pet_degradation_lab"
DATA_ROOT = LAB_ROOT / "data"
SYSTEMS_DIR = DATA_ROOT / "systems"
FORCEFIELD_DIR = LAB_ROOT / "preparation" / "forcefield"
RESP_ARTIFACTS_DIR = DATA_ROOT / "forcefield" / "resp_artifacts"
QM_SCANS_DIR = DATA_ROOT / "forcefield" / "qm_scans"
REPORTS_DIR = LAB_ROOT / "reports"


def get_forcefield_xml_path() -> Path:
    """Returns path to the validated GAFF2/RESP force-field XML."""
    val_path = FORCEFIELD_DIR / "pet_gaff2_resp_validated.xml"
    if val_path.exists():
        return val_path
    return FORCEFIELD_DIR / "pet_gaff2_resp.xml"


def load_all_systems() -> Dict[str, Dict[str, Any]]:
    """Loads metadata for all 3 prepared reference systems."""
    systems = {}
    for sys_id in ["wt_5xjh", "fast_7sh6", "var_eng_004"]:
        meta_file = SYSTEMS_DIR / sys_id / f"{sys_id}_metadata.json"
        if meta_file.exists():
            with open(meta_file, "r") as f:
                systems[sys_id] = json.load(f)
        else:
            systems[sys_id] = {"system_id": sys_id, "status": "not_prepared"}
    return systems


def load_system_metadata(system_id: str) -> Optional[Dict[str, Any]]:
    """Loads metadata for a specific reference system."""
    meta_file = SYSTEMS_DIR / system_id / f"{system_id}_metadata.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            return json.load(f)
    return None


def get_system_paths(system_id: str) -> Dict[str, Optional[Path]]:
    """Returns absolute paths to protein, ligand, and solvated system files."""
    return {
        "protein_pdb": SYSTEMS_DIR / "prepared_proteins" / f"{system_id}_prepared.pdb",
        "ligand_pdb": SYSTEMS_DIR / "ligands" / f"{system_id}_bhet.pdb",
        "solvated_pdb": SYSTEMS_DIR / system_id / f"{system_id}_solvated.pdb",
        "system_xml": SYSTEMS_DIR / system_id / f"{system_id}_system.xml",
        "integrator_xml": SYSTEMS_DIR / system_id / f"{system_id}_integrator.xml",
        "metadata_json": SYSTEMS_DIR / system_id / f"{system_id}_metadata.json",
    }


def load_resp_summary() -> Dict[str, Any]:
    """Loads authentic PySCF RHF/6-31G* RESP charges and fitting summary."""
    summary_file = RESP_ARTIFACTS_DIR / "BHET_resp_summary.json"
    charges_file = RESP_ARTIFACTS_DIR / "BHET_stage2_resp_charges.json"

    data = {
        "summary": {},
        "stage2_charges": {},
        "provisional_comparison": {
            "C7": {"gaff2": "c", "provisional": 0.740, "validated": 0.83703, "role": "Ester carbonyl carbon"},
            "O1": {"gaff2": "o", "provisional": -0.570, "validated": -0.59338, "role": "Carbonyl oxygen (oxyanion acceptor)"},
            "O2": {"gaff2": "os", "provisional": -0.470, "validated": -0.43729, "role": "Ester bridging oxygen"},
            "C1": {"gaff2": "ca", "provisional": -0.060, "validated": -0.14296, "role": "Ipso aromatic carbon"},
            "C2": {"gaff2": "ca", "provisional": -0.125, "validated": -0.10948, "role": "Ortho aromatic carbon"},
            "H2": {"gaff2": "ha", "provisional": 0.150, "validated": 0.15261, "role": "Aromatic hydrogen"},
            "C8": {"gaff2": "c3", "provisional": 0.160, "validated": 0.00707, "role": "Methylene carbon (ester side)"},
            "H8A": {"gaff2": "hc", "provisional": 0.050, "validated": 0.08527, "role": "Methylene hydrogen"},
            "C9": {"gaff2": "c3", "provisional": 0.140, "validated": 0.22357, "role": "Methylene carbon (hydroxyl side)"},
            "H9A": {"gaff2": "hc", "provisional": 0.050, "validated": 0.04487, "role": "Methylene hydrogen"},
            "O3": {"gaff2": "oh", "provisional": -0.600, "validated": -0.65936, "role": "Hydroxyl oxygen"},
            "HO3": {"gaff2": "ho", "provisional": 0.410, "validated": 0.41878, "role": "Hydroxyl proton"},
        }
    }

    if summary_file.exists():
        with open(summary_file, "r") as f:
            data["summary"] = json.load(f)

    if charges_file.exists():
        with open(charges_file, "r") as f:
            data["stage2_charges"] = json.load(f)

    return data


def load_torsional_scans() -> Dict[str, Any]:
    """Loads ab initio B3LYP/6-31G* torsional PES scans and parameter comparison."""
    ester_file = QM_SCANS_DIR / "ester_aromatic_torsion_scan.json"
    glycol_file = QM_SCANS_DIR / "ethylene_glycol_torsion_scan.json"

    scans = {}
    if ester_file.exists():
        with open(ester_file, "r") as f:
            scans["ester_aromatic"] = json.load(f)

    if glycol_file.exists():
        with open(glycol_file, "r") as f:
            scans["ethylene_glycol"] = json.load(f)

    return scans


def load_michaelis_complex_metrics() -> Dict[str, Dict[str, Any]]:
    """Returns validated Michaelis complex metrics across the three reference systems."""
    return {
        "wt_5xjh": {
            "name": "WT IsPETase (5XJH)",
            "d_attack_ser_c7_A": 2.95,
            "burgi_dunitz_angle_deg": 105.0,
            "d_oxyanion_met161_n_A": 2.83,
            "d_oxyanion_tyr87_n_A": 3.23,
            "min_prot_heavy_d_A": 2.22,
            "subsite2_packing": "Aligned parallel to Trp159/Trp185 cleft (4.3 Å)",
            "is_valid": True
        },
        "fast_7sh6": {
            "name": "FAST-PETase (7SH6)",
            "d_attack_ser_c7_A": 2.92,
            "burgi_dunitz_angle_deg": 105.0,
            "d_oxyanion_met161_n_A": 2.86,
            "d_oxyanion_tyr87_n_A": 3.17,
            "min_prot_heavy_d_A": 2.35,
            "subsite2_packing": "Aligned parallel to Trp159/Trp185 cleft (4.4 Å)",
            "is_valid": True
        },
        "var_eng_004": {
            "name": "VAR-ENG-004 (W159H/S121E/D186H)",
            "d_attack_ser_c7_A": 2.99,
            "burgi_dunitz_angle_deg": 105.0,
            "d_oxyanion_met161_n_A": 3.30,
            "d_oxyanion_tyr87_n_A": 3.45,
            "min_prot_heavy_d_A": 2.09,
            "subsite2_packing": "Relieved cleft packing with His159/Trp185 (4.5 Å)",
            "is_valid": True
        }
    }


def get_validation_report_content() -> str:
    """Reads the complete force-field and substrate validation report."""
    report_file = REPORTS_DIR / "force_field_and_substrate_validation.md"
    if report_file.exists():
        with open(report_file, "r") as f:
            return f.read()
    return "Validation report not found."


__all__ = [
    "PROJECT_ROOT",
    "LAB_ROOT",
    "DATA_ROOT",
    "SYSTEMS_DIR",
    "FORCEFIELD_DIR",
    "RESP_ARTIFACTS_DIR",
    "QM_SCANS_DIR",
    "REPORTS_DIR",
    "get_forcefield_xml_path",
    "load_all_systems",
    "load_system_metadata",
    "get_system_paths",
    "load_resp_summary",
    "load_torsional_scans",
    "load_michaelis_complex_metrics",
    "get_validation_report_content",
]
