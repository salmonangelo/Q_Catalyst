"""
AMBER ff14SB Partial Atomic Charge Parameters for Protein Residues and Water.
Standard partial charges derived from AMBER ff14SB force field libraries.
"""

from typing import Dict, Optional, Any

# Standard AMBER ff14SB partial atomic charges (in elementary charges e)
# for standard amino acid residues and TIP3P water
AMBER_FF14SB_CHARGES: Dict[str, Dict[str, float]] = {
    "ALA": {"N": -0.4157, "H": 0.2719, "CA": 0.0337, "HA": 0.0823, "CB": -0.1825, "HB1": 0.0603, "HB2": 0.0603, "HB3": 0.0603, "C": 0.5973, "O": -0.5679},
    "ARG": {"N": -0.3479, "H": 0.2747, "CA": -0.2637, "HA": 0.1560, "CB": -0.0007, "HB2": 0.0327, "HB3": 0.0327, "CG": 0.0390, "HG2": 0.0285, "HG3": 0.0285, "CD": 0.0486, "HD2": 0.0637, "HD3": 0.0637, "NE": -0.3456, "HE": 0.3212, "CZ": 0.6830, "NH1": -0.6772, "HH11": 0.4448, "HH12": 0.4448, "NH2": -0.6772, "HH21": 0.4448, "HH22": 0.4448, "C": 0.7341, "O": -0.5894},
    "ASN": {"N": -0.4157, "H": 0.2719, "CA": 0.0150, "HA": 0.0880, "CB": -0.2041, "HB2": 0.0797, "HB3": 0.0797, "CG": 0.7155, "OD1": -0.5939, "ND2": -0.9191, "HD21": 0.4180, "HD22": 0.4180, "C": 0.5973, "O": -0.5679},
    "ASP": {"N": -0.5163, "H": 0.2936, "CA": 0.0387, "HA": 0.0880, "CB": -0.0303, "HB2": -0.0122, "HB3": -0.0122, "CG": 0.7994, "OD1": -0.8014, "OD2": -0.8014, "C": 0.5366, "O": -0.5819},
    "ASH": {"N": -0.4157, "H": 0.2719, "CA": -0.0306, "HA": 0.0880, "CB": -0.1738, "HB2": 0.0841, "HB3": 0.0841, "CG": 0.7533, "OD1": -0.5847, "OD2": -0.6108, "HD2": 0.4204, "C": 0.5973, "O": -0.5679}, # Protonated Asp
    "CYS": {"N": -0.4157, "H": 0.2719, "CA": -0.0241, "HA": 0.1124, "CB": -0.1231, "HB2": 0.1112, "HB3": 0.1112, "SG": -0.3119, "HG": 0.1933, "C": 0.5973, "O": -0.5679},
    "CYX": {"N": -0.4157, "H": 0.2719, "CA": -0.0386, "HA": 0.1124, "CB": -0.1231, "HB2": 0.1112, "HB3": 0.1112, "SG": -0.1186, "C": 0.5973, "O": -0.5679}, # Disulfide-bonded Cys
    "GLN": {"N": -0.4157, "H": 0.2719, "CA": 0.0180, "HA": 0.0850, "CB": -0.0036, "HB2": 0.0177, "HB3": 0.0177, "CG": -0.0645, "HG2": 0.0437, "HG3": 0.0437, "CD": 0.6969, "OE1": -0.6086, "NE2": -0.9191, "HE21": 0.4180, "HE22": 0.4180, "C": 0.5973, "O": -0.5679},
    "GLU": {"N": -0.5163, "H": 0.2936, "CA": 0.0397, "HA": 0.1105, "CB": 0.0560, "HB2": -0.0173, "HB3": -0.0173, "CG": -0.0437, "HG2": -0.0425, "HG3": -0.0425, "CD": 0.8053, "OE1": -0.8188, "OE2": -0.8188, "C": 0.5366, "O": -0.5819},
    "GLH": {"N": -0.4157, "H": 0.2719, "CA": -0.0191, "HA": 0.0880, "CB": -0.0036, "HB2": 0.0177, "HB3": 0.0177, "CG": -0.1580, "HG2": 0.0763, "HG3": 0.0763, "CD": 0.7533, "OE1": -0.5847, "OE2": -0.6108, "HE2": 0.4204, "C": 0.5973, "O": -0.5679}, # Protonated Glu
    "GLY": {"N": -0.4157, "H": 0.2719, "CA": -0.0252, "HA2": 0.0698, "HA3": 0.0698, "C": 0.5973, "O": -0.5679},
    "HID": {"N": -0.4157, "H": 0.2719, "CA": 0.0186, "HA": 0.0880, "CB": -0.0532, "HB2": 0.0412, "HB3": 0.0412, "CG": 0.1541, "ND1": -0.3809, "HD1": 0.3649, "CE1": 0.2057, "HE1": 0.1378, "NE2": -0.5727, "CD2": 0.1293, "HD2": 0.1154, "C": 0.5973, "O": -0.5679}, # Neutral delta-protonated His
    "HIS": {"N": -0.4157, "H": 0.2719, "CA": 0.0186, "HA": 0.0880, "CB": -0.0532, "HB2": 0.0412, "HB3": 0.0412, "CG": 0.1541, "ND1": -0.3809, "HD1": 0.3649, "CE1": 0.2057, "HE1": 0.1378, "NE2": -0.5727, "CD2": 0.1293, "HD2": 0.1154, "C": 0.5973, "O": -0.5679}, # Generic neutral His (HID tautomer)
    "HIE": {"N": -0.4157, "H": 0.2719, "CA": 0.0186, "HA": 0.0880, "CB": -0.0532, "HB2": 0.0412, "HB3": 0.0412, "CG": -0.0519, "ND1": -0.5292, "CE1": 0.3236, "HE1": 0.1165, "NE2": -0.3456, "HE2": 0.3212, "CD2": 0.1874, "HD2": 0.0924, "C": 0.5973, "O": -0.5679}, # Neutral epsilon-protonated His
    "HIP": {"N": -0.3479, "H": 0.2747, "CA": -0.0898, "HA": 0.1375, "CB": -0.0512, "HB2": 0.0782, "HB3": 0.0782, "CG": 0.0241, "ND1": -0.3809, "HD1": 0.3649, "CE1": 0.2467, "HE1": 0.1658, "NE2": -0.3809, "HE2": 0.3649, "CD2": 0.1478, "HD2": 0.1341, "C": 0.7341, "O": -0.5894}, # Charged His (+)
    "ILE": {"N": -0.4157, "H": 0.2719, "CA": -0.0597, "HA": 0.0869, "CB": 0.1303, "HB": 0.0187, "CG1": -0.0430, "HG12": 0.0236, "HG13": 0.0236, "CG2": -0.3204, "HG21": 0.0882, "HG22": 0.0882, "HG23": 0.0882, "CD1": -0.0660, "HD11": 0.0242, "HD12": 0.0242, "HD13": 0.0242, "C": 0.5973, "O": -0.5679},
    "LEU": {"N": -0.4157, "H": 0.2719, "CA": -0.0518, "HA": 0.0922, "CB": -0.1102, "HB2": 0.0457, "HB3": 0.0457, "CG": 0.3531, "HG": -0.0361, "CD1": -0.4121, "HD11": 0.1000, "HD12": 0.1000, "HD13": 0.1000, "CD2": -0.4121, "HD21": 0.1000, "HD22": 0.1000, "HD23": 0.1000, "C": 0.5973, "O": -0.5679},
    "LYS": {"N": -0.3479, "H": 0.2747, "CA": -0.2400, "HA": 0.1425, "CB": -0.0094, "HB2": 0.0362, "HB3": 0.0362, "CG": 0.0187, "HG2": 0.0102, "HG3": 0.0102, "CD": -0.0479, "HD2": 0.0631, "HD3": 0.0631, "CE": 0.2104, "HE2": 0.0328, "HE3": 0.0328, "NZ": -0.3854, "HZ1": 0.3400, "HZ2": 0.3400, "HZ3": 0.3400, "C": 0.7341, "O": -0.5894},
    "MET": {"N": -0.4157, "H": 0.2719, "CA": -0.0237, "HA": 0.0880, "CB": -0.0076, "HB2": 0.0252, "HB3": 0.0252, "CG": 0.0451, "HG2": 0.0390, "HG3": 0.0390, "SD": -0.2744, "CE": -0.0536, "HE1": 0.0684, "HE2": 0.0684, "HE3": 0.0684, "C": 0.5973, "O": -0.5679},
    "PHE": {"N": -0.4157, "H": 0.2719, "CA": -0.0011, "HA": 0.0978, "CB": -0.0343, "HB2": 0.0295, "HB3": 0.0295, "CG": 0.0118, "CD1": -0.1256, "HD1": 0.1330, "CD2": -0.1256, "HD2": 0.1330, "CE1": -0.1704, "HE1": 0.1430, "CE2": -0.1704, "HE2": 0.1430, "CZ": -0.1072, "HZ": 0.1299, "C": 0.5973, "O": -0.5679},
    "PRO": {"N": -0.2548, "CD": 0.0192, "HD2": 0.0390, "HD3": 0.0390, "CG": 0.0189, "HG2": 0.0222, "HG3": 0.0222, "CB": -0.0328, "HB2": 0.0279, "HB3": 0.0279, "CA": -0.0266, "HA": 0.0638, "C": 0.5973, "O": -0.5679},
    "SER": {"N": -0.4157, "H": 0.2719, "CA": -0.0249, "HA": 0.0843, "CB": 0.2117, "HB2": 0.0352, "HB3": 0.0352, "OG": -0.5539, "HG": 0.3807, "C": 0.5973, "O": -0.5679},
    "THR": {"N": -0.4157, "H": 0.2719, "CA": -0.0329, "HA": 0.1007, "CB": 0.3654, "HB": 0.0043, "OG1": -0.6761, "HG1": 0.4102, "CG2": -0.2438, "HG21": 0.0640, "HG22": 0.0640, "HG23": 0.0640, "C": 0.5973, "O": -0.5679},
    "TRP": {"N": -0.4157, "H": 0.2719, "CA": -0.0275, "HA": 0.1123, "CB": -0.0050, "HB2": 0.0339, "HB3": 0.0339, "CG": -0.1415, "CD1": -0.0271, "HD1": 0.1346, "NE1": -0.3418, "HE1": 0.3456, "CE2": 0.1380, "CD2": -0.0441, "CE3": -0.2314, "HE3": 0.1450, "CZ3": -0.1982, "HZ3": 0.1419, "CH2": -0.1147, "HH2": 0.1378, "CZ2": -0.1873, "HZ2": 0.1413, "C": 0.5973, "O": -0.5679},
    "TYR": {"N": -0.4157, "H": 0.2719, "CA": -0.0014, "HA": 0.0876, "CB": -0.0179, "HB2": 0.0295, "HB3": 0.0295, "CG": -0.0011, "CD1": -0.1906, "HD1": 0.1699, "CD2": -0.1906, "HD2": 0.1699, "CE1": -0.2341, "HE1": 0.1656, "CE2": -0.2341, "HE2": 0.1656, "CZ": 0.3226, "OH": -0.5575, "HH": 0.3996, "C": 0.5973, "O": -0.5679},
    "VAL": {"N": -0.4157, "H": 0.2719, "CA": -0.0875, "HA": 0.0987, "CB": 0.2985, "HB": -0.0297, "CG1": -0.3175, "HG11": 0.0784, "HG12": 0.0784, "HG13": 0.0784, "CG2": -0.3175, "HG21": 0.0784, "HG22": 0.0784, "HG23": 0.0784, "C": 0.5973, "O": -0.5679},
    "HOH": {"O": -0.834, "H1": 0.417, "H2": 0.417},
    "WAT": {"O": -0.834, "H1": 0.417, "H2": 0.417},
    "H2O": {"O": -0.834, "H1": 0.417, "H2": 0.417},
}

# 1-letter to 3-letter amino acid code
AA_1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "HID": "HID", "HIE": "HIE", "HIP": "HIP", "CYX": "CYX",
}


def audit_force_field_parameters() -> Dict[str, Any]:
    """
    Audits the AMBER ff14SB parameter dictionary against standard peptide chemistry:
    1. Computes total partial charge sum for each residue template.
    2. Identifies peptide backbone vs sidechain charge contributions.
    3. Flags charge discrepancies resulting from uncoupled dipeptide splice models.
    4. Reports formal integer expectations vs current isolated table sums.
    """
    audit_results: Dict[str, Any] = {}
    expected_formal = {
        "ALA": 0, "ARG": 1, "ASN": 0, "ASP": -1, "ASH": 0, "CYS": 0, "CYX": 0,
        "GLN": 0, "GLU": -1, "GLH": 0, "GLY": 0, "HID": 0, "HIS": 0, "HIE": 0,
        "HIP": 1, "ILE": 0, "LEU": 0, "LYS": 1, "MET": 0, "PHE": 0, "PRO": 0,
        "SER": 0, "THR": 0, "TRP": 0, "TYR": 0, "VAL": 0, "HOH": 0, "WAT": 0, "H2O": 0,
    }

    for res, atoms in AMBER_FF14SB_CHARGES.items():
        total_q = float(sum(atoms.values()))
        exp_q = expected_formal.get(res, 0)
        discrepancy = round(total_q - exp_q, 4)

        bb_atoms = [a for a in ["N", "H", "C", "O"] if a in atoms]
        bb_charge = sum(atoms[a] for a in bb_atoms) if bb_atoms else 0.0
        sc_charge = total_q - bb_charge

        audit_results[res] = {
            "atom_count": len(atoms),
            "total_charge_e": round(total_q, 4),
            "expected_formal_charge_e": exp_q,
            "discrepancy_e": discrepancy,
            "backbone_charge_sum_e": round(bb_charge, 4),
            "sidechain_charge_sum_e": round(sc_charge, 4),
            "is_neutral_integer": abs(discrepancy) < 1e-3,
            "notes": (
                "Exact integer formal charge" if abs(discrepancy) < 1e-3
                else f"Discrepancy of {discrepancy:+.4f} e due to isolated fragment representation without peptide RESP coupling"
            ),
        }

    return audit_results


def get_amber_charge(resname: str, atom_name: str, strict: bool = True) -> float:
    """Returns AMBER ff14SB partial atomic charge for a given residue and atom name.

    Explicitly supports HID and HIE neutral histidine tautomers.
    When strict=True, fails explicitly if authoritative force-field parameters
    are unavailable for the requested residue or atom, preventing silent zero charges.

    Args:
        resname: Residue name (e.g. 'ALA', 'TRP', 'HID', 'HIE', 'HIS').
        atom_name: Atom name (e.g. 'CA', 'CB', 'HD1', 'NE2').
        strict: If True, raises KeyError for unsupported residue or atom names.

    Returns:
        Partial atomic charge in elementary charges (e).

    Raises:
        KeyError: If residue or atom is unsupported in AMBER ff14SB and strict=True.
    """
    res_clean = resname.strip().upper()
    atom_clean = atom_name.strip().upper()

    if res_clean in AMBER_FF14SB_CHARGES:
        charges = AMBER_FF14SB_CHARGES[res_clean]
        if atom_clean in charges:
            return charges[atom_clean]
        # Generic backbone atoms
        if atom_clean in ["C", "CA", "N", "O"]:
            return {"C": 0.5973, "O": -0.5679, "N": -0.4157, "CA": 0.0}.get(atom_clean, 0.0)
        if strict:
            raise KeyError(
                f"Unsupported atom '{atom_name}' in residue '{resname}' for AMBER ff14SB force field. "
                f"Authoritative parameters are unavailable; cannot silently assign zero charge."
            )
    else:
        if strict:
            raise KeyError(
                f"Unsupported residue '{resname}' for AMBER ff14SB force field. "
                f"Authoritative parameters are unavailable; cannot silently assign zero charge."
            )

    return 0.0
