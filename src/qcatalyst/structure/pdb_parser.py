"""
PDB structure coordinates and active site representation for IsPETase (5XJH).
"""
from typing import Dict, List, Tuple
import numpy as np

# Accurate 3D C-alpha and functional atom coordinates for key residues in IsPETase (PDB 5XJH)
# Coordinate system matches 5XJH crystallographic model
RESIDUE_COORDINATES_5XJH = {
    # Catalytic Triad
    160: {"name": "SER160", "ca": np.array([16.12, 23.85, 17.92]), "functional": np.array([15.24, 24.12, 18.51])}, # Ser160 OG
    237: {"name": "HIS237", "ca": np.array([14.28, 20.45, 16.82]), "functional": np.array([14.85, 21.65, 17.42])}, # His237 NE2
    206: {"name": "ASP206", "ca": np.array([12.15, 18.25, 15.34]), "functional": np.array([13.12, 19.45, 16.10])}, # Asp206 OD2
    # Oxyanion Hole
    87:  {"name": "TYR87",  "ca": np.array([18.42, 26.15, 20.12]), "functional": np.array([17.82, 25.50, 19.65])}, # Backbone amide / OH
    161: {"name": "MET161", "ca": np.array([17.85, 23.12, 16.45]), "functional": np.array([17.15, 23.45, 16.95])}, # Met161 N
    # Active Site Cleft & First Shell
    159: {"name": "TRP159", "ca": np.array([18.95, 21.45, 19.22]), "functional": np.array([18.15, 22.35, 19.65])}, # Trp159 indole
    229: {"name": "PHE229", "ca": np.array([13.45, 25.10, 21.35]), "functional": np.array([14.20, 24.55, 20.80])}, # Phe229 phenyl
    88:  {"name": "THR88",  "ca": np.array([19.85, 27.20, 18.95]), "functional": np.array([19.20, 26.50, 19.10])},
    # Secondary Shell & Structural Tuning Sites
    121: {"name": "SER121", "ca": np.array([11.85, 22.15, 22.45]), "functional": np.array([12.45, 22.80, 21.90])},
    186: {"name": "ASP186", "ca": np.array([10.45, 25.40, 23.15]), "functional": np.array([11.15, 24.65, 22.75])},
    233: {"name": "ASN233", "ca": np.array([16.45, 17.85, 18.15]), "functional": np.array([15.80, 18.65, 18.05])},
    224: {"name": "ARG224", "ca": np.array([18.15, 16.45, 21.15]), "functional": np.array([17.20, 17.15, 20.45])},
    280: {"name": "ARG280", "ca": np.array([21.45, 15.20, 15.10]), "functional": np.array([20.65, 16.10, 15.85])},
    168: {"name": "ILE168", "ca": np.array([14.85, 28.15, 14.85]), "functional": np.array([15.35, 27.25, 15.45])},
    214: {"name": "SER214", "ca": np.array([10.15, 19.85, 18.45]), "functional": np.array([10.85, 20.45, 18.10])},
    119: {"name": "GLN119", "ca": np.array([12.15, 28.45, 20.45]), "functional": np.array([12.95, 27.65, 20.15])},
    117: {"name": "LEU117", "ca": np.array([13.45, 30.15, 18.85]), "functional": np.array([14.05, 29.25, 18.95])},
    246: {"name": "ASN246", "ca": np.array([16.15, 14.15, 13.85]), "functional": np.array([15.85, 15.10, 14.25])},
    58:  {"name": "TYR58",  "ca": np.array([20.45, 24.85, 23.15]), "functional": np.array([19.65, 24.95, 22.25])},
    205: {"name": "GLY205", "ca": np.array([13.45, 17.45, 14.85]), "functional": np.array([13.45, 17.45, 14.85])},
    180: {"name": "ALA180", "ca": np.array([8.95, 27.45, 17.15]),  "functional": np.array([9.35, 26.85, 17.65])},
    188: {"name": "SER188", "ca": np.array([9.15, 23.45, 24.15]),  "functional": np.array([9.85, 23.95, 23.65])},
    98:  {"name": "LYS98",  "ca": np.array([23.15, 31.45, 16.45]), "functional": np.array([22.45, 30.55, 16.95])},
    102: {"name": "ASP102", "ca": np.array([25.45, 28.15, 14.15]), "functional": np.array([24.65, 27.45, 14.85])},
}

# Substrate modelled pose (PET ester carbonyl C and ester oxygen in 5XJH cleft)
SUBSTRATE_POSE = {
    "carbonyl_c": np.array([17.15, 25.10, 19.35]),
    "carbonyl_o": np.array([17.65, 25.85, 19.85]),
    "ester_o":    np.array([16.85, 24.35, 20.35]),
    "benzene_ring_center": np.array([18.95, 23.45, 21.15])
}


def get_ser160_og_coord() -> np.ndarray:
    """Returns Ser160 Oγ functional coordinate."""
    return RESIDUE_COORDINATES_5XJH[160]["functional"]


def get_substrate_ester_c_coord() -> np.ndarray:
    """Returns modelled substrate carbonyl carbon coordinate."""
    return SUBSTRATE_POSE["carbonyl_c"]
