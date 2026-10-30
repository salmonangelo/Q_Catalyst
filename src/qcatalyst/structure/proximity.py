"""
Structural proximity and distance calculator for mutated residues.
"""
from typing import Dict, List, Tuple
import numpy as np
from src.qcatalyst.structure.pdb_parser import (
    RESIDUE_COORDINATES_5XJH,
    get_ser160_og_coord,
    get_substrate_ester_c_coord
)


def calculate_mutation_distance(pos: int) -> float:
    """
    Computes minimum Euclidean distance from a residue position to
    either Ser160 Oγ or modelled substrate carbonyl C in 5XJH.
    """
    ser_coord = get_ser160_og_coord()
    sub_coord = get_substrate_ester_c_coord()

    if pos in RESIDUE_COORDINATES_5XJH:
        res_coord = RESIDUE_COORDINATES_5XJH[pos]["ca"]
        d_ser = np.linalg.norm(res_coord - ser_coord)
        d_sub = np.linalg.norm(res_coord - sub_coord)
        return float(min(d_ser, d_sub))
    else:
        # Fallback estimation for distant solvent-exposed residues based on primary sequence distance
        d_approx = 8.0 + 0.15 * min(abs(pos - 160), abs(pos - 237), abs(pos - 206))
        return float(d_approx)


def calculate_variant_min_distance(mutations_str: str) -> float:
    """
    Computes minimum distance among all mutations in a variant to the catalytic center.
    For WT, returns 0.0 (reference baseline).
    """
    if mutations_str == "WT":
        return 0.0

    muts = [m.strip() for m in mutations_str.split(",") if m.strip()]
    distances = []
    for m in muts:
        try:
            pos = int(m[1:-1])
            d = calculate_mutation_distance(pos)
            distances.append(d)
        except Exception:
            distances.append(15.0)

    return float(min(distances)) if distances else 15.0


def is_eligible_for_chemistry(mutations_str: str, radius_angstrom: float = 8.0) -> Tuple[bool, float]:
    """
    Evaluates whether a variant passes the structural proximity filter (R <= radius).
    """
    min_dist = calculate_variant_min_distance(mutations_str)
    # Wild type is always eligible as reference
    if mutations_str == "WT":
        return True, 0.0
    return (min_dist <= radius_angstrom), min_dist
