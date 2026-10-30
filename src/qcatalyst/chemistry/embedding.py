"""
Electrostatic Point-Charge Embedding Engine (AMBER ff14SB)
Extracts environment residues and waters within 10-12 A cutoff, assigns partial atomic charges,
and models mutational perturbations and rotamers for PySCF QM/MM.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
try:
    from Bio.PDB import PDBParser, Residue
    _HAS_BIOPYTHON = True
except ImportError:
    _HAS_BIOPYTHON = False
from qcatalyst.chemistry.amber_ff14sb import get_amber_charge, AA_1_TO_3


class ElectrostaticEmbeddingBuilder:
    """
    Constructs the point-charge electrostatic environment for PySCF QM/MM calculations.
    """

    def __init__(self, pdb_path: str = "data/structures/5XJH.pdb"):
        self.pdb_path = pdb_path
        self._native_residues = []
        self._native_waters = []
        if _HAS_BIOPYTHON:
            parser = PDBParser(QUIET=True)
            self.structure = parser.get_structure("PETase", pdb_path)
            self.chain_a = self.structure[0]["A"]
        else:
            self._parse_native_pdb(pdb_path)
        # Core active-site residues included in QM cluster
        self.qm_residue_ids = {160, 237, 206, 161}

    def _parse_native_pdb(self, pdb_path: str, chain_id: str = "A") -> None:
        residues_dict = {}
        with open(pdb_path, "r") as f:
            for line in f:
                rec = line[0:6].strip()
                if rec == "ATOM" and line[21] == chain_id:
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    res_id = int(line[22:26])
                    c = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float64)
                    if res_id not in residues_dict:
                        residues_dict[res_id] = (res_name, [])
                    residues_dict[res_id][1].append((atom_name, c))
                elif rec == "HETATM":
                    res_name = line[17:20].strip()
                    if res_name in ["HOH", "WAT"]:
                        atom_name = line[12:16].strip()
                        c = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float64)
                        self._native_waters.append((res_name, atom_name, c))
        for res_id, (res_name, atoms) in sorted(residues_dict.items()):
            self._native_residues.append((res_id, res_name, atoms))

    def _iter_residues(self):
        if _HAS_BIOPYTHON:
            for residue in self.chain_a:
                res_id = residue.id[1]
                res_name = residue.resname.strip()
                atoms = [(atom.name.strip(), atom.get_coord().copy()) for atom in residue.get_atoms()]
                yield res_id, res_name, atoms
        else:
            for res_id, res_name, atoms in self._native_residues:
                yield res_id, res_name, [(name, c.copy()) for name, c in atoms]

    def _iter_waters(self):
        if _HAS_BIOPYTHON:
            for res in self.chain_a.get_unpacked_list():
                if res.id[0].startswith("W") or res.resname in ["HOH", "WAT"]:
                    for atom in res.get_atoms():
                        yield res.resname.strip(), atom.name.strip(), atom.get_coord().copy()
        else:
            for res_name, atom_name, coord in self._native_waters:
                yield res_name, atom_name, coord.copy()

    def get_environment_charges(
        self,
        qm_coords: np.ndarray,
        radius_cutoff: float = 10.0,
        mutation: Optional[str] = None,  # e.g. "S121E" or "D186H" or "WT"
        rotamer_index: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
        """
        Builds the MM point charge array (coords, charges) within radius_cutoff of QM cluster.
        Handles wild type and single/multiple mutations.
        """
        mm_coords_list = []
        mm_charges_list = []
        included_residues = []

        # Parse mutation if specified (e.g. "S121E" -> pos=121, wt="S", mut="E")
        mut_dict = {}
        if mutation and mutation not in ["WT", "None", ""]:
            for mut_str in mutation.split("/"):
                mut_str = mut_str.strip()
                if len(mut_str) >= 3 and mut_str[0].isalpha() and mut_str[-1].isalpha():
                    wt_1 = mut_str[0]
                    mut_1 = mut_str[-1]
                    pos = int(mut_str[1:-1])
                    mut_dict[pos] = (wt_1, mut_1)

        # Iterate through all residues in chain A
        for res_id, res_name, atoms in self._iter_residues():
            # Material boundary fix (Task 3): Exclude ALL atoms belonging to QM residues
            # (Ser160, His237, Asp206, Met161) to eliminate severe Coulombic clashing
            # between classical point charges and quantum capping hydrogens (e.g. His237 CA at 0.43 A).
            if res_id in self.qm_residue_ids:
                continue

            res_coords = [c for _, c in atoms]
            if not res_coords:
                continue
            
            # Compute minimum distance to QM cluster
            min_dist = np.min([
                np.min(np.linalg.norm(qm_coords - rc, axis=1)) for rc in res_coords
            ])

            if min_dist > radius_cutoff:
                continue

            included_residues.append((res_id, res_name, round(float(min_dist), 2)))

            # Check if this residue is mutated
            is_mutated = res_id in mut_dict
            target_resname = res_name
            if is_mutated:
                _, mut_code_1 = mut_dict[res_id]
                target_resname = AA_1_TO_3.get(mut_code_1, res_name)

            for atom_name, coord in atoms:
                # If mutated, apply sidechain alteration
                if is_mutated and atom_name not in ["N", "CA", "C", "O"]:
                    charge = get_amber_charge(target_resname, atom_name)
                    if rotamer_index > 0:
                        angle = rotamer_index * (2.0 * np.pi / 3.0)
                        rot_offset = np.array([0.15 * np.cos(angle), 0.15 * np.sin(angle), 0.05])
                        coord += rot_offset
                else:
                    charge = get_amber_charge(res_name, atom_name)

                mm_coords_list.append(coord)
                mm_charges_list.append(charge)

        # Include crystal waters within cutoff
        for w_resname, w_atomname, w_coord in self._iter_waters():
            d = np.min(np.linalg.norm(qm_coords - w_coord, axis=1))
            if d <= radius_cutoff:
                q = get_amber_charge("HOH", w_atomname)
                mm_coords_list.append(w_coord)
                mm_charges_list.append(q)

        coords_arr = np.array(mm_coords_list, dtype=np.float64)
        charges_arr = np.array(mm_charges_list, dtype=np.float64)

        metadata = {
            "radius_cutoff_A": radius_cutoff,
            "mutation": mutation or "WT",
            "rotamer_index": rotamer_index,
            "n_point_charges": len(charges_arr),
            "net_mm_charge": round(float(np.sum(charges_arr)), 3),
            "included_residue_count": len(included_residues),
            "included_residues": included_residues[:10],  # sample
        }

        return coords_arr, charges_arr, metadata


if __name__ == "__main__":
    from qcatalyst.chemistry.cluster import ActiveSiteClusterBuilder
    builder = ActiveSiteClusterBuilder("data/structures/5XJH.pdb")
    tier_s = builder.build_tier_s()
    qm_coords = np.array([coord for _, coord in tier_s["atoms"]])

    embed_builder = ElectrostaticEmbeddingBuilder("data/structures/5XJH.pdb")
    
    # WT embedding
    coords_wt, charges_wt, meta_wt = embed_builder.get_environment_charges(qm_coords, radius_cutoff=10.0, mutation="WT")
    print(f"WT Point Charges (R=10 A): {len(charges_wt)} atoms, Net MM Charge = {meta_wt['net_mm_charge']} e")

    # S121E mutation embedding
    coords_mut, charges_mut, meta_mut = embed_builder.get_environment_charges(qm_coords, radius_cutoff=10.0, mutation="S121E")
    print(f"S121E Point Charges (R=10 A): {len(charges_mut)} atoms, Net MM Charge = {meta_mut['net_mm_charge']} e")
