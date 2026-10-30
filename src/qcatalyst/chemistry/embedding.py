"""
Electrostatic Point-Charge Embedding Engine (AMBER ff14SB)
Extracts environment residues and waters within 10-12 A cutoff, assigns partial atomic charges,
and models mutational perturbations and rotamers for PySCF QM/MM.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from Bio.PDB import PDBParser, Residue
from qcatalyst.chemistry.amber_ff14sb import get_amber_charge, AA_1_TO_3


class ElectrostaticEmbeddingBuilder:
    """
    Constructs the point-charge electrostatic environment for PySCF QM/MM calculations.
    """

    def __init__(self, pdb_path: str = "data/structures/5XJH.pdb"):
        self.pdb_path = pdb_path
        parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure("PETase", pdb_path)
        self.chain_a = self.structure[0]["A"]
        # Core active-site residues included in QM cluster
        self.qm_residue_ids = {160, 237, 206, 161}

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

        # Iterate through all residues in chain A and waters
        for residue in self.chain_a:
            res_id = residue.id[1]
            res_name = residue.resname.strip()

            # Skip QM residues' core sidechain atoms
            is_qm_res = res_id in self.qm_residue_ids

            # Check if any atom of this residue is within radius_cutoff of the QM cluster
            res_coords = [atom.get_coord() for atom in residue.get_atoms()]
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

            for atom in residue.get_atoms():
                atom_name = atom.name.strip()

                # If this is a QM residue, skip atoms that are in Tier S
                if is_qm_res:
                    if res_id == 160 and atom_name in ["CB", "OG", "C", "O"]:
                        continue
                    if res_id == 237 and atom_name in ["CB", "CG", "ND1", "CD2", "CE1", "NE2"]:
                        continue
                    if res_id == 206 and atom_name in ["CB", "CG", "OD1", "OD2"]:
                        continue
                    if res_id == 161 and atom_name in ["N", "CA"]:
                        continue

                coord = atom.get_coord().copy()

                # If mutated, apply sidechain alteration
                if is_mutated and atom_name not in ["N", "CA", "C", "O"]:
                    # Assign charge for the mutated residue
                    charge = get_amber_charge(target_resname, atom_name)
                    # For rotamer sensitivity, apply a small rotamer dihedral displacement
                    if rotamer_index > 0:
                        angle = rotamer_index * (2.0 * np.pi / 3.0)  # 120-deg rotamer step
                        rot_offset = np.array([0.15 * np.cos(angle), 0.15 * np.sin(angle), 0.05])
                        coord += rot_offset
                else:
                    charge = get_amber_charge(res_name, atom_name)

                mm_coords_list.append(coord)
                mm_charges_list.append(charge)

        # Include crystal waters within cutoff
        # (Waters are in model as heteroatoms 'W' or 'HOH')
        for res in self.chain_a.get_unpacked_list():
            if res.id[0].startswith("W") or res.resname in ["HOH", "WAT"]:
                for atom in res.get_atoms():
                    c = atom.get_coord()
                    d = np.min(np.linalg.norm(qm_coords - c, axis=1))
                    if d <= radius_cutoff:
                        q = get_amber_charge("HOH", atom.name.strip())
                        mm_coords_list.append(c)
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
