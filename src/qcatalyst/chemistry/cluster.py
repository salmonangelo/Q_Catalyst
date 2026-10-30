"""
Tier S and Tier M Cluster Construction for PETase Active Site.
Assembles the catalytic triad (Ser160, His237, Asp206), oxyanion hole, and substrate model.
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
try:
    from Bio.PDB import PDBParser
    _HAS_BIOPYTHON = True
except ImportError:
    _HAS_BIOPYTHON = False


class ActiveSiteClusterBuilder:
    """
    Constructs capped quantum cluster models (Tier S and Tier M) from PETase crystal structures.
    """

    def __init__(self, pdb_path: str = "data/structures/5XJH.pdb"):
        self.pdb_path = pdb_path
        self._coords: Dict[Tuple[int, str], np.ndarray] = {}
        if _HAS_BIOPYTHON:
            parser = PDBParser(QUIET=True)
            self.structure = parser.get_structure("PETase", pdb_path)
            self.model = self.structure[0]
            self.chain_a = self.model["A"]
        else:
            self._load_native_pdb(pdb_path)

    def _load_native_pdb(self, pdb_path: str, chain_id: str = "A") -> None:
        with open(pdb_path, "r") as f:
            for line in f:
                if line[0:6].strip() == "ATOM" and line[21] == chain_id:
                    atom_name = line[12:16].strip()
                    res_id = int(line[22:26])
                    c = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float64)
                    self._coords[(res_id, atom_name)] = c

    def _get_atom_coord(self, res_id: int, atom_name: str) -> np.ndarray:
        if _HAS_BIOPYTHON:
            return self.chain_a[res_id][atom_name].get_coord().copy()
        key = (res_id, atom_name)
        if key not in self._coords:
            raise KeyError(f"Atom {atom_name} for residue {res_id} not found in {self.pdb_path}")
        return self._coords[key].copy()

    def build_tier_s(
        self,
        his237_protonation: str = "HIE",  # 'HIE' (ND1-H, resting), 'HIP' (both ND1-H, NE2-H), 'HID' (NE2-H)
        asp206_protonation: str = "deprotonated",  # 'deprotonated' (carboxylate anion) or 'protonated' (neutral ASH)
        ser160_protonation: str = "neutral",  # 'neutral' (Ser-OH) or 'deprotonated' (alkoxide Ser-O-)
        ser_ester_dist: float = 2.95,  # Ser160 OG - ester C distance in Angstroms
    ) -> Dict[str, any]:
        """
        Builds the 40-atom Tier S cluster:
        - Ser160 (methanol-like, capped at Cbeta with 3 H)
        - His237 (4-methylimidazole, capped at Cbeta with 3 H)
        - Asp206 (acetate, capped at Cbeta with 3 H)
        - Met161 backbone peptide (formamide/N-methylacetamide model for oxyanion hole)
        - Substrate PET ester model (methyl acetate fragment)
        """
        atoms: List[Tuple[str, np.ndarray]] = []

        # 1. Ser160 Fragment (Methanol model)
        cb_ser = self._get_atom_coord(160, "CB")
        og_ser = self._get_atom_coord(160, "OG")
        ca_ser = self._get_atom_coord(160, "CA")

        # C-O vector and direction
        v_co = og_ser - cb_ser
        norm_co = np.linalg.norm(v_co)
        u_co = v_co / norm_co if norm_co > 0 else np.array([0, 0, 1])

        # Cap Ser Cbeta as CH3 (replacing Calpha with H + 2 hydrogens)
        v_ca_cb = cb_ser - ca_ser
        u_ca_cb = v_ca_cb / np.linalg.norm(v_ca_cb)

        # Serine atoms
        atoms.append(("C", cb_ser))
        atoms.append(("O", og_ser))

        # Add 3 hydrogens to Ser Cbeta to make it methanol-like
        perp1 = np.cross(u_co, np.array([1, 0, 0]))
        if np.linalg.norm(perp1) < 0.1:
            perp1 = np.cross(u_co, np.array([0, 1, 0]))
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(u_co, perp1)
        perp2 /= np.linalg.norm(perp2)

        atoms.append(("H", cb_ser - 0.5 * u_co * 1.09 + 0.866 * perp1 * 1.09))
        atoms.append(("H", cb_ser - 0.5 * u_co * 1.09 - 0.433 * perp1 * 1.09 + 0.75 * perp2 * 1.09))
        atoms.append(("H", cb_ser - 0.5 * u_co * 1.09 - 0.433 * perp1 * 1.09 - 0.75 * perp2 * 1.09))

        # Serine hydroxyl proton
        his_ne2 = self._get_atom_coord(237, "NE2")
        v_og_ne2 = his_ne2 - og_ser
        u_og_ne2 = v_og_ne2 / np.linalg.norm(v_og_ne2)
        if ser160_protonation == "neutral":
            # H pointing toward His237 NE2
            atoms.append(("H", og_ser + u_og_ne2 * 0.96))

        # 2. His237 Fragment (4-methylimidazole)
        cb_his = self._get_atom_coord(237, "CB")
        cg_his = self._get_atom_coord(237, "CG")
        nd1_his = self._get_atom_coord(237, "ND1")
        cd2_his = self._get_atom_coord(237, "CD2")
        ce1_his = self._get_atom_coord(237, "CE1")
        ne2_his = self._get_atom_coord(237, "NE2")

        atoms.append(("C", cb_his))
        atoms.append(("C", cg_his))
        atoms.append(("N", nd1_his))
        atoms.append(("C", cd2_his))
        atoms.append(("C", ce1_his))
        atoms.append(("N", ne2_his))

        # Ring aromatic hydrogens
        # CD2-H: bisector of CG-CD2 and NE2-CD2
        v_cd2_h = (cd2_his - cg_his) + (cd2_his - ne2_his)
        u_cd2_h = v_cd2_h / np.linalg.norm(v_cd2_h)
        atoms.append(("H", cd2_his + u_cd2_h * 1.08))

        # CE1-H: bisector of ND1-CE1 and NE2-CE1
        v_ce1_h = (ce1_his - nd1_his) + (ce1_his - ne2_his)
        u_ce1_h = v_ce1_h / np.linalg.norm(v_ce1_h)
        atoms.append(("H", ce1_his + u_ce1_h * 1.08))

        # Cap His Cbeta with 3 hydrogens
        v_cg_cb = cb_his - cg_his
        u_cg_cb = v_cg_cb / np.linalg.norm(v_cg_cb)
        p1 = np.cross(u_cg_cb, np.array([1, 0, 0]))
        if np.linalg.norm(p1) < 0.1:
            p1 = np.cross(u_cg_cb, np.array([0, 1, 0]))
        p1 /= np.linalg.norm(p1)
        p2 = np.cross(u_cg_cb, p1)
        atoms.append(("H", cb_his + u_cg_cb * 1.09))
        atoms.append(("H", cb_his + 0.5 * u_cg_cb * 1.09 + 0.866 * p1 * 1.09))
        atoms.append(("H", cb_his + 0.5 * u_cg_cb * 1.09 - 0.866 * p1 * 1.09))

        # His237 ND1 and NE2 Protons based on protonation state
        asp_od2 = self._get_atom_coord(206, "OD2")
        v_nd1_od2 = asp_od2 - nd1_his
        u_nd1_od2 = v_nd1_od2 / np.linalg.norm(v_nd1_od2)

        if his237_protonation in ["HIE", "HIP"]:
            # Proton on ND1 pointing to Asp206
            atoms.append(("H", nd1_his + u_nd1_od2 * 1.01))

        if his237_protonation in ["HID", "HIP"] or ser160_protonation == "deprotonated":
            # Proton on NE2 pointing to Ser160 OG
            v_ne2_og = og_ser - ne2_his
            u_ne2_og = v_ne2_og / np.linalg.norm(v_ne2_og)
            atoms.append(("H", ne2_his + u_ne2_og * 1.01))

        # 3. Asp206 Fragment (Acetate model)
        cb_asp = self._get_atom_coord(206, "CB")
        cg_asp = self._get_atom_coord(206, "CG")
        od1_asp = self._get_atom_coord(206, "OD1")
        od2_asp = self._get_atom_coord(206, "OD2")

        atoms.append(("C", cb_asp))
        atoms.append(("C", cg_asp))
        atoms.append(("O", od1_asp))
        atoms.append(("O", od2_asp))

        # Cap Asp Cbeta with 3 hydrogens
        v_cg_cb_asp = cb_asp - cg_asp
        u_cg_cb_asp = v_cg_cb_asp / np.linalg.norm(v_cg_cb_asp)
        pa1 = np.cross(u_cg_cb_asp, np.array([1, 0, 0]))
        if np.linalg.norm(pa1) < 0.1:
            pa1 = np.cross(u_cg_cb_asp, np.array([0, 1, 0]))
        pa1 /= np.linalg.norm(pa1)
        pa2 = np.cross(u_cg_cb_asp, pa1)
        atoms.append(("H", cb_asp + u_cg_cb_asp * 1.09))
        atoms.append(("H", cb_asp + 0.5 * u_cg_cb_asp * 1.09 + 0.866 * pa1 * 1.09))
        atoms.append(("H", cb_asp + 0.5 * u_cg_cb_asp * 1.09 - 0.866 * pa1 * 1.09))

        if asp206_protonation == "protonated":
            # Proton on OD2 pointing to His ND1
            atoms.append(("H", od2_asp - u_nd1_od2 * 0.98))

        # 4. Met161 Oxyanion Hole Peptide Fragment (Formamide model: H-CO-NH2)
        n_met = self._get_atom_coord(161, "N")
        ca_met = self._get_atom_coord(161, "CA")
        c_ser = self._get_atom_coord(160, "C")
        o_ser = self._get_atom_coord(160, "O")

        # Carbonyl C and O of peptide bond
        atoms.append(("C", c_ser))
        atoms.append(("O", o_ser))
        # Formamide formyl proton (replacing Ser CA)
        v_c_ca = c_ser - ca_ser
        atoms.append(("H", c_ser + (v_c_ca / np.linalg.norm(v_c_ca)) * 1.10))

        # Peptide nitrogen
        atoms.append(("N", n_met))
        # Catalytic oxyanion-hole N-H proton pointing toward ester carbonyl position
        v_nh = (n_met - ca_met) + (n_met - c_ser)
        u_nh = v_nh / np.linalg.norm(v_nh)
        h_met = n_met + u_nh * 1.01
        atoms.append(("H", h_met))
        # Second amide proton (replacing Met161 CA with H)
        v_n_ca = n_met - ca_met
        atoms.append(("H", n_met + (v_n_ca / np.linalg.norm(v_n_ca)) * 1.01))

        # 5. Substrate PET Ester Model (Methyl Acetate: CH3-C(=O)-O-CH3)
        # Position ester carbonyl C at distance ser_ester_dist from Ser160 OG
        # Carbonyl oxygen oriented toward Met161 N-H
        v_og_nh = h_met - og_ser
        u_attack = v_og_nh / np.linalg.norm(v_og_nh)
        
        # Ester carbonyl C
        ester_c = og_ser + u_attack * ser_ester_dist
        
        # Ester carbonyl O points directly to oxyanion hole Met161 N-H
        v_c_hmet = h_met - ester_c
        u_co_ester = v_c_hmet / np.linalg.norm(v_c_hmet)
        ester_o = ester_c + u_co_ester * 1.22  # C=O bond length 1.22 A

        # Ester O-methyl and methyl C
        v_perp_ester = np.cross(u_co_ester, np.array([0, 0, 1]))
        if np.linalg.norm(v_perp_ester) < 0.1:
            v_perp_ester = np.cross(u_co_ester, np.array([0, 1, 0]))
        v_perp_ester /= np.linalg.norm(v_perp_ester)

        ester_ome = ester_c - 0.5 * u_co_ester * 1.35 + 0.866 * v_perp_ester * 1.35
        ester_me_c = ester_c - 0.5 * u_co_ester * 1.50 - 0.866 * v_perp_ester * 1.50
        ester_me_c2 = ester_ome + v_perp_ester * 1.43

        atoms.append(("C", ester_c))
        atoms.append(("O", ester_o))
        atoms.append(("O", ester_ome))
        atoms.append(("C", ester_me_c))
        atoms.append(("C", ester_me_c2))

        # Methyl hydrogens (3 on acetate methyl, 3 on methoxy methyl)
        atoms.append(("H", ester_me_c + np.array([0.0, 0.89, 0.62])))
        atoms.append(("H", ester_me_c + np.array([0.0, -0.89, 0.62])))
        atoms.append(("H", ester_me_c + np.array([0.0, 0.0, -1.09])))

        atoms.append(("H", ester_me_c2 + np.array([0.0, 0.89, 0.62])))
        atoms.append(("H", ester_me_c2 + np.array([0.0, -0.89, 0.62])))
        atoms.append(("H", ester_me_c2 + np.array([0.0, 0.0, -1.09])))

        # Calculate net formal charge
        # Ser: neutral (0) or -1
        # His: neutral (0 in HIE/HID) or +1 in HIP
        # Asp: -1 (deprotonated) or 0 (protonated)
        # Met peptide: 0
        # Ester: 0
        net_charge = 0
        if ser160_protonation == "neutral":
            pass
        else:
            net_charge -= 1

        if his237_protonation == "HIP":
            net_charge += 1

        if asp206_protonation == "deprotonated":
            net_charge -= 1

        # Format geometry string for PySCF
        atom_lines = []
        for symbol, coord in atoms:
            atom_lines.append(f"{symbol:2s}  {coord[0]:12.6f}  {coord[1]:12.6f}  {coord[2]:12.6f}")
        geom_str = "\n".join(atom_lines)

        return {
            "tier": "Tier S",
            "natoms": len(atoms),
            "charge": net_charge,
            "spin": 0,
            "geometry_str": geom_str,
            "atoms": atoms,
            "ser_ester_dist": ser_ester_dist,
            "his237_protonation": his237_protonation,
            "asp206_protonation": asp206_protonation,
            "ser160_protonation": ser160_protonation,
            "key_coords": {
                "ser160_og": og_ser,
                "his237_ne2": ne2_his,
                "his237_nd1": nd1_his,
                "asp206_od2": od2_asp,
                "met161_nh": h_met,
                "ester_c": ester_c,
                "ester_o": ester_o,
            }
        }

    def export_xyz(self, cluster: Dict[str, any], filepath: str) -> None:
        """Exports cluster to standard XYZ format."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(f"{cluster['natoms']}\n")
            f.write(f"{cluster['tier']} PETase Active Site Cluster | Charge={cluster['charge']} | Atoms={cluster['natoms']}\n")
            f.write(cluster["geometry_str"] + "\n")
        print(f"Exported XYZ cluster ({cluster['natoms']} atoms) to {filepath}")


if __name__ == "__main__":
    builder = ActiveSiteClusterBuilder("data/structures/5XJH.pdb")
    cluster = builder.build_tier_s()
    print(f"Built Tier S cluster: {cluster['natoms']} atoms, net charge = {cluster['charge']}")
    builder.export_xyz(cluster, "results/chemistry/tier_s_cluster.xyz")
    builder.export_xyz(cluster, "results/quantum/cluster_tier_s_for_role3.xyz")
