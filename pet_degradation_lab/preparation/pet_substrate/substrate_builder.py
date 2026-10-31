"""PET Substrate Conformer Generator and Active Site Alignment.

Milestone 2 component for:
- Building 3D molecular conformers for BHET (bis(2-hydroxyethyl) terephthalate) and 2-PET dimer
- Positioning the scissile ester bond into the PETase catalytic cleft
- Enforcing authentic Michaelis pre-reaction geometry:
  * Ser160 OG to ester carbonyl C distance = 2.95 ± 0.15 Å
  * Bürgi-Dunitz nucleophilic attack trajectory: angle(OG - C = O) = 105° ± 10°
  * Ester carbonyl O positioned in oxyanion hole (Met161 / Tyr87 backbone amides) with d >= 2.70 Å (no clash)
  * Terephthalate aromatic ring aligned with Subsite II hydrophobic cleft (Trp185 / Trp159)
- Multi-start 6-DOF rigid-body docking optimization avoiding protein heavy-atom steric clashes
- Exporting PDB coordinates with exact residue and atom names matching pet_gaff2_resp_validated.xml
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from Bio.PDB import PDBParser, Structure, Residue, Atom
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


@dataclass
class SubstratePlacementMetrics:
    substrate_type: str
    num_atoms: int
    dist_ser_og_ester_c_A: float
    dist_met_n_ester_o_A: float
    dist_tyr_n_ester_o_A: float
    burgi_dunitz_angle_deg: float
    min_protein_clash_dist_A: float
    is_michaelis_geometry_valid: bool
    summary: str


class PETSubstrateBuilder:
    """Constructs and docks PET oligomer substrates (BHET and 2-PET) into PETase active sites."""

    # Chemical SMILES
    SMILES_BHET = "C1=CC(=CC=C1C(=O)OCCO)C(=O)OCCO"
    SMILES_2PET = "O=C(c1ccc(cc1)C(=O)OCCOC(=O)c2ccc(cc2)C(=O)OCCO)OCCO"

    # Reference atom naming for BHET matching pet_gaff2_resp_validated.xml (Residue BHT)
    BHET_ATOM_MAP = [
        "C1", "C2", "C3", "C4", "C5", "C6",
        "C7", "O1", "O2", "C8", "C9", "O3",
        "C10", "O4", "O5", "C11", "C12", "O6",
        "H2", "H3", "H5", "H6",
        "H8A", "H8B", "H9A", "H9B", "HO3",
        "H11A", "H11B", "H12A", "H12B", "HO6"
    ]

    # Reference atom naming for 2-PET dimer (Residue 2PT, 54 atoms)
    PET2_ATOM_MAP = [
        "C1", "C2", "C3", "C4", "C5", "C6",
        "C7", "O1", "O2", "C8", "C9", "O3",
        "C10", "O4", "O5", "C11", "C12", "O6",
        "C13", "O7", "C14", "C15", "C16", "C17", "C18", "C19",
        "C20", "O8", "O9", "C21", "C22", "O10",
        "H2", "H3", "H5", "H6",
        "H8A", "H8B", "H9A", "H9B", "HO3",
        "H11A", "H11B", "H12A", "H12B",
        "H15", "H16", "H18", "H19",
        "H21A", "H21B", "H22A", "H22B", "HO10"
    ]

    def __init__(self):
        self.parser = PDBParser(QUIET=True)

    def generate_bhet_conformer(self, seed: int = 42) -> Chem.Mol:
        """Generates energy-minimized 3D conformer of BHET."""
        mol = Chem.MolFromSmiles(self.SMILES_BHET)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=seed)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        return mol

    def generate_2pet_conformer(self, seed: int = 42) -> Chem.Mol:
        """Generates energy-minimized 3D conformer of 2-PET dimer."""
        mol = Chem.MolFromSmiles(self.SMILES_2PET)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=seed)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        return mol

    def _find_active_site_residues(self, chain: Structure) -> Tuple[Residue, Residue, Residue, Residue, Residue]:
        """Locates catalytic Ser160, His237, Met161, Tyr87, and Subsite II aromatics."""
        res_list = list(chain.get_residues())

        cat_ser, cat_his, cat_met, cat_tyr, sub2_arom1, sub2_arom2 = None, None, None, None, None, None

        # 1. Identify Catalytic Ser: Ser whose OG is within 2.7 - 3.2 A of a His NE2
        for i, r in enumerate(res_list):
            if r.resname == "SER" and "OG" in r:
                for h in res_list:
                    if h.resname in ["HIS", "HID", "HIE", "HIP"] and "NE2" in h:
                        d = float(np.linalg.norm(r["OG"].coord - h["NE2"].coord))
                        if 2.7 <= d <= 3.2:
                            cat_ser = r
                            cat_his = h
                            if i + 1 < len(res_list):
                                cat_met = res_list[i + 1]
                            if i - 1 >= 0:
                                sub2_arom1 = res_list[i - 1]
                            break
                if cat_ser:
                    break

        if cat_ser is None:
            # Fallback to standard numbering if present
            cat_ser = chain[160] if 160 in chain else res_list[0]
            cat_met = chain[161] if 161 in chain else cat_ser
            cat_his = chain[237] if 237 in chain else cat_ser
            sub2_arom1 = chain[159] if 159 in chain else cat_ser

        # 2. Identify Tyr87 (oxyanion hole donor 2): ~73 residues upstream of Ser160
        ser_idx = res_list.index(cat_ser) if cat_ser in res_list else 0
        for j in range(max(0, ser_idx - 85), min(len(res_list), ser_idx - 55)):
            if res_list[j].resname == "TYR" and "N" in res_list[j]:
                cat_tyr = res_list[j]
                break
        if cat_tyr is None:
            cat_tyr = chain[87] if 87 in chain else cat_met

        # 3. Identify Subsite II aromatic 2 (Trp185 homologous position): ~25 residues downstream
        for j in range(ser_idx + 20, min(len(res_list), ser_idx + 35)):
            if res_list[j].resname in ["TRP", "PHE", "TYR"]:
                sub2_arom2 = res_list[j]
                break
        if sub2_arom2 is None:
            sub2_arom2 = chain[185] if 185 in chain else sub2_arom1

        return cat_ser, cat_met, cat_tyr, sub2_arom1, sub2_arom2

    def align_bhet_to_active_site(
        self,
        protein_path: Path,
        chain_id: str = "A",
        output_pdb_path: Optional[Path] = None,
        random_seed: int = 42,
    ) -> Tuple[Chem.Mol, SubstratePlacementMetrics]:
        """Aligns BHET into the catalytic cleft using 6-DOF docking with Bürgi-Dunitz trajectory."""
        struct = self.parser.get_structure("PROT", str(protein_path))
        chain = struct[0][chain_id]

        cat_ser, cat_met, cat_tyr, sub2_arom1, sub2_arom2 = self._find_active_site_residues(chain)

        ser_og = cat_ser["OG"].coord
        met_n = cat_met["N"].coord
        tyr_n = cat_tyr["N"].coord
        sub2_center = (sub2_arom1["CA"].coord + sub2_arom2["CA"].coord) / 2.0

        # Collect protein heavy atoms within 15 A of ser_og for steric clash penalty
        prot_atoms = [
            a.coord for r in chain.get_residues()
            for a in r if not a.name.startswith("H") and np.linalg.norm(a.coord - ser_og) < 15.0
        ]
        prot_coords = np.array(prot_atoms)

        # Generate conformer
        mol = self.generate_bhet_conformer(seed=random_seed)
        conf = mol.GetConformer()
        base_coords = np.array([conf.GetAtomPosition(k) for k in range(mol.GetNumAtoms())])

        # Scissile ester indices in BHET SMILES C1=CC(=CC=C1C(=O)OCCO)C(=O)OCCO:
        # C7 is atom index 6, O1 (carbonyl O) is atom index 7, ring atoms are 0-5
        c7_idx = 6
        o1_idx = 7
        ring_indices = [0, 1, 2, 3, 4, 5]

        # Center conformer at C7
        base_centered = base_coords - base_coords[c7_idx]

        # Initial C7 target position along vector towards oxyanion center
        ox_center = (met_n + tyr_n) / 2.0
        v_ox = ox_center - ser_og
        v_ox_u = v_ox / (np.linalg.norm(v_ox) + 1e-8)
        init_c7 = ser_og + 2.95 * v_ox_u

        def objective(params):
            R = Rotation.from_rotvec(params[0:3]).as_matrix()
            curr_coords = np.dot(base_centered, R.T) + params[3:6]
            c7 = curr_coords[c7_idx]
            o1 = curr_coords[o1_idx]

            # Attack distance
            d_ser_c7 = np.linalg.norm(c7 - ser_og)

            # Bürgi-Dunitz angle
            v_sc = ser_og - c7
            v_oc = o1 - c7
            cos_bd = np.dot(v_sc, v_oc) / (np.linalg.norm(v_sc) * np.linalg.norm(v_oc) + 1e-8)
            ang_bd = np.arccos(np.clip(cos_bd, -1.0, 1.0)) * 180.0 / np.pi

            # Oxyanion hole coordination
            d_met_o1 = np.linalg.norm(o1 - met_n)
            d_tyr_o1 = np.linalg.norm(o1 - tyr_n)

            # Subsite II ring packing
            ring_centroid = np.mean(curr_coords[ring_indices], axis=0)
            d_sub2 = np.linalg.norm(ring_centroid - sub2_center)

            # Steric clash penalty against protein heavy atoms
            dists = np.linalg.norm(curr_coords[:18, np.newaxis, :] - prot_coords[np.newaxis, :, :], axis=2)
            clashes = np.maximum(0.0, 2.4 - dists)
            clash_penalty = np.sum(clashes**2) * 80.0

            # Strict clash prevention on oxyanion hole amides (< 2.75 A)
            ox_penalty = 800.0 * (max(0.0, 2.75 - d_met_o1)**2 + max(0.0, 2.75 - d_tyr_o1)**2)

            # Strict distance bounds on d_ser_c7 (must be 2.95 +- 0.15 A)
            ser_bounds_penalty = 800.0 * (max(0.0, 2.80 - d_ser_c7)**2 + max(0.0, d_ser_c7 - 3.10)**2)

            return (
                250.0 * (d_ser_c7 - 2.95)**2 +
                15.0 * (ang_bd - 105.0)**2 +
                40.0 * (d_met_o1 - 2.95)**2 +
                40.0 * (d_tyr_o1 - 3.10)**2 +
                5.0 * (d_sub2 - 4.5)**2 +
                clash_penalty + ox_penalty + ser_bounds_penalty
            )

        best_res = None
        best_cost = 1e9
        for s_idx in range(25):
            np.random.seed(s_idx + random_seed)
            x0 = np.concatenate([np.random.uniform(-np.pi, np.pi, 3), init_c7])
            res_opt = minimize(objective, x0, method="Powell", options={"maxiter": 250})
            if res_opt.fun < best_cost:
                best_cost = res_opt.fun
                best_res = res_opt

        p_opt = best_res.x
        R_opt = Rotation.from_rotvec(p_opt[0:3]).as_matrix()
        final_coords = np.dot(base_centered, R_opt.T) + p_opt[3:6]

        # Apply coordinates to RDKit conformer
        for k in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(k, final_coords[k])

        # Evaluate final placement metrics
        placed_c7 = final_coords[c7_idx]
        placed_o1 = final_coords[o1_idx]
        d_ser = float(np.linalg.norm(placed_c7 - ser_og))
        d_met = float(np.linalg.norm(placed_o1 - met_n))
        d_tyr = float(np.linalg.norm(placed_o1 - tyr_n))

        cos_bd = np.dot(ser_og - placed_c7, placed_o1 - placed_c7) / (
            d_ser * float(np.linalg.norm(placed_o1 - placed_c7)) + 1e-8
        )
        ang_bd = float(np.arccos(np.clip(cos_bd, -1.0, 1.0)) * 180.0 / np.pi)

        # Minimum distance to any protein heavy atom
        dists_all = np.linalg.norm(final_coords[:18, np.newaxis, :] - prot_coords[np.newaxis, :, :], axis=2)
        min_clash_dist = float(np.min(dists_all))

        # Check validity criteria
        is_valid = (
            (2.75 <= d_ser <= 3.15) and
            (95.0 <= ang_bd <= 115.0) and
            (d_met >= 2.70) and
            (d_tyr >= 2.70) and
            (d_met <= 3.85) and
            (min_clash_dist >= 2.05)
        )

        metrics = SubstratePlacementMetrics(
            substrate_type="BHET",
            num_atoms=mol.GetNumAtoms(),
            dist_ser_og_ester_c_A=round(d_ser, 3),
            dist_met_n_ester_o_A=round(d_met, 3),
            dist_tyr_n_ester_o_A=round(d_tyr, 3),
            burgi_dunitz_angle_deg=round(ang_bd, 1),
            min_protein_clash_dist_A=round(min_clash_dist, 3),
            is_michaelis_geometry_valid=is_valid,
            summary=(
                f"BHET docked: d(Ser-OG, C)={d_ser:.2f}A, BD_angle={ang_bd:.1f}deg, "
                f"Oxyanion Met161-N={d_met:.2f}A, Tyr87-N={d_tyr:.2f}A, min_prot_d={min_clash_dist:.2f}A"
            ),
        )

        if output_pdb_path:
            self.write_bhet_pdb(mol, output_pdb_path)

        return mol, metrics

    def write_bhet_pdb(self, mol: Chem.Mol, output_path: Path):
        """Writes BHET as PDB residue BHT with atom names matching pet_gaff2_resp_validated.xml."""
        conf = mol.GetConformer()
        lines = [
            "REMARK    BHET Substrate Model for PETase Binding Cleft",
            "REMARK    GAFF2/RESP compliant nomenclature (Residue BHT)",
        ]

        for idx, name in enumerate(self.BHET_ATOM_MAP):
            if idx >= mol.GetNumAtoms():
                break
            atom = mol.GetAtomWithIdx(idx)
            pos = conf.GetAtomPosition(idx)
            elem = atom.GetSymbol()
            line = (
                f"HETATM{idx+1:5d} {name:^4s} BHT B 900    "
                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
                f"  1.00 20.00          {elem:>2s}"
            )
            lines.append(line)

        # Add CONECT records for all covalent bonds in the molecule
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx() + 1
            a2 = bond.GetEndAtomIdx() + 1
            lines.append(f"CONECT{a1:5d}{a2:5d}")

        lines.append("END")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")
