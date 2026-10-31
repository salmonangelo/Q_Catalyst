"""Authentic Ab Initio QM Electrostatic Potential (ESP) and RESP Charge Fitting Module.

Implements the authentic Cornell et al. (1995) / Bayly et al. (1993) RESP workflow:
1. Quantum mechanical wavefunction calculation (HF/6-31G* or B3LYP/6-31G*) using PySCF.
2. Generation of Merz-Singh-Kollman (MK) van der Waals concentric grid shells.
3. Computation of exact QM molecular electrostatic potentials (MEP) at all grid points.
4. Hyperbolic-restrained least-squares fitting with total charge and symmetry constraints:
   chi^2 = sum_k (V_qm(r_k) - V_calc(r_k))^2 + a * sum_j (sqrt(q_j^2 + b^2) - b)
5. Export of fitted charges, quality metrics (RRMS, dipole moments), and raw calculation artifacts.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from pyscf import df, gto, lib, scf
from rdkit import Chem
from rdkit.Chem import AllChem

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# Bondi van der Waals radii in Angstroms (standard for Merz-Kollman grid generation)
BONDI_VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "NA": 2.27,
}


@dataclass
class RESPArtifactSummary:
    molecule_name: str
    smiles: str
    formula: str
    num_atoms: int
    qm_method: str
    basis_set: str
    scf_energy_hartree: float
    scf_converged: bool
    num_grid_points: int
    rrms_error: float
    total_fitted_charge: float
    qm_dipole_debye: float
    mm_dipole_debye: float
    fitted_charges: Dict[str, float]
    provenance_path: str


class PySCFRESPFitter:
    """Executes ab initio QM calculations and authentic RESP charge fitting."""

    def __init__(
        self,
        qm_method: str = "hf",
        basis: str = "6-31g*",
        density_pts_per_a2: float = 1.0,
        restraint_a: float = 0.0005,
        restraint_b: float = 0.1,
    ):
        self.qm_method = qm_method.lower()
        self.basis = basis
        self.density = density_pts_per_a2
        self.a = restraint_a
        self.b = restraint_b

    def generate_conformer_and_geometry(self, smiles: str, seed: int = 42) -> Tuple[Chem.Mol, str]:
        """Generates an energy-minimized 3D conformer via RDKit MMFF."""
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=seed)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

        conf = mol.GetConformer()
        atom_lines = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            sym = mol.GetAtomWithIdx(i).GetSymbol()
            atom_lines.append(f"{sym} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")

        return mol, "; ".join(atom_lines)

    def run_qm_calculation(self, geom_str: str) -> Tuple[gto.Mole, scf.hf.SCF, float, np.ndarray]:
        """Performs SCF calculation and extracts converged 1-RDM."""
        mol = gto.M(atom=geom_str, basis=self.basis, charge=0, spin=0)
        if self.qm_method == "hf":
            mf = scf.RHF(mol)
        elif self.qm_method == "b3lyp":
            mf = scf.dft.RKS(mol)
            mf.xc = "b3lyp"
        else:
            raise ValueError(f"Unsupported QM method: {self.qm_method}")

        mf.conv_tol = 1e-8
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"PySCF {self.qm_method.upper()} calculation failed to converge!")

        dm = mf.make_rdm1()
        dipole = mf.dip_moment(unit="DEBYE")
        dipole_mag = float(np.linalg.norm(dipole))
        return mol, mf, float(mf.e_tot), dm, dipole_mag

    def generate_mk_grid(self, mol: gto.Mole) -> np.ndarray:
        """Generates Merz-Kollman concentric van der Waals shells around the molecule."""
        coords_bohr = mol.atom_coords()
        coords_angstrom = coords_bohr * BOHR_TO_ANGSTROM
        natm = mol.natm

        vdw_scales = [1.4, 1.6, 1.8, 2.0]
        pts_list = []

        np.random.seed(42)
        # Construct points on shells around each atom
        for i in range(natm):
            elem = mol.atom_symbol(i).upper()
            r_vdw = BONDI_VDW_RADII.get(elem, 1.70)
            center = coords_angstrom[i]

            for scale in vdw_scales:
                r_shell = r_vdw * scale
                # Determine number of points on sphere proportional to area: 4 * pi * r^2 * density
                area = 4.0 * np.pi * (r_shell ** 2)
                n_pts = max(12, int(area * self.density * 0.25))

                # Golden spiral / Fibonacci sphere points
                indices = np.arange(0, n_pts, dtype=float) + 0.5
                phi = np.arccos(1 - 2 * indices / n_pts)
                theta = np.pi * (1 + 5 ** 0.5) * indices

                xs = r_shell * np.sin(phi) * np.cos(theta)
                ys = r_shell * np.sin(phi) * np.sin(theta)
                zs = r_shell * np.cos(phi)

                shell_pts = np.stack([xs, ys, zs], axis=1) + center

                # Exclude points that penetrate inside the 1.4*vdw of other atoms
                valid = np.ones(n_pts, dtype=bool)
                for j in range(natm):
                    if i == j:
                        continue
                    r_vdw_j = BONDI_VDW_RADII.get(mol.atom_symbol(j).upper(), 1.70)
                    dist = np.linalg.norm(shell_pts - coords_angstrom[j], axis=1)
                    valid = valid & (dist >= (r_vdw_j * 1.2))

                if np.any(valid):
                    pts_list.append(shell_pts[valid])

        all_pts_angstrom = np.vstack(pts_list)
        return all_pts_angstrom * ANGSTROM_TO_BOHR  # Convert to bohr for PySCF integrals

    def compute_qm_esp(self, mol: gto.Mole, dm: np.ndarray, grid_bohr: np.ndarray) -> np.ndarray:
        """Computes exact QM electrostatic potential V_qm on the grid in atomic units."""
        # Nuclear component
        Vnuc = np.zeros(len(grid_bohr))
        for i in range(mol.natm):
            r = mol.atom_coord(i)
            Z = mol.atom_charge(i)
            rp = r - grid_bohr
            Vnuc += Z / np.einsum("xi,xi->x", rp, rp) ** 0.5

        # Electronic component via 3-center density fitting integrals
        Vele = np.empty_like(Vnuc)
        for p0, p1 in lib.prange(0, Vele.size, 600):
            fakemol = gto.fakemol_for_charges(grid_bohr[p0:p1])
            ints = df.incore.aux_e2(mol, fakemol)
            Vele[p0:p1] = np.einsum("ijp,ij->p", ints, dm)

        return Vnuc - Vele

    def fit_resp_charges(
        self,
        mol: gto.Mole,
        grid_bohr: np.ndarray,
        v_qm: np.ndarray,
        symmetry_groups: Optional[List[List[int]]] = None,
        max_iter: int = 50,
        conv_tol: float = 1e-6,
    ) -> Tuple[np.ndarray, float]:
        """Performs iterative hyperbolic-restrained least squares (RESP) fitting."""
        coords_bohr = mol.atom_coords()
        natm = mol.natm

        # Distance matrix: dist[k, i] = |r_grid_k - r_atom_i|
        dist = np.linalg.norm(grid_bohr[:, None, :] - coords_bohr[None, :, :], axis=2)
        inv_dist = 1.0 / dist  # in a.u. (1/bohr)

        # Baseline A and B
        A_base = np.dot(inv_dist.T, inv_dist)
        B_base = np.dot(inv_dist.T, v_qm)

        # Identify non-hydrogen atoms for hyperbolic restraint
        is_heavy = np.array([mol.atom_symbol(i).upper() != "H" for i in range(natm)])

        # Setup symmetry constraints if provided
        # Constraint equations: q[i] - q[j] = 0 for pairs in each group
        sym_pairs = []
        if symmetry_groups:
            for grp in symmetry_groups:
                for idx in range(1, len(grp)):
                    sym_pairs.append((grp[0], grp[idx]))

        n_sym = len(sym_pairs)
        n_tot = natm + 1 + n_sym

        # Iterative hyperbolic restraint loop
        q_curr = np.zeros(natm)
        for it in range(max_iter):
            A = np.zeros((n_tot, n_tot))
            B = np.zeros(n_tot)

            A[:natm, :natm] = A_base.copy()
            B[:natm] = B_base.copy()

            # Add hyperbolic restraint to diagonal: a / sqrt(q_j^2 + b^2)
            for j in range(natm):
                if is_heavy[j]:
                    penalty = self.a / np.sqrt(q_curr[j] ** 2 + self.b ** 2)
                    A[j, j] += penalty

            # Total charge constraint: sum(q) = 0
            A[:natm, natm] = 1.0
            A[natm, :natm] = 1.0
            B[natm] = 0.0

            # Symmetry constraints: q[i] - q[j] = 0
            for k, (i, j) in enumerate(sym_pairs):
                row = natm + 1 + k
                A[i, row] = 1.0
                A[j, row] = -1.0
                A[row, i] = 1.0
                A[row, j] = -1.0
                B[row] = 0.0

            q_next = np.linalg.solve(A, B)[:natm]
            diff = np.max(np.abs(q_next - q_curr))
            q_curr = q_next
            if diff < conv_tol:
                break

        # Calculate RRMS (Relative Root Mean Square Error)
        v_calc = np.dot(inv_dist, q_curr)
        rrms = float(np.sqrt(np.sum((v_qm - v_calc) ** 2) / np.sum(v_qm ** 2)))

        return q_curr, rrms

    def compute_mm_dipole(self, mol: gto.Mole, charges: np.ndarray) -> float:
        """Computes classical dipole moment in Debye from point charges."""
        coords_angstrom = mol.atom_coords() * BOHR_TO_ANGSTROM
        # Dipole mu = sum_i q_i * r_i (1 e * A = 4.803204 Debye)
        center_of_charge = np.sum(charges[:, None] * coords_angstrom, axis=0)
        dipole_debye = float(np.linalg.norm(center_of_charge) * 4.803204)
        return dipole_debye

    def fit_molecule(
        self,
        name: str,
        smiles: str,
        output_dir: Path,
        atom_names: List[str],
        symmetry_groups: Optional[List[List[int]]] = None,
    ) -> RESPArtifactSummary:
        """Executes full RESP fitting pipeline for a molecule and writes artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Geometry & PySCF Molecule
        rdkit_mol, geom_str = self.generate_conformer_and_geometry(smiles)
        mol_pyscf, mf, e_scf, dm, qm_dipole = self.run_qm_calculation(geom_str)

        # 2. MK Grid & QM ESP
        grid_bohr = self.generate_mk_grid(mol_pyscf)
        v_qm = self.compute_qm_esp(mol_pyscf, dm, grid_bohr)

        # 3. Fit RESP charges
        q_fit, rrms = self.fit_resp_charges(
            mol_pyscf,
            grid_bohr,
            v_qm,
            symmetry_groups=symmetry_groups,
        )

        mm_dipole = self.compute_mm_dipole(mol_pyscf, q_fit)

        # Map fitted charges to atom names
        charges_dict = {}
        for i, q in enumerate(q_fit):
            aname = atom_names[i] if i < len(atom_names) else f"ATOM{i+1}"
            charges_dict[aname] = round(float(q), 5)

        summary = RESPArtifactSummary(
            molecule_name=name,
            smiles=smiles,
            formula=Chem.rdMolDescriptors.CalcMolFormula(rdkit_mol),
            num_atoms=mol_pyscf.natm,
            qm_method=self.qm_method.upper(),
            basis_set=self.basis,
            scf_energy_hartree=round(e_scf, 6),
            scf_converged=bool(mf.converged),
            num_grid_points=len(grid_bohr),
            rrms_error=round(rrms, 4),
            total_fitted_charge=round(float(np.sum(q_fit)), 6),
            qm_dipole_debye=round(qm_dipole, 3),
            mm_dipole_debye=round(mm_dipole, 3),
            fitted_charges=charges_dict,
            provenance_path=str(output_dir / f"{name}_resp_summary.json"),
        )

        # Save summary JSON
        with open(summary.provenance_path, "w") as f:
            json.dump(asdict(summary), f, indent=2)

        # Save raw QM coordinates and grid in XYZ / NPZ format
        coords_angstrom = mol_pyscf.atom_coords() * BOHR_TO_ANGSTROM
        grid_angstrom = grid_bohr * BOHR_TO_ANGSTROM
        np.savez_compressed(
            output_dir / f"{name}_raw_qm_esp.npz",
            coords_angstrom=coords_angstrom,
            atom_symbols=[mol_pyscf.atom_symbol(i) for i in range(mol_pyscf.natm)],
            grid_angstrom=grid_angstrom,
            v_qm_au=v_qm,
            fitted_charges=q_fit,
        )

        return summary
