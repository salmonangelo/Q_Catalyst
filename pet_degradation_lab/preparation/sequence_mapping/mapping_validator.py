"""Sequence-to-Structure Mapping and Catalytic Site Validator.

Milestone 2 component for verifying:
- Chain identifiers
- Residue numbering and sequence alignment
- Catalytic triad identities and spatial metrics (Ser160, His237, Asp206)
- Disulfide bond integrity (Cys203-Cys239 and Cys273-Cys289)
- Non-catalytic histidine environments and mutation mapping
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from Bio.PDB import PDBParser, Structure, Chain, Residue


@dataclass
class CatalyticTriadMetrics:
    ser160_og_coord: List[float]
    his237_ne2_coord: List[float]
    his237_nd1_coord: List[float]
    asp206_od1_coord: List[float]
    asp206_od2_coord: List[float]
    dist_ser_his_ne2_A: float
    dist_his_nd1_asp_od2_A: float
    dist_his_nd1_asp_od1_A: float
    is_geometry_intact: bool


@dataclass
class DisulfideBondMetrics:
    res1_id: int
    res2_id: int
    dist_sg_sg_A: float
    is_intact: bool  # Standard S-S bond distance is 2.05 ± 0.15 Å


@dataclass
class StructureValidationResult:
    system_id: str
    pdb_path: str
    chain_id: str
    num_standard_residues: int
    start_residue: Tuple[str, int]
    end_residue: Tuple[str, int]
    catalytic_triad: CatalyticTriadMetrics
    disulfides: List[DisulfideBondMetrics]
    histidines: List[int]
    mutations_from_5xjh: List[str]
    non_standard_heterogens: List[str]
    passed_validation: bool
    notes: List[str]


class SequenceStructureValidator:
    """Validates PETase crystal and modeled structures against biochemical references."""

    # Reference canonical positions in mature IsPETase (uniprot A0A0K8P6T7 without signal peptide)
    CATALYTIC_SER = 160
    CATALYTIC_ASP = 206
    CATALYTIC_HIS = 237

    DISULFIDE_1 = (203, 239)
    DISULFIDE_2 = (273, 289)

    def __init__(self):
        self.parser = PDBParser(QUIET=True)

    def load_structure(self, file_path: Path, struct_id: str) -> Structure:
        if not file_path.exists():
            raise FileNotFoundError(f"Structure file not found: {file_path}")
        return self.parser.get_structure(struct_id, str(file_path))

    def validate_system(
        self,
        file_path: Path,
        system_id: str,
        expected_chain: str = "A",
        reference_5xjh_path: Optional[Path] = None,
    ) -> StructureValidationResult:
        struct = self.load_structure(file_path, system_id)
        model = struct[0]

        if expected_chain not in model:
            raise ValueError(f"Chain {expected_chain} not found in {file_path}")
        chain = model[expected_chain]

        # Extract standard residues (id[0] == ' ')
        standard_res = [r for r in chain.get_residues() if r.id[0] == " "]
        hetero_res = [r for r in chain.get_residues() if r.id[0] != " "]

        notes: List[str] = []

        if not standard_res:
            raise ValueError(f"No standard amino acid residues found in chain {expected_chain}")

        start_res = (standard_res[0].resname, standard_res[0].id[1])
        end_res = (standard_res[-1].resname, standard_res[-1].id[1])

        # Validate Catalytic Triad
        triad_metrics = self._validate_catalytic_triad(chain, notes)

        # Validate Disulfides
        disulfides = self._validate_disulfides(chain, notes)

        # Histidine inventory
        histidines = [r.id[1] for r in standard_res if r.resname == "HIS"]

        # Heterogens
        hetero_names = sorted(list(set(r.resname for r in hetero_res)))

        # Mutation analysis relative to 5XJH if reference provided
        mutations: List[str] = []
        if reference_5xjh_path and reference_5xjh_path.exists() and reference_5xjh_path != file_path:
            mutations = self._calculate_mutations(chain, reference_5xjh_path, expected_chain)

        # Validation verdict
        passed = (
            triad_metrics.is_geometry_intact
            and all(ds.is_intact for ds in disulfides)
            and len(standard_res) >= 260
        )

        return StructureValidationResult(
            system_id=system_id,
            pdb_path=str(file_path),
            chain_id=expected_chain,
            num_standard_residues=len(standard_res),
            start_residue=start_res,
            end_residue=end_res,
            catalytic_triad=triad_metrics,
            disulfides=disulfides,
            histidines=histidines,
            mutations_from_5xjh=mutations,
            non_standard_heterogens=hetero_names,
            passed_validation=passed,
            notes=notes,
        )

    def _validate_catalytic_triad(self, chain: Chain, notes: List[str]) -> CatalyticTriadMetrics:
        # Check presence of Ser160, Asp206, His237
        for pos, name in [(self.CATALYTIC_SER, "SER"), (self.CATALYTIC_ASP, "ASP"), (self.CATALYTIC_HIS, "HIS")]:
            if pos not in chain:
                raise ValueError(f"Missing catalytic residue {name}{pos} in chain {chain.id}")
            if chain[pos].resname != name:
                raise ValueError(f"Residue {pos} is {chain[pos].resname}, expected {name}")

        ser = chain[self.CATALYTIC_SER]
        asp = chain[self.CATALYTIC_ASP]
        his = chain[self.CATALYTIC_HIS]

        og = ser["OG"].coord
        ne2 = his["NE2"].coord
        nd1 = his["ND1"].coord
        od1 = asp["OD1"].coord
        od2 = asp["OD2"].coord

        d_ser_his_ne2 = float(np.linalg.norm(og - ne2))
        d_his_nd1_asp_od2 = float(np.linalg.norm(nd1 - od2))
        d_his_nd1_asp_od1 = float(np.linalg.norm(nd1 - od1))

        # Check geometry bounds:
        # Ser OG to His NE2 hydrogen bond: typically 2.6 - 3.2 Å
        # His ND1 to Asp OD2 hydrogen bond: typically 2.5 - 3.0 Å
        is_intact = (2.5 <= d_ser_his_ne2 <= 3.3) and (2.4 <= min(d_his_nd1_asp_od1, d_his_nd1_asp_od2) <= 3.1)

        if not is_intact:
            notes.append(
                f"Catalytic triad geometry outside typical bounds: d(Ser-His)={d_ser_his_ne2:.2f}A, d(His-Asp)={min(d_his_nd1_asp_od1, d_his_nd1_asp_od2):.2f}A"
            )

        return CatalyticTriadMetrics(
            ser160_og_coord=og.tolist(),
            his237_ne2_coord=ne2.tolist(),
            his237_nd1_coord=nd1.tolist(),
            asp206_od1_coord=od1.tolist(),
            asp206_od2_coord=od2.tolist(),
            dist_ser_his_ne2_A=round(d_ser_his_ne2, 3),
            dist_his_nd1_asp_od2_A=round(d_his_nd1_asp_od2, 3),
            dist_his_nd1_asp_od1_A=round(d_his_nd1_asp_od1, 3),
            is_geometry_intact=is_intact,
        )

    def _validate_disulfides(self, chain: Chain, notes: List[str]) -> List[DisulfideBondMetrics]:
        results = []
        for cys1_pos, cys2_pos in [self.DISULFIDE_1, self.DISULFIDE_2]:
            if cys1_pos not in chain or cys2_pos not in chain:
                notes.append(f"Cysteines {cys1_pos} or {cys2_pos} missing from chain {chain.id}")
                continue

            r1 = chain[cys1_pos]
            r2 = chain[cys2_pos]
            if "SG" not in r1 or "SG" not in r2:
                notes.append(f"SG atom missing in Cys{cys1_pos} or Cys{cys2_pos}")
                continue

            dist = float(np.linalg.norm(r1["SG"].coord - r2["SG"].coord))
            is_intact = 1.90 <= dist <= 2.25

            if not is_intact:
                notes.append(f"Disulfide Cys{cys1_pos}-Cys{cys2_pos} distance {dist:.2f}A outside covalent range")

            results.append(
                DisulfideBondMetrics(
                    res1_id=cys1_pos,
                    res2_id=cys2_pos,
                    dist_sg_sg_A=round(dist, 3),
                    is_intact=is_intact,
                )
            )
        return results

    def _calculate_mutations(self, target_chain: Chain, ref_path: Path, chain_id: str) -> List[str]:
        ref_struct = self.load_structure(ref_path, "REF_5XJH")
        ref_chain = ref_struct[0][chain_id]

        ref_residues = {r.id[1]: r.resname for r in ref_chain.get_residues() if r.id[0] == " "}
        target_residues = {r.id[1]: r.resname for r in target_chain.get_residues() if r.id[0] == " "}

        mutations = []
        common_positions = sorted(set(ref_residues.keys()) & set(target_residues.keys()))
        for pos in common_positions:
            if ref_residues[pos] != target_residues[pos]:
                mutations.append(f"{ref_residues[pos]}{pos}{target_residues[pos]}")
        return mutations
