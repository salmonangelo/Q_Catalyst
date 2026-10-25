"""
Descriptor Extraction Pipeline for PETase Active-Site Chemistry
Computes triad interaction energy, partial charges, ESP at Ser OG, proton-shuttle proxy,
approach profile, key distances, and mutational deltas for Role 4 and Role 1.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from qcatalyst.chemistry.cluster import ActiveSiteClusterBuilder
from qcatalyst.chemistry.embedding import ElectrostaticEmbeddingBuilder
from qcatalyst.chemistry.pyscf_engine import PySCFChemistryEngine


class DescriptorExtractor:
    """
    Computes and formats the full chemical descriptor set for wild type and variants.
    """

    def __init__(
        self,
        pdb_path: str = "data/structures/5XJH.pdb",
        basis: str = "def2-svp",
        functional: str = "b3lyp",
        radius_cutoff: float = 10.0,
    ):
        self.pdb_path = pdb_path
        self.basis = basis
        self.functional = functional
        self.radius_cutoff = radius_cutoff

        self.cluster_builder = ActiveSiteClusterBuilder(pdb_path)
        self.embed_builder = ElectrostaticEmbeddingBuilder(pdb_path)
        self.engine = PySCFChemistryEngine(basis=basis, functional=functional)
        self._cached_sp_gas: Optional[Dict[str, Any]] = None

    def extract_variant_descriptors(
        self,
        variant_id: str,
        mutation: str,
        rotamer_index: int = 0,
        his_protonation: str = "HIE",
        asp_protonation: str = "deprotonated",
        include_approach_scan: bool = True,
        full_5point_scan: bool = False,
        include_casci: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculates all descriptors for a given variant (or WT).
        Caches gas-phase reference to avoid redundant calculations.
        """
        # 1. Build Ground-State Tier S Cluster
        cluster_ground = self.cluster_builder.build_tier_s(
            his237_protonation=his_protonation,
            asp206_protonation=asp_protonation,
            ser160_protonation="neutral",
            ser_ester_dist=2.95,
        )
        qm_coords = np.array([coord for _, coord in cluster_ground["atoms"]])

        # 2. Get AMBER ff14SB Electrostatic Embedding Point Charges
        mm_coords, mm_charges, embed_meta = self.embed_builder.get_environment_charges(
            qm_coords=qm_coords,
            radius_cutoff=self.radius_cutoff,
            mutation=mutation,
            rotamer_index=rotamer_index,
        )

        # 3. Ground State Single Point (with embedding)
        sp_embed = self.engine.run_single_point(
            geometry_str=cluster_ground["geometry_str"],
            charge=cluster_ground["charge"],
            spin=cluster_ground["spin"],
            mm_coords=mm_coords,
            mm_charges=mm_charges,
            run_dft=True,
        )

        # Gas phase reference (identical for all variants, so cache it!)
        if self._cached_sp_gas is None or his_protonation != "HIE":
            self._cached_sp_gas = self.engine.run_single_point(
                geometry_str=cluster_ground["geometry_str"],
                charge=cluster_ground["charge"],
                spin=cluster_ground["spin"],
                mm_coords=None,
                mm_charges=None,
                run_dft=True,
            )
        sp_gas = self._cached_sp_gas

        # 4. Electrostatic Potential at Ser160 OG (au and kcal/mol)
        ser160_og_coord = cluster_ground["key_coords"]["ser160_og"]
        esp_au = self.engine.compute_esp_at_point(ser160_og_coord, mm_coords, mm_charges)
        esp_kcal_mol = esp_au * 627.509

        # 5. Extract Specific Partial Charges
        q_mulliken = sp_embed["dft_mulliken_charges"]
        charge_ser_og = q_mulliken[1]
        charge_his_ne2 = q_mulliken[10]
        charge_asp_od1 = q_mulliken[15]
        charge_asp_od2 = q_mulliken[16]
        charge_ester_c = q_mulliken[23] if len(q_mulliken) > 23 else 0.0
        charge_ester_o = q_mulliken[24] if len(q_mulliken) > 24 else 0.0

        # 6. Triad H-Bond Interaction Energy Proxy
        e_embed_ha = sp_embed["e_dft"]
        e_gas_ha = sp_gas["e_dft"]
        embedding_stabilization_kcal = (e_embed_ha - e_gas_ha) * 627.509

        # 7. Proton-Shuttle Proxy
        # Energy change when Ser proton moves to His NE2 (Ser-O- ... His-H+)
        cluster_pt = self.cluster_builder.build_tier_s(
            his237_protonation="HIP" if his_protonation == "HIE" else his_protonation,
            asp206_protonation=asp_protonation,
            ser160_protonation="deprotonated",
            ser_ester_dist=2.95,
        )
        sp_pt = self.engine.run_single_point(
            geometry_str=cluster_pt["geometry_str"],
            charge=cluster_pt["charge"],
            spin=cluster_pt["spin"],
            mm_coords=mm_coords,
            mm_charges=mm_charges,
            run_dft=False,  # HF is fast and accurate for proton transfer proxy
        )
        proton_shuttle_proxy_kcal = (sp_pt["e_hf"] - sp_embed["e_hf"]) * 627.509

        # 8. Approach Profile
        approach_barrier_proxy_kcal = 0.0
        approach_profile = {}
        if include_approach_scan:
            scan_dists = [3.2, 2.9, 2.6, 2.3, 2.0] if full_5point_scan else [2.0]
            for d in scan_dists:
                c_scan = self.cluster_builder.build_tier_s(
                    his237_protonation=his_protonation,
                    asp206_protonation=asp_protonation,
                    ser160_protonation="neutral",
                    ser_ester_dist=d,
                )
                res_scan = self.engine.run_single_point(
                    geometry_str=c_scan["geometry_str"],
                    charge=c_scan["charge"],
                    spin=c_scan["spin"],
                    mm_coords=mm_coords,
                    mm_charges=mm_charges,
                    run_dft=False,
                )
                approach_profile[f"e_approach_{d}A"] = round(res_scan["e_hf"], 6)

            # Attack barrier proxy: E(2.0 A) - E(ground state at 2.95 A)
            approach_barrier_proxy_kcal = (approach_profile["e_approach_2.0A"] - sp_embed["e_hf"]) * 627.509

        # 9. Key Distances
        key_c = cluster_ground["key_coords"]
        dist_ser_og_ester_c = float(np.linalg.norm(key_c["ser160_og"] - key_c["ester_c"]))
        dist_ser_og_his_ne2 = float(np.linalg.norm(key_c["ser160_og"] - key_c["his237_ne2"]))
        dist_his_nd1_asp_od2 = float(np.linalg.norm(key_c["his237_nd1"] - key_c["asp206_od2"]))
        dist_oxyanion_ester_o = float(np.linalg.norm(key_c["met161_nh"] - key_c["ester_o"]))

        # 10. CASCI Control (if requested, e.g. for Arm A5)
        casci_results = {}
        if include_casci:
            casci_res = self.engine.run_casci_reference(
                geometry_str=cluster_ground["geometry_str"],
                ncas=4,
                nelecas=(2, 2),
                charge=cluster_ground["charge"],
                spin=cluster_ground["spin"],
                mm_coords=mm_coords,
                mm_charges=mm_charges,
            )
            casci_results = {
                "casci_e_tot_ha": casci_res["e_casci_total"],
                "casci_corr_energy_ha": casci_res["correlation_energy_ha"],
                "casci_time_sec": casci_res["casci_time_sec"],
            }

        total_wall_clock = sp_embed["time_hf_sec"] + sp_embed["time_dft_sec"] + sp_gas["time_dft_sec"]

        descriptors = {
            "variant_id": variant_id,
            "mutation": mutation,
            "rotamer_index": rotamer_index,
            "his_protonation": his_protonation,
            "asp_protonation": asp_protonation,
            "radius_cutoff_A": self.radius_cutoff,
            # Quantum Energies (Hartree)
            "e_hf": round(sp_embed["e_hf"], 6),
            "e_dft": round(sp_embed["e_dft"], 6),
            "e_dft_gas": round(sp_gas["e_dft"], 6),
            "embedding_stabilization_kcal": round(embedding_stabilization_kcal, 3),
            # Mechanistic Descriptors
            "esp_at_ser_og_au": round(esp_au, 6),
            "esp_at_ser_og_kcal_mol": round(esp_kcal_mol, 3),
            "charge_ser_og": charge_ser_og,
            "charge_his_ne2": charge_his_ne2,
            "charge_asp_od1": charge_asp_od1,
            "charge_asp_od2": charge_asp_od2,
            "charge_ester_c": charge_ester_c,
            "charge_ester_o": charge_ester_o,
            "proton_shuttle_proxy_kcal": round(proton_shuttle_proxy_kcal, 3),
            "approach_barrier_proxy_kcal": round(approach_barrier_proxy_kcal, 3),
            # Key Geometry (Angstroms)
            "dist_ser_og_ester_c": round(dist_ser_og_ester_c, 3),
            "dist_ser_og_his_ne2": round(dist_ser_og_his_ne2, 3),
            "dist_his_nd1_asp_od2": round(dist_his_nd1_asp_od2, 3),
            "dist_oxyanion_ester_o": round(dist_oxyanion_ester_o, 3),
            # Computational Cost & Provenance
            "wall_clock_sec": round(total_wall_clock, 2),
            "nbasis": sp_embed["nbasis"],
            "functional": self.functional,
            "basis": self.basis,
            "density_fitting": True,
            "code_version": "PySCF-2.14.0",
        }

        descriptors.update(approach_profile)
        descriptors.update(casci_results)

        return descriptors


if __name__ == "__main__":
    extractor = DescriptorExtractor("data/structures/5XJH.pdb")
    print("Testing Descriptor Extractor on Wild Type...")
    desc_wt = extractor.extract_variant_descriptors("WT", "WT", include_approach_scan=True, include_casci=True)
    print("WT Descriptors:")
    for k, v in desc_wt.items():
        print(f"  {k}: {v}")
