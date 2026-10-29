"""
Pytest Suite for Q-Catalyst Role 2: Computational Chemistry Subsystem.
Validates cluster construction, electrostatic embedding, PySCF execution,
CASCI reference calculations, and descriptor extraction.
"""

import os
import pytest
import numpy as np

from qcatalyst.chemistry.benchmarks import run_benchmark_molecule, REFERENCE_VALUES
from qcatalyst.chemistry.cluster import ActiveSiteClusterBuilder
from qcatalyst.chemistry.embedding import ElectrostaticEmbeddingBuilder
from qcatalyst.chemistry.pyscf_engine import PySCFChemistryEngine
from qcatalyst.chemistry.descriptors import DescriptorExtractor


def test_phase0_h2_benchmark():
    """Validates H2 Hartree-Fock and DFT energy against literature."""
    res = run_benchmark_molecule("H2", REFERENCE_VALUES["H2"])
    assert res["status"] == "PASSED"
    assert res["hf_converged"] is True
    assert res["dft_converged"] is True
    assert res["hf_diff_hartree"] < 0.05


def test_tier_s_cluster_builder():
    """Validates Tier S cluster construction from 5XJH."""
    builder = ActiveSiteClusterBuilder("data/structures/5XJH.pdb")
    cluster = builder.build_tier_s()
    assert cluster["natoms"] == 42
    assert cluster["charge"] == -1
    assert cluster["spin"] == 0
    assert len(cluster["atoms"]) == 42

    # Check key coordinates
    key_c = cluster["key_coords"]
    d_ser_his = np.linalg.norm(key_c["ser160_og"] - key_c["his237_ne2"])
    assert 2.5 <= d_ser_his <= 3.5  # Standard catalytic triad H-bond distance


def test_amber_embedding_builder():
    """Validates electrostatic embedding point charge generation and mutation sensitivity."""
    builder = ActiveSiteClusterBuilder("data/structures/5XJH.pdb")
    cluster = builder.build_tier_s()
    qm_coords = np.array([coord for _, coord in cluster["atoms"]])

    embed_builder = ElectrostaticEmbeddingBuilder("data/structures/5XJH.pdb")
    coords_wt, charges_wt, meta_wt = embed_builder.get_environment_charges(
        qm_coords=qm_coords, radius_cutoff=10.0, mutation="WT"
    )
    coords_mut, charges_mut, meta_mut = embed_builder.get_environment_charges(
        qm_coords=qm_coords, radius_cutoff=10.0, mutation="S121E"
    )

    assert len(coords_wt) > 100
    assert len(coords_wt) == len(charges_wt)
    # Crucial test: check embedding responds to mutation
    assert meta_wt["net_mm_charge"] != meta_mut["net_mm_charge"]


def test_pyscf_engine_and_casci():
    """Validates PySCF single-point and CASCI on Tier S cluster."""
    builder = ActiveSiteClusterBuilder("data/structures/5XJH.pdb")
    cluster = builder.build_tier_s()
    engine = PySCFChemistryEngine(basis="def2-svp", functional="b3lyp")

    # Run quick RHF single point
    sp = engine.run_single_point(
        geometry_str=cluster["geometry_str"],
        charge=cluster["charge"],
        spin=cluster["spin"],
        run_dft=False,
    )
    assert sp["hf_converged"] is True
    assert sp["nbasis"] == 390
    assert sp["e_hf"] < -1000.0

    # Run CASCI (4e, 4o)
    casci_res = engine.run_casci_reference(
        geometry_str=cluster["geometry_str"],
        ncas=4,
        nelecas=(2, 2),
        charge=cluster["charge"],
        spin=cluster["spin"],
    )
    assert casci_res["e_casci_total"] <= casci_res["e_hf_ref"]
    assert casci_res["correlation_energy_ha"] <= 0.0


def test_descriptor_extraction():
    """Validates extraction of all Section 6 descriptors on WT."""
    extractor = DescriptorExtractor("data/structures/5XJH.pdb", basis="def2-svp", functional="b3lyp")
    desc = extractor.extract_variant_descriptors(
        variant_id="VAR_WT",
        mutation="WT",
        include_approach_scan=False,
        include_casci=False,
    )
    required_keys = [
        "e_hf", "e_dft", "esp_at_ser_og_au", "esp_at_ser_og_kcal_mol",
        "charge_ser_og", "charge_his_ne2", "charge_asp_od1", "charge_asp_od2",
        "proton_shuttle_proxy_kcal", "dist_ser_og_ester_c", "dist_ser_og_his_ne2",
        "wall_clock_sec", "nbasis", "code_version"
    ]
    for key in required_keys:
        assert key in desc, f"Missing descriptor key: {key}"
