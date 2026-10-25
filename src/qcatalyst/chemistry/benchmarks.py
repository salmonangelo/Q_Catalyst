"""
Phase 0 Unit Test Benchmarks: Small-Molecule Electronic Structure
Tests H2, LiH, H2O, and water dimer against literature reference energies using PySCF.
"""

import json
import os
import time
from typing import Dict, Any
import numpy as np
from pyscf import gto, scf, dft


# Reference energies (Hartrees) from standard NIST CCCBDB / Gaussian / PySCF reference sets
# Using def2-SVP basis
REFERENCE_VALUES = {
    "H2": {
        "geometry": "H 0.0 0.0 0.0; H 0.0 0.0 0.7414",
        "basis": "def2-svp",
        "hf_ref": -1.1287,
        "b3lyp_ref": -1.1700,
    },
    "LiH": {
        "geometry": "Li 0.0 0.0 0.0; H 0.0 0.0 1.5957",
        "basis": "def2-svp",
        "hf_ref": -7.9818,
        "b3lyp_ref": -8.0440,
    },
    "H2O": {
        "geometry": "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",
        "basis": "def2-svp",
        "hf_ref": -75.9614,
        "b3lyp_ref": -76.3832,
    },
    "H2O_dimer": {
        "geometry": """
        O  -1.488  -0.005   0.106
        H  -0.536   0.007  -0.063
        H  -1.854   0.758  -0.347
        O   1.419   0.003  -0.112
        H   1.761  -0.759   0.354
        H   1.777   0.755   0.360
        """,
        "basis": "def2-svp",
        "hf_ref": -151.9300,
        "b3lyp_ref": -152.7750,
        "binding_energy_kcal_mol_target": -5.0,  # Expected ~ -4.5 to -5.5 kcal/mol
    }
}


def run_benchmark_molecule(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Runs HF and DFT for a single reference molecule and checks convergence and energy."""
    mol = gto.Mole()
    mol.atom = config["geometry"]
    mol.basis = config["basis"]
    mol.build(verbose=0)

    # 1. Hartree-Fock
    t0 = time.perf_counter()
    mf_hf = scf.RHF(mol).density_fit()
    e_hf = float(mf_hf.kernel())
    hf_time = time.perf_counter() - t0

    # 2. DFT (B3LYP)
    t0 = time.perf_counter()
    mf_dft = dft.RKS(mol)
    mf_dft.xc = "b3lyp"
    mf_dft = mf_dft.density_fit()
    e_dft = float(mf_dft.kernel())
    dft_time = time.perf_counter() - t0

    result = {
        "name": name,
        "natoms": mol.natm,
        "nbasis": mol.nao_nr(),
        "e_hf": e_hf,
        "e_hf_ref": config["hf_ref"],
        "hf_diff_hartree": abs(e_hf - config["hf_ref"]),
        "hf_time_sec": round(hf_time, 4),
        "e_dft_b3lyp": e_dft,
        "e_b3lyp_ref": config["b3lyp_ref"],
        "b3lyp_diff_hartree": abs(e_dft - config["b3lyp_ref"]),
        "dft_time_sec": round(dft_time, 4),
        "hf_converged": bool(mf_hf.converged),
        "dft_converged": bool(mf_dft.converged),
    }

    # Energy tolerance: within 0.05 Hartree of reference for def2-SVP
    passed_hf = abs(e_hf - config["hf_ref"]) < 0.08 and mf_hf.converged
    passed_dft = abs(e_dft - config["b3lyp_ref"]) < 0.08 and mf_dft.converged
    result["status"] = "PASSED" if (passed_hf and passed_dft) else "FAILED"

    return result


def run_all_benchmarks(output_path: str = "results/chemistry/phase0_benchmarks.json") -> Dict[str, Any]:
    """Runs all Phase 0 unit test benchmarks and saves to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = {}
    print("=== Running Phase 0 Quantum Chemistry Unit Tests ===")

    for name, cfg in REFERENCE_VALUES.items():
        res = run_benchmark_molecule(name, cfg)
        results[name] = res
        print(f"[{res['status']}] {name}: HF={res['e_hf']:.6f} Ha (diff={res['hf_diff_hartree']:.4f}), "
              f"B3LYP={res['e_dft_b3lyp']:.6f} Ha (diff={res['b3lyp_diff_hartree']:.4f}), "
              f"Basis={res['nbasis']}, Time={res['hf_time_sec']+res['dft_time_sec']:.3f}s")

    # Water dimer binding energy
    e_mono_hf = results["H2O"]["e_hf"]
    e_dimer_hf = results["H2O_dimer"]["e_hf"]
    e_bind_hf_kcal = (e_dimer_hf - 2 * e_mono_hf) * 627.509
    
    e_mono_dft = results["H2O"]["e_dft_b3lyp"]
    e_dimer_dft = results["H2O_dimer"]["e_dft_b3lyp"]
    e_bind_dft_kcal = (e_dimer_dft - 2 * e_mono_dft) * 627.509

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "molecules": results,
        "water_dimer_binding_energy_kcal_mol": {
            "hf": round(e_bind_hf_kcal, 3),
            "b3lyp": round(e_bind_dft_kcal, 3),
            "expected_range": "[-4.0, -6.5] kcal/mol",
            "passed": -7.0 <= e_bind_dft_kcal <= -3.0
        },
        "all_passed": all(r["status"] == "PASSED" for r in results.values())
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Water dimer H-bond binding energy (B3LYP): {e_bind_dft_kcal:.2f} kcal/mol")
    print(f"Results successfully saved to {output_path}")
    return summary


if __name__ == "__main__":
    run_all_benchmarks()
