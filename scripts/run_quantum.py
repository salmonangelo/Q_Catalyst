"""
Script 5: Execute quantum workflow - CASCI exact reference vs VQE simulator with noise sweep.
"""
import os
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.qcatalyst.quantum.vqe_solver import QuantumVQESolver


def main():
    print("=" * 60)
    print("Q-CATALYST: Step 5 - Quantum Electronic Structure (CASCI vs VQE)")
    print("=" * 60)

    chem_path = PROJECT_ROOT / "results" / "chemistry" / "chem_results.parquet"
    if not chem_path.exists():
        raise FileNotFoundError(f"Missing {chem_path}. Run run_chemistry.py first.")

    df_chem = pd.read_parquet(chem_path)
    # Select top 5 chemistry variants for quantum evaluation
    top_quantum_candidates = df_chem.head(5)

    solver = QuantumVQESolver(n_qubits=8, active_space="(4e, 4o)")
    all_quantum_results = []

    print(f"Running (4e, 4o) active space VQE and exact CASCI control on {len(top_quantum_candidates)} variants...")

    for _, row in top_quantum_candidates.iterrows():
        q_rows = solver.evaluate_variant_quantum(
            variant_id=row["variant_id"],
            mutations=row["mutations"],
            ts_stab_kcal_mol=row["ts_stabilization_kcal_mol"],
            noise_levels=[0.0, 0.002, 0.008, 0.020]
        )
        all_quantum_results.extend(q_rows)

        # Log noiseless parity
        noiseless = q_rows[0]
        print(f"  [{row['variant_id']}] {row['mutations']} -> CASCI={noiseless['e_casci_hartree']} Ha, "
              f"VQE={noiseless['e_vqe_hartree']} Ha, Error={noiseless['energy_error_kcal_mol']:.3f} kcal/mol "
              f"(Chemical Accuracy: {noiseless['chemical_accuracy_met']})")

    df_quantum = pd.DataFrame(all_quantum_results)
    q_dir = PROJECT_ROOT / "results" / "quantum"
    q_dir.mkdir(parents=True, exist_ok=True)
    out_path = q_dir / "quantum_results.parquet"
    df_quantum.to_parquet(out_path, index=False)

    print(f"\nSaved quantum validation results to: {out_path}")
    print("Step 5 Complete!\n")


if __name__ == "__main__":
    main()
