# Interface Contracts and Schema Specifications

This document defines the authoritative file contracts across all 4 roles in the Q-Catalyst pipeline.

## 1. `variants.parquet` (Role 1)
- `variant_id` (str): Unique variant identifier (e.g. `PET-VAR-001`).
- `parent_enzyme` (str): Parent wild-type enzyme name (e.g. `IsPETase`).
- `sequence` (str): Full amino acid sequence (mature form, length 290).
- `mutations` (str): Comma-separated list of mutations in 5XJH numbering (e.g. `S121E,D186H,R280A`) or `WT`.
- `mutation_count` (int): Number of point mutations from parent.
- `study_id` (str): Source publication or dataset release.
- `assay_id` (str): Assay identifier.
- `temperature_c` (float): Assay incubation temperature in °C.
- `ph` (float): Assay buffer pH.
- `substrate_type` (str): Substrate state (e.g. `PET_film_amorphous`, `PET_powder`).
- `raw_activity` (float): Direct experimental activity metric.
- `activity_unit` (str): Measurement unit (e.g. `umol_product_h_mg`).
- `relative_activity` (float): Normalized relative activity ($A_{\text{variant}} / A_{\text{WT}}$).
- `tm_c` (float, nullable): Melting temperature in °C.
- `delta_tm` (float, nullable): Thermal stability change vs WT in °C ($T_m - T_{m,\text{WT}}$).
- `is_prior_art` (bool): True if variant corresponds to known published high-performance variant (e.g. FAST-PETase, DuraPETase, ThermoPETase).
- `data_quality_flag` (str): Quality flag (`VERIFIED_EXPERIMENTAL`, `INFERRED_REPLICATE`, etc.).

## 2. `predictions.parquet` (Role 1)
- `variant_id` (str): Matches `variants.parquet`.
- `pred_activity` (float): Supervised predicted relative activity.
- `pred_interval_lower` (float): 90% conformal lower bound.
- `pred_interval_upper` (float): 90% conformal upper bound.
- `pred_delta_tm` (float): Predicted thermal stability shift $\Delta T_m$.
- `zero_shot_esm_score` (float): ESM zero-shot fitness score.
- `split` (str): Dataset split (`train`, `val`, `test`).
- `model_version` (str): Predictor version identifier.

## 3. `uncertainty.parquet` (Role 1)
- `variant_id` (str): Matches `variants.parquet`.
- `aggregate_uncertainty` (float): Normalized uncertainty score $U \in [0, 1]$.
- `ensemble_std` (float): Standard deviation across model ensemble predictions.
- `conformal_interval_width` (float): Upper minus lower conformal bound.
- `ood_distance` (float): Mahalanobis/Euclidean embedding distance from WT reference cluster.
- `structural_risk` (float): Risk penalty based on proximity to catalytic triad or buried core.
- `gate0_pass` (bool): Whether Gate 0 validation criteria were satisfied.
- `abstention_flag` (bool): True if uncertainty is excessively high for safe recommendation.

## 4. `chem_jobs.json` (Role 4)
- `job_id` (str): Computational chemistry job ID (e.g. `CHEM-JOB-001`).
- `variant_id` (str): Target variant.
- `mutations` (str): Mutation description.
- `min_dist_to_triad_angstrom` (float): Minimum distance to Ser160/active site.
- `acquisition_score` (float): Computed score $a(x) = \lambda U + \eta P + \mu D - \nu C$.
- `selection_rank` (int): Rank in chemistry queue.
- `status` (str): `SCHEDULED`, `COMPLETED`, `SKIPPED`.

## 5. `chem_results.parquet` (Role 2)
- `job_id` (str): Matches `chem_jobs.json`.
- `variant_id` (str): Matches `variants.parquet`.
- `cluster_model` (str): Active-site cluster description (Triad + oxyanion + substrate).
- `d_attack_angstrom` (float): Distance from Ser160-Oγ to substrate ester carbonyl C.
- `d_hbond_his_ser` (float): His237-Nε2 to Ser160-Oγ hydrogen bond length.
- `d_hbond_asp_his` (float): Asp206-Oδ to His237-Nδ1 hydrogen bond length.
- `esp_active_site_au` (float): Electrostatic potential at catalytic center.
- `delta_e_int_kcal_mol` (float): Interaction energy proxy relative to wild type.
- `homo_lumo_gap_ev` (float): Active site HOMO-LUMO energy gap.
- `ts_stabilization_kcal_mol` (float): Transition-state stabilization proxy.
- `qm_runtime_seconds` (float): PySCF cluster execution duration.
- `qm_converged` (bool): Convergence flag.

## 6. `quantum_results.parquet` (Role 3)
- `job_id` (str): Quantum evaluation job ID.
- `variant_id` (str): Target variant.
- `active_space` (str): Active space specification (e.g. `(4e, 4o)`).
- `e_casci_hartree` (float): Exact classical ground-state energy.
- `e_vqe_hartree` (float): Ground-state energy from VQE simulation.
- `energy_error_hartree` (float): $|E_{\text{VQE}} - E_{\text{CASCI}}|$.
- `energy_error_kcal_mol` (float): Absolute error in kcal/mol.
- `chemical_accuracy_met` (bool): True if error $\le 1.6$ kcal/mol.
- `qubit_count` (int): Number of qubits (8 for (4e, 4o) Jordan-Wigner).
- `circuit_depth` (int): Quantum circuit depth.
- `cnot_count` (int): Number of two-qubit entangling gates.
- `vqe_iterations` (int): Number of optimizer evaluations.
- `noise_level` (float): Simulated depolarizing error rate.
- `vqe_runtime_seconds` (float): Simulator runtime.

## 7. `ranking.parquet` (Role 4)
- `final_rank` (int): Final prioritised ranking order.
- `variant_id` (str): Identifier.
- `mutations` (str): Mutations in 5XJH numbering.
- `confidence_tier` (str): Tier 1 (High Confidence Mechanism-Backed), Tier 2 (AI-Promising Needs Wet Lab), Tier 3 (High Risk / High Uncertainty).
- `final_composite_score` (float): Fused score incorporating AI, uncertainty, and chemistry evidence.
- `ai_rank` (int): Baseline AI-only rank.
- `rank_delta` (int): Rank change after chemistry evidence incorporation ($R_{\text{AI}} - R_{\text{final}}$).
- `chem_supported` (bool): True if candidate was validated by classical chemistry.
- `quantum_supported` (bool): True if candidate was validated by quantum workflow.
- `prior_art_name` (str): Known literature designation (e.g. `FAST-PETase`, `DuraPETase`, `Novel`).
- `recommendation_reason` (str): Traceable rationale for recommendation.
- `limitations` (str): Explicit boundaries and caveats.

## 8. `benchmark_results.parquet` (Roles 1 & 4)
- `arm_id` (str): Arm identifier (`A0` through `A7`).
- `arm_name` (str): Strategy name.
- `split` (str): Evaluation split (`test`).
- `precision_at_5` (float): Precision@5.
- `precision_at_10` (float): Precision@10.
- `precision_at_20` (float): Precision@20.
- `ci_lower_p10` (float): 95% bootstrap lower bound for Precision@10.
- `ci_upper_p10` (float): 95% bootstrap upper bound for Precision@10.
- `spearman_rho` (float): Rank correlation with ground truth activity.
- `voc_efficiency` (float): Value of Chemistry per CPU hour.
- `total_cpu_seconds` (float): Cumulative compute cost.
