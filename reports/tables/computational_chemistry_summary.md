# Computational Chemistry Deliverables Summary (Role 2 Lead)

## 1. Executive Summary
- **Total variants computed:** 31 (1 WT, 15 Arm A4 Gated, 15 Arm A3 Random Control)
- **Quantum CASCI Controls computed for Role 3:** 6 candidates (Arm A5)
- **Average CPU wall-clock time per variant:** 446.54 s
- **Primary Parquet Database:** [`results/chemistry/chem_results.parquet`](file:///home/luckycelestial/Pavithran/QCatalyst/Q_Catalyst/results/chemistry/chem_results.parquet)

## 2. Top Gated Variants (Arm A4) vs WT Descriptors

| variant_id   | mutation   |   dist_to_ser160_og |   delta_esp_kcal_mol |   delta_charge_ser_og |   delta_proton_shuttle_proxy_kcal |   delta_approach_barrier_proxy_kcal |
|:-------------|:-----------|--------------------:|---------------------:|----------------------:|----------------------------------:|------------------------------------:|
| VAR_A4_01    | S121E      |                6.82 |                9.178 |               -0.0436 |                             0.678 |                               2.199 |
| VAR_A4_02    | D186H      |                7.15 |               28.475 |                0.0062 |                             1.457 |                               4.42  |
| VAR_A4_03    | W159H      |                4.51 |               82.694 |               -0.0267 |                            25.357 |                              25.633 |
| VAR_A4_04    | S238F      |                5.12 |                8.919 |               -0.0312 |                             1.103 |                              -0.931 |
| VAR_A4_05    | T88I       |                5.83 |               14.025 |               -0.0288 |                             3.112 |                               4.929 |
| VAR_A4_06    | N233K      |                7.48 |               16.391 |               -0.0089 |                            -3.426 |                              -6.349 |
| VAR_A4_07    | I208V      |                6.09 |               -0.795 |               -0.0325 |                            -0.521 |                               0.249 |
| VAR_A4_08    | R224Q      |                7.91 |                0     |                0      |                             0     |                               0     |
| VAR_A4_09    | L117F      |                6.94 |                7.763 |               -0.0586 |                             1.102 |                               2.479 |
| VAR_A4_10    | S238Y      |                5.12 |                9.501 |               -0.0316 |                             1.236 |                              -0.909 |
| VAR_A4_11    | T88M       |                5.83 |               20.14  |               -0.0351 |                             4.177 |                               7.124 |
| VAR_A4_12    | S121D      |                6.82 |                7.195 |               -0.0014 |                             0.509 |                               1.653 |
| VAR_A4_13    | W159F      |                4.51 |               39.597 |               -0.0305 |                            10.407 |                               9.869 |
| VAR_A4_14    | S238A      |                5.12 |                3.654 |               -0.0245 |                            -0.094 |                              -1.129 |
| VAR_A4_15    | D186N      |                7.15 |               21.516 |               -0.0538 |                             0.951 |                               3.051 |

## 3. Comparison with Random Control (Arm A3)

| variant_id   | mutation   |   dist_to_ser160_og |   delta_esp_kcal_mol |   delta_charge_ser_og |   delta_proton_shuttle_proxy_kcal |   delta_approach_barrier_proxy_kcal |
|:-------------|:-----------|--------------------:|---------------------:|----------------------:|----------------------------------:|------------------------------------:|
| VAR_A3_01    | G158A      |                7.1  |                0     |                0      |                             0     |                               0     |
| VAR_A3_02    | Q119A      |                7.42 |               23.842 |               -0.0053 |                             2.857 |                               8.105 |
| VAR_A3_03    | A162G      |                4.81 |                0     |                0      |                             0     |                               0     |
| VAR_A3_04    | T88A       |                5.83 |               14.344 |               -0.0333 |                             3.06  |                               5.087 |
| VAR_A3_05    | S121A      |                6.82 |                3.699 |               -0.0323 |                             0.206 |                               0.707 |
| VAR_A3_06    | N233A      |                7.48 |               13.085 |                0.0029 |                            -2.816 |                              -5.155 |
| VAR_A3_07    | I208A      |                6.09 |                9.336 |               -0.0383 |                             2.276 |                               3.478 |
| VAR_A3_08    | V207A      |                6.41 |                5.616 |               -0.032  |                             0.68  |                              -0.724 |
| VAR_A3_09    | L117A      |                6.94 |                9.978 |               -0.0344 |                             1.425 |                               3.11  |
| VAR_A3_10    | P184A      |                7.62 |               -8.78  |               -0.0327 |                            -0.195 |                              -0.387 |
| VAR_A3_11    | E204A      |                7.84 |               10.462 |               -0.019  |                            -2.72  |                              -4.688 |
| VAR_A3_12    | N241A      |                6.72 |               33.526 |               -0.032  |                             8.23  |                               6.546 |
| VAR_A3_13    | Y87F       |                5.6  |               11.482 |               -0.1265 |                             1.57  |                               3.996 |
| VAR_A3_14    | M161A      |                3.29 |                2.496 |               -0.0013 |                            -0.182 |                               1.31  |
| VAR_A3_15    | S238T      |                5.12 |               23.119 |               -0.0297 |                             4.333 |                              -0.408 |

## 4. FAST-PETase (7SH6) Directional Plausibility Check

- **IsPETase (5XJH) Ser160-His237 distance:** 2.923 Å
- **FAST-PETase (7SH6) Ser160-His237 distance:** 2.851 Å
- **IsPETase (5XJH) His237-Asp206 distance:** 2.675 Å
- **FAST-PETase (7SH6) His237-Asp206 distance:** 2.662 Å
- **Δ Ser160 OG Mulliken Charge:** +0.0066
- **Literature Benchmark:** Acylation barrier 12.1 kcal/mol (FAST-PETase) vs 16.5 kcal/mol (IsPETase)
- **Verdict:** CONSISTENT (FAST-PETase maintains intact, slightly tightened catalytic triad geometry and preserved nucleophile polarisation)

## 5. Sensitivity Summary

### Protonation State Sensitivity (HIE vs HIP)
| variant_id   | test_parameter   |   esp_at_ser_og_kcal_mol |   charge_ser_og |
|:-------------|:-----------------|-------------------------:|----------------:|
| WT           | HIE              |                 -3738.77 |         -0.2654 |
| S121E        | HIE              |                 -3729.59 |         -0.309  |
| WT           | HIP              |                 -3738.77 |         -0.2104 |
| S121E        | HIP              |                 -3729.59 |         -0.197  |

### Radius Cutoff Sensitivity (R = 6, 8, 10, 12 Å)
| variant_id   | test_parameter   |   radius_cutoff_A |   esp_at_ser_og_kcal_mol |   charge_ser_og |
|:-------------|:-----------------|------------------:|-------------------------:|----------------:|
| WT           | R=6.0A           |                 6 |                 -2102.25 |         -0.2701 |
| WT           | R=8.0A           |                 8 |                 -2843.14 |         -0.2888 |
| WT           | R=10.0A          |                10 |                 -3738.77 |         -0.2654 |
| WT           | R=12.0A          |                12 |                 -4332.3  |         -0.3052 |

