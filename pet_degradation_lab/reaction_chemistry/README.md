# Reaction Chemistry Module (Milestone 3 Architecture)

Defines the planned hybrid QM/MM enzyme reaction barrier calculations:

- `qmmm_inputs/`: Cluster boundary definitions, link atoms, capping, and CP2K/PySCF calculation parameters.
- `qmmm_results/`: Transition-state energy barriers, nudged-elastic-band (NEB) reaction profiles for tetrahedral intermediate formation and collapse.

> [!CAUTION]
> Single-point electronic potential energies are not reaction free energies. Full QM/MM free-energy pathways will be computed in Milestone 3.
