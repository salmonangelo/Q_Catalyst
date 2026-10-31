# Simulation Module (Milestone 2+ Architecture)

This package specifies the molecular dynamics protocol for PETase–PET complexes:

- `molecular_dynamics/`: OpenMM simulation pipeline (minimization, NVT thermalisation, NPT equilibration, production).
- `gromacs/`: Reference GROMACS run configurations and topology builders based on Colizzi et al. (PETase_Cryst-vs-Amor).
- `plumed/`: PLUMED input templates for Hamiltonian Replica Exchange (HREX) and Well-Tempered Metadynamics.

> [!IMPORTANT]
> **Scientific Rule:** Molecular dynamics trajectories must be calculated on validated systems, never mocked. In Milestone 1, no simulations are launched.
