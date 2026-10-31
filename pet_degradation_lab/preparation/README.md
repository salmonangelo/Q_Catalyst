# Preparation Module (Milestone 2 Architecture)

This package defines the future structural preparation workflow for PETase variants and PET substrates:

- `sequence_mapping/`: Explicit alignment and sequence-to-structure residue number translation.
- `mutant_structure/`: In silico mutagenesis and rotamer optimization.
- `pet_substrate/`: Oligomeric PET substrate model construction (mono-, di-, and tri-PET models).
- `forcefield/`: AMBER ff14SB protein parameters + GAFF2/RESP parameterization for PET ester linkages.
- `solvated_system/`: Explicit TIP3P water box, neutralising counterions, and periodic boundary conditions.

> [!NOTE]
> **Status:** Pending Milestone 2. No pretend simulation structures are generated in Milestone 1.
