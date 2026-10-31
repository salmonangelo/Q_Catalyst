# Analysis Module (Milestone 2+ Architecture)

This package contains planned trajectory analysis pipelines:

- `rmsd/`: Backbone and active-site root-mean-square deviations across production time.
- `rmsf/`: Per-residue root-mean-square fluctuations evaluating catalytic loop flexibilities (β1-β2, β3-α2, β7-α5).
- `substrate_contacts/`: Protein–ligand interaction profiling using MDAnalysis, PLIP, and ProLIF patterns.
- `catalytic_geometry/`: Attack distance ($d(\text{Ser160 } O_\gamma - \text{PET } C)$) and proton transfer hydrogen bonds ($d(\text{His237 } N_{\epsilon2} - \text{Ser160 } O_\gamma)$, $d(\text{Asp206 } O_\delta - \text{His237 } N_{\delta1})$).
- `free_energy/`: Metadynamics free-energy surface reweighting (TI / WHAM / PLUMED sum_hills).

> [!NOTE]
> All metrics will be calculated on real trajectories in future milestones.
