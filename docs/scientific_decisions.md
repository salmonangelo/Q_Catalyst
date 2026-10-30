# Scientific Decisions and Boundaries

This document records the foundational scientific rationale, boundaries, and trade-offs in Q-Catalyst.

## 1. Numbering Convention: 5XJH Mature Reference
- **Decision:** All mutations are standardized to the mature *Ideonella sakaiensis* PETase (IsPETase) crystallographic numbering from PDB **5XJH** (residues 1–290).
- **Rationale:** Literature sources frequently alternate between pre-protein (with 27-residue signal peptide, e.g. Ser160 numbered as Ser187) and mature protein. Unifying on 5XJH prevents critical off-by-27 indexing errors.
- **Catalytic Triad in 5XJH:**
  - Nucleophile: Ser160 (Oγ)
  - General base: His237 (Nε2 / Nδ1)
  - Acid stabilizer: Asp206 (Oδ)
  - Oxyanion hole: Tyr87 & Met161 backbone amides

## 2. Active-Site Cluster Scope
- **Boundary:** A reduced active-site cluster (~60-90 atoms) represents local electrostatic and hydrogen-bonding energetics during nucleophilic attack.
- **Limitation:** Cluster models do NOT capture whole-protein conformational entropy, long-range allosteric dynamics, or full macroscopic solvent reorganization free energies ($\Delta G^\ddagger$). Single-point electronic energies serve as mechanistic proxies, not absolute kinetic rates.

## 3. Quantum VQE Scope & No-Advantage Boundary
- **Boundary:** Small active spaces (e.g. $(4e, 4o)$ on 8 qubits) are computationally tractable for exact classical Full Configuration Interaction (CASCI).
- **No Claim of Quantum Advantage:** This work demonstrates quantum algorithm readiness, circuit depth scalability, and noise resilience. Parity against exact CASCI is mandatory to validate the qubit Hamiltonian mapping.

## 4. Prior-Art & Experimental Disclaimer
- **Boundary:** Computational ranking prioritises testable hypotheses. Without prospective wet-lab assays, no newly proposed variant can be claimed to be experimentally proven.
- **Known Controls:** FAST-PETase (N233K, S121E, D186H, R224Q, R280A) and DuraPETase are tracked as positive benchmark controls.
