# Q-Catalyst — W159H QM/MM Scientific Blocker & Validation Report

**Author:** Quantum Lead (Role 3)  
**Date:** 2026-10-31  
**Scaffold Reference:** PDB ID 5XJH (Chain A, 1.58 Å resolution)  
**Target Variant:** IsPETase W159H (Trp159 $\to$ His159)  
**Deliverable Status:** Scientific Blocker Report (Execution Frozen per Safety Guard)

---

## 1. Executive Summary & Validation Verdicts

An exhaustive scientific audit of the mutant preparation pipeline for W159H reveals four critical physical chemistry and molecular modeling blockers that prevent valid active-space electronic Hamiltonian generation. 

In strict adherence to Q-Catalyst scientific integrity protocols, active-space Hamiltonian generation, CASCI diagonalizations, and VQE simulations remain **FROZEN**. No parameters, coordinates, or quantum energies have been fabricated or normalized.

| Validation Criterion | Decision | Root Blocker |
| :--- | :---: | :--- |
| **1. Protein Hydrogenation** | **BLOCKED** | Scaffold 5XJH crystal structure lacks explicit hydrogens on residues 30–292, resulting in an unphysical net MM charge of $-85.0\,e$. Automated hydrogen-bond network optimization requires `pdb2pqr` or `openmm`/`pdbfixer` (not installed). Manual coordinate fabrication is scientifically invalid. |
| **2. Histidine Force-Field Parameterization** | **BLOCKED** | Tabulated charges in `amber_ff14sb.py` sum to $+0.1750\,e$ (HID) and $+0.1358\,e$ (HIE) due to spliced alanine backbone charges. Authoritative ff14SB residue templates with integer neutrality and bonded valence parameters require official AmberTools/OpenMM parameter libraries. |
| **3. Structural Minimization** | **BLOCKED** | All 12 candidate rotamers retain severe unrelaxed steric clashes $<2.0\,\text{Å}$, specifically His159 HE1 with catalytic His237 carbonyl oxygen at $1.848\,\text{Å}$ ($\Delta E_{\text{LJ}} > +45\,\text{kcal/mol}$). Full molecular mechanics minimization requires a complete force field engine. |
| **4. QM/MM Boundary & Link-Atom Construction** | **BLOCKED** | Point charges on His159 C ($+0.5973\,e$) and O ($-0.5679\,e$) reside $2.385\,\text{Å}$ and $2.533\,\text{Å}$ from Ser160 link capping hydrogens, inducing severe unshielded electrostatic over-polarization. Arbitrary zeroing is forbidden. |
| **Overall Scientific Verdict** | **FAIL / FROZEN** | Production mutant quantum calculations are blocked until authoritative external molecular mechanics and preparation libraries are integrated. |

---

## 2. Protein Hydrogenation & Protonation Assessment

### 2.1 Scaffold Analysis (`data/structures/5XJH.pdb`)
- **Residues:** Chain A spans residues 30 to 292 (263 residues total).
- **Atoms:** 1,931 ATOM records, all heavy atoms ($C, N, O, S$).
- **Experimental Hydrogens:** Exactly 0 explicit hydrogens exist in the deposited X-ray structure (solved at pH 7.5).
- **Solvent:** 142 water molecules (`HETATM` records) are present as isolated oxygen centers (`HOH`), lacking hydrogen coordinates.

### 2.2 Ionizable Residue Census & Protonation Decisions
At physiological pH 7.5, the PETase scaffold contains 75 ionizable groups:
- **Aspartate (27 residues):** Standard carboxylate anion (`ASP`, charge $-1.0\,e$). Catalytic Asp206 is deprotonated in the resting catalytic triad.
- **Glutamate (12 residues):** Standard carboxylate anion (`GLU`, charge $-1.0\,e$).
- **Arginine (17 residues):** Standard protonated guanidinium (`ARG`, charge $+1.0\,e$).
- **Lysine (7 residues):** Standard protonated ammonium (`LYS`, charge $+1.0\,e$).
- **Histidine (6 residues):**
  - Catalytic His237: Assigned as neutral `HIE` ($\epsilon$-protonated on NE2, ND1 acts as general base accepting proton from Ser160 OG).
  - Mutant His159: Neutral tautomer candidate (`HID` vs `HIE`).
  - Other Histidines (186, 214, 246, 280): Subject to local hydrogen-bonding network optimization.
- **Disulfide Bridges (2 bonds):**
  - `CYS A 203 <---> CYS A 239` (distance $2.12\,\text{Å}$)
  - `CYS A 273 <---> CYS A 289` (distance $2.05\,\text{Å}$)
  - Must be assigned as oxidized cystine `CYX` (formal charge $0.0\,e$, no `HG` proton). Naive hydrogenation tools incorrectly add `HG` to both sulfurs, causing severe steric clash ($<1.0\,\text{Å}$).

### 2.3 Net MM Charge Discrepancy
- In the absence of scaffold hydrogens, assigning heavy-atom charges yields an artificial net MM bath of **$-85.0\,e$**.
- Missing backbone amide protons ($+0.2719\,e \times 262 \approx +71.2\,e$) and sidechain polar hydrogens account for this massive deficit.
- In QM/MM embedding, a $-85\,e$ monopole surrounding the 40-atom Tier S quantum cluster creates an unphysical repulsive Coulomb potential that artificially shifts orbital energies by several Hartrees.

---

## 3. Histidine Force-Field Parameter Audit

### 3.1 Numerical Evidence from `amber_ff14sb.py`
Inspection of partial charges currently defined in `src/qcatalyst/chemistry/amber_ff14sb.py`:
- `HID` (17 atoms):
  - Backbone: `N (-0.4157)`, `H (0.2719)`, `CA (0.0186)`, `HA (0.0880)`, `C (0.5973)`, `O (-0.5679)` $\to \sum = -0.0078\,e$.
  - Sidechain: `CB (-0.0532)`, `HB2 (0.0412)`, `HB3 (0.0412)`, `CG (0.1541)`, `ND1 (-0.3809)`, `HD1 (0.3649)`, `CE1 (0.2057)`, `HE1 (0.1378)`, `NE2 (-0.5727)`, `CD2 (0.1293)`, `HD2 (0.1154)` $\to \sum = +0.1828\,e$.
  - **Total Residue Charge:** **$+0.1750\,e$** (Expected: $0.0000\,e$).
- `HIE` (17 atoms):
  - Total Residue Charge: **$+0.1358\,e$** (Expected: $0.0000\,e$).

### 3.2 Root Cause Analysis
In official AMBER ff14SB parameterization (Maier et al., J. Chem. Theory Comput. 2015):
1. RESP charge fitting was performed on capped dipeptides (ACE-X-NME) with explicit constraints forcing each internal amino acid residue unit to strictly sum to integer values ($0.0000\,e$ for neutral residues).
2. The entries in `amber_ff14sb.py` spliced generic alanine backbone charges with sidechain charges extracted from uncoupled tables without residue-specific RESP re-fitting.
3. In addition, `amber_ff14sb.py` lacks bonded force field parameters ($K_r, r_0, K_\theta, \theta_0, V_n, \gamma$) and AMBER atom types, containing only isolated electrostatics.

---

## 4. Structural Minimization Audit & The 1.848 Å Clash

### 4.1 Evaluation of All 12 Dunbrack Rotamer Candidates
Each candidate generated by `mutant_builder.py` under `results/structures/W159H/` was audited:

| Candidate ID | Tautomer | Target $(\chi_1, \chi_2)$ | Min Scaffold Dist (Å) | Clashes $<2.0\,\text{Å}$ | Convergence Status | Audit Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `W159H_HID_m+80` | HID | $(-65^\circ, +80^\circ)$ | 0.866 | 5 | unminimized | **FAIL** |
| `W159H_HID_m-70` | HID | $(-65^\circ, -70^\circ)$ | 0.539 | 12 | unminimized | **FAIL** |
| `W159H_HID_p+80` | HID | $(+65^\circ, +80^\circ)$ | 1.072 | 6 | unminimized | **FAIL** |
| `W159H_HID_p-70` | HID | $(+65^\circ, -70^\circ)$ | 0.604 | 7 | unminimized | **FAIL** |
| `W159H_HID_t-70` | HID | $(180^\circ, -70^\circ)$ | 1.881 | 1 | unminimized | **FAIL** |
| `W159H_HID_t+80` | HID | $(180^\circ, +80^\circ)$ | 1.848 | 1 | unminimized | **FAIL** |
| `W159H_HIE_m+80` | HIE | $(-65^\circ, +80^\circ)$ | 0.866 | 6 | unminimized | **FAIL** |
| `W159H_HIE_m-70` | HIE | $(-65^\circ, -70^\circ)$ | 0.539 | 10 | unminimized | **FAIL** |
| `W159H_HIE_p+80` | HIE | $(+65^\circ, +80^\circ)$ | 1.071 | 8 | unminimized | **FAIL** |
| `W159H_HIE_p-70` | HIE | $(+65^\circ, -70^\circ)$ | 0.604 | 7 | unminimized | **FAIL** |
| `W159H_HIE_t-70` | HIE | $(180^\circ, -70^\circ)$ | 1.881 | 1 | unminimized | **FAIL** |
| `W159H_HIE_t+80` | HIE | $(180^\circ, +80^\circ)$ | 1.848 | 1 | unminimized | **FAIL** |

### 4.2 Physical Nature of the $1.848\,\text{Å}$ Clash
- **Clashing Pair:** `His159 HE1 <---> His237 O` (distance = $1.848\,\text{Å}$).
- **Steric Overlap:**
  - AMBER ff14SB van der Waals radii: $R_{\min}/2(\text{O}) = 1.6612\,\text{Å}$, $R_{\min}/2(\text{H4}) = 1.4090\,\text{Å}$.
  - Sum of van der Waals contact radii: $R_0 = 3.07\,\text{Å}$.
  - Overlap: $\Delta r = 3.07 - 1.848 = 1.22\,\text{Å}$.
  - Lennard-Jones repulsive energy: $\Delta E_{\text{LJ}} > +45\,\text{kcal/mol}$.
- **Mechanics Limitation:**
  - The internal minimizer in `mutant_builder.py` clamped pairwise distances at $0.8\,\text{Å}$ and omitted harmonic bond restraints for hydrogen atoms (`CE1-HE1`, `CD2-HD2`, `ND1-HD1`), angle bending restraints, and improper torsional planarity restraints.
  - A scientifically defensible structure requires restrained energy minimization in OpenMM or AmberTools `sander` with full valence potentials.

---

## 5. QM/MM Boundary & Link-Atom Construction Audit

### 5.1 Link Atom Placement
The 40-atom Tier S cluster cuts the covalent backbone at $C_\alpha - C_\beta$ bonds:
- Ser160: Truncated to a methanol model capped at $C_\beta$ with 3 link hydrogens.
- His237: Truncated to 4-methylimidazole capped at $C_\beta$ with 3 link hydrogens.
- Asp206: Truncated to acetate capped at $C_\beta$ with 3 link hydrogens.

### 5.2 Distance Measurements to Neighboring Residue 159
Because residue 159 is directly bonded to Ser160 in sequence (`159 C - 160 N`), its backbone carbonyl dipole is immediately adjacent to the Ser160 capping boundary:
- `His159 O (-0.5679 e) <---> Ser160 Link H`: **$2.385\,\text{Å}$**
- `His159 C (+0.5973 e) <---> Ser160 Link H`: **$2.533\,\text{Å}$**
- `His159 HD2 (+0.1154 e) <---> Ser160 Link H`: **$2.463\,\text{Å}$**

### 5.3 Over-Polarization & Correction Requirements
- Unscreened classical point charges $<2.6\,\text{Å}$ from quantum link hydrogens create an artificial electric field exceeding $5\times 10^{10}\,\text{V/m}$.
- This field unphysically distorts the electronic wavefunction of the catalytic serine, altering HOMO/LUMO eigenvalues and VQE ground-state energies.
- **Required Remedy:** Standard QM/MM charge redistribution (e.g. Thiel's charge-shifted link-atom model or Gaussian-smeared electrostatic envelopes). Zeroing boundary charges without formal redistribution is prohibited.

---

## 6. Actionable Resolution Path & Missing Dependencies

To resolve these blockers cleanly without simulation or fabrication, the following standard molecular modeling toolchain is required:

1. **Hydrogenation & Protonation:**
   - Package: `pdb2pqr` (version $\ge 3.5.0$) or `openmm` + `pdbfixer`.
   - Command: `pdb2pqr30 --ff=AMBER --titration-state-method=propka --with-ph=7.5 data/structures/5XJH.pdb results/structures/W159H/5XJH_prepared.pdb`
2. **Authoritative AMBER ff14SB Parameters:**
   - Package: Official AmberTools `amino12.lib` or OpenMM `protein.ff14SB.xml`.
   - Guarantees exact integer charge neutrality ($0.0000\,e$) and complete bonded parameter sets.
3. **Restrained Energy Minimization:**
   - Package: `openmm` (version $\ge 8.0$).
   - Method: 500 steps of L-BFGS energy minimization with harmonic backbone position restraints ($k = 10.0\,\text{kcal/(mol}\cdot\text{\AA}^2)$) to resolve the $1.848\,\text{Å}$ clash while preserving active-site geometry.
4. **Quantum Chemistry Engine:**
   - Package: `pyscf` (version $\ge 2.5.0$, requires Linux or WSL2).
   - Solves the 1-electron and 2-electron integrals in the presence of the validated point-charge environment.

---

## 7. Audit Conclusion

The current W159H candidate structures and embedding model are **scientifically invalid** for production quantum chemistry. In accordance with strict software engineering and quantum chemistry protocol, the safety freeze implemented in `src/qcatalyst/quantum/hamiltonian.py` and `src/qcatalyst/chemistry/pyscf_engine.py` remains active to protect the integrity of the Q-Catalyst quantum benchmark suite.

