# Q-Catalyst: PET Degradation Lab (`pet_degradation_lab/`)

**Independent Research Module for Molecular Dynamics, Substrate Interaction, and Reaction-Energy Analysis of PETase Variants**

---

## 1. Objective and Scientific Scope

The **PET Degradation Lab** is an independent research module created within the Q-Catalyst repository. Its overarching mission is to investigate whether shortlisted PETase enzyme variants improve poly(ethylene terephthalate) (PET) plastic degradation through:
- Explicit-solvent molecular dynamics (MD) simulations.
- Quantitative protein–ligand and protein–polymer interaction profiling.
- Transition-state activation and reaction-energy barrier calculations (QM/MM).
- Retrospective and prospective experimental kinetic benchmarking ($k_{\text{cat}}, K_M$).
- Standalone 3D molecular and trajectory visualization using Mol*.

### Non-Negotiable Isolation Principles
1. **Complete File Isolation:** All code, configs, tests, snapshots, and visualization assets reside strictly inside `pet_degradation_lab/`.
2. **Zero Modification to Parent Project:** No existing Q-Catalyst source code, tests, configs, databases, API endpoints, or frontends are touched.
3. **Zero Module Imports:** Never imports from `src/qcatalyst/` or any parent Q-Catalyst modules (`test_import_isolation.py` enforces this).
4. **Independent Python Environment:** Dedicated virtual environment targeting Python 3.12 (`pet_degradation_lab/.venv`).
5. **Read-Only Snapshots:** Original Q-Catalyst scientific outputs are ingested via immutable, SHA-256 verified read-only snapshots.
6. **Scientific Non-Invention:** Missing values, uncomputed structures, and unrun simulations remain explicitly `NOT_AVAILABLE`. No dummy numbers or fabricated degradation rates are created.

---

## 2. Directory Structure

```text
pet_degradation_lab/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   └── lab_config.yaml                      # Configuration, allowlists, and landmarks
├── data/
│   ├── project_snapshots/                   # Immutable snapshot store
│   │   └── snap_baseline_v1/                # Initial validated snapshot
│   │       ├── snapshot_manifest.json
│   │       ├── candidate_manifest.json
│   │       ├── sequence_manifest.json
│   │       ├── structure_manifest.json
│   │       ├── structural_risk_manifest.json
│   │       ├── dft_manifest.json
│   │       ├── quantum_manifest.json
│   │       └── raw/                         # Exact copies of source files
│   ├── benchmark_variants/                  # Future reference variant library
│   ├── structures/                          # Structure staging
│   └── experimental_references/             # Literature kinetic benchmarks
├── schemas/
│   ├── evidence_status.py                   # EXPERIMENTAL, COMPUTED, PREDICTED, REPORTED
│   ├── provenance.py                        # Full provenance metadata and SHA-256 tracking
│   ├── sequence.py                          # Amino-acid validation, WT alignment, indels
│   ├── structure.py                         # Sequence pos vs PDB residue numbering
│   ├── structural_risk.py                   # 3D proximity risk vs heuristics
│   ├── dft.py                               # PySCF DFT electronic energy and descriptors
│   ├── quantum.py                           # Active-space CASCI/VQE simulation metrics
│   └── candidate.py                         # Master CandidateVariant model
├── ingestion/
│   ├── inventory.py                         # Source inventory auditor
│   ├── snapshot_adapter.py                  # ProjectSnapshotAdapter with immutability
│   ├── sequence_importer.py                 # Sequence extraction and alignment
│   ├── structure_importer.py                # 3D coordinate and residue parsing
│   ├── chemistry_importer.py                # DFT and quantum ingestion
│   └── validation.py                        # Scientific integrity and anti-fabrication rules
├── preparation/                             # Milestone 2: Structure, forcefield, solvated box
├── simulation/                              # Milestone 2: OpenMM, GROMACS, PLUMED
├── analysis/                                # Milestone 2: RMSD, RMSF, contacts, free energy
├── reaction_chemistry/                      # Milestone 3: QM/MM reaction pathways (CP2K)
├── benchmarking/                            # Milestone 3: Experimental kinetic validation
├── visualization/
│   ├── app/
│   │   ├── index.html                       # Standalone 5-view UI with embedded PDBe Mol*
│   │   └── server.py                        # Standalone Python HTTP server & REST API
│   ├── molstar/                             # Mol* assets
│   ├── structures/                          # Local PDB rendering cache
│   └── charts/                              # Trajectory and energy plots
├── reports/
│   ├── source_inventory.md                  # Comprehensive audit of 20 Q-Catalyst artifacts
│   ├── candidate_reports/                   # Per-candidate dossiers
│   └── benchmark_reports/                   # Systematic comparative reports
├── scripts/
│   ├── inspect_project.py                   # Source inventory CLI
│   ├── create_snapshot.py                   # Snapshot creation CLI (--dry-run, --snapshot-id)
│   └── launch_ui.py                         # Standalone UI runner (--port 8501)
└── tests/
    ├── fixtures/                            # Synthetic minimal test fixtures
    ├── test_snapshot_adapter.py             # Immutability, checksums, dry-run
    ├── test_provenance.py                   # Evidence classifications
    ├── test_sequence_validation.py          # Amino-acid alphabet, indels, substitutions
    ├── test_residue_mapping.py              # Sequence pos vs PDB resis, triad detection
    └── test_import_isolation.py             # Zero qcatalyst imports, non-invention rule
```

---

## 3. Environment Setup & Installation

### Linux / macOS
```bash
# Navigate to module directory
cd pet_degradation_lab

# Create independent Python 3.12 virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install module in editable mode with development dependencies
uv pip install -e ".[dev]"
```

### Windows (PowerShell)
```powershell
cd pet_degradation_lab

# Target Python 3.12 launcher
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 4. Key CLI Commands

### 1. Audit Existing Sources
Inspects the parent Q-Catalyst repository in read-only mode and generates `reports/source_inventory.md`:
```bash
python scripts/inspect_project.py --source-root ..
```

### 2. Snapshot Dry Run
Shows planned file copies, checksum calculations, and missing sources without modifying files:
```bash
python scripts/create_snapshot.py --dry-run
```

### 3. Create Immutable Snapshot
Copies source files to `data/project_snapshots/<snapshot_id>/`, computes SHA-256 hashes, and writes manifests:
```bash
python scripts/create_snapshot.py --snapshot-id snap_baseline_v1
```
*(Attempting to overwrite an existing snapshot ID safely raises `FileExistsError`)*.

### 4. Run Independent Test Suite
Executes the isolated test suite using synthetic fixtures:
```bash
pytest tests/ -v
```

### 5. Launch Standalone Visualisation Explorer
Starts the zero-dependency Web dashboard with embedded PDBe Mol* 3D viewer:
```bash
python scripts/launch_ui.py --port 8501
```
Then navigate to `http://127.0.0.1:8501/` in your browser.

---

## 5. Standalone Visualisation Interface (5 Views)

The standalone interface is served by `visualization/app/server.py` and rendered by `visualization/app/index.html`:
- **View 1: Candidate Inventory:** Summary cards and table of all 26 shortlisted variants, showing mutation notation, sequence availability, PDB structure availability, and DFT status.
- **View 2: Candidate Evidence:** Detailed evidence dossier for any selected variant showing AI prediction (XGBoost), uncertainty $U(x)$, Gate 0 status, Tier S DFT descriptors, quantum VQE parity, and complete provenance.
- **View 3: 3D Structure Explorer:** Interactive PDBe Mol* 3D WebGL viewer loading local PDB crystal structures (5XJH native IsPETase, 7SH6 FAST-PETase) and candidate mutant models directly from snapshot data.
- **View 4: Simulation Readiness:** Multi-point pre-flight checklist identifying required vs pending inputs for molecular dynamics and QM/MM.
- **View 5: Future Comparisons:** Standardised comparative dashboards (RMSD, RMSF, contact analysis, reaction barriers) strictly marked **NOT YET CALCULATED**.

---

## 6. Required Literature & Simulation References

1. **Colizzi et al. (PETase_Cryst-vs-Amor):**
   - GitHub: [https://github.com/fcolizzi/PETase_Cryst-vs-Amor](https://github.com/fcolizzi/PETase_Cryst-vs-Amor)
   - PLUMED-NEST: [https://www.plumed-nest.org/eggs/25/029/](https://www.plumed-nest.org/eggs/25/029/)
   - Focus: Well-tempered metadynamics, HREX, and amorphous vs crystalline PET chain extraction. Atom indices must be remapped to the actual topology being simulated.
2. **Wolber Lab (OpenMMDL):**
   - GitHub: [https://github.com/wolberlab/OpenMMDL](https://github.com/wolberlab/OpenMMDL)
   - Focus: Protein–ligand explicit solvent simulation workflows and interaction analysis (PLIP, ProLIF). Note that PET is a polymer requiring custom torsion parameters.
3. **OpenMM Simulation Toolchain:**
   - Documentation: [https://docs.openmm.org/latest/userguide/application/02_running_sims.html](https://docs.openmm.org/latest/userguide/application/02_running_sims.html)
   - Focus: System creation, Langevin middle integration, PME electrostatics, and trajectory logging.
4. **MDAnalysis:**
   - User Guide: [https://github.com/MDAnalysis/UserGuide](https://github.com/MDAnalysis/UserGuide)
   - Focus: Backbone RMSD alignment, per-residue RMSF flexibilities, and distance distributions.
5. **PDBe Mol* / Molstar:**
   - GitHub: [https://github.com/molstar/pdbe-molstar](https://github.com/molstar/pdbe-molstar)
   - Focus: Lightweight WebGL molecular component for coordinate inspection and future trajectory playback.
6. **CP2K QM/MM:**
   - Documentation: [https://github.com/cp2k/cp2k/blob/master/docs/methods/qm_mm/builtin.md](https://github.com/cp2k/cp2k/blob/master/docs/methods/qm_mm/builtin.md)
   - Focus: QM/MM partitioning for enzyme active sites.

---

## 7. Deliverables & Roadmap Status

### Completed in Milestone 1:
- [x] Full module architecture inside `pet_degradation_lab/`.
- [x] 100% isolation verified (zero edits to existing Q-Catalyst code or configs).
- [x] Independent Python 3.12 virtual environment configured and tested.
- [x] Comprehensive source inventory report generated ([`reports/source_inventory.md`](file:///home/luckycelestial/Pavithran/QCatalyst/Q_Catalyst/pet_degradation_lab/reports/source_inventory.md)).
- [x] Read-only snapshot adapter with SHA-256 verification and immutability protection.
- [x] Initial baseline snapshot created ([`snap_baseline_v1`](file:///home/luckycelestial/Pavithran/QCatalyst/Q_Catalyst/pet_degradation_lab/data/project_snapshots/snap_baseline_v1/)).
- [x] Pydantic data contracts for candidate, sequence, structure, structural risk, DFT, quantum, and provenance.
- [x] Standalone 5-view Web interface with embedded Mol* 3D viewer.
- [x] 16 independent pytest unit and isolation tests passing.

### Completed in Milestone 2:
- [x] Sequence-to-structure, chain ID, and residue numbering validation for 5XJH, 7SH6, and VAR-ENG-004 ([`preparation/sequence_mapping/mapping_validator.py`](file:///home/luckycelestial/Pavithran/QCatalyst/Q_Catalyst/pet_degradation_lab/preparation/sequence_mapping/mapping_validator.py)).
- [x] Quantitative catalytic triad (Ser160-His237-Asp206) and disulfide bridge (Cys203-Cys239, Cys273-Cys289) verification.
- [x] Protein preparation pipeline resolving terminal capping ($OXT$), missing heavy atoms, and non-water heteroatoms ([`preparation/protein_preparer.py`](file:///home/luckycelestial/Pavithran/QCatalyst/Q_Catalyst/pet_degradation_lab/preparation/protein_preparer.py)).
- [x] Chemical protonation analysis: His237 assigned as delta-protonated `HID` at pH 7.5; non-catalytic histidines systematically assigned.
- [x] Substrate model investigation (MHET vs BHET vs 2-PET) and active-site Michaelis alignment at $2.95\text{ \AA}$ ([`preparation/pet_substrate/substrate_builder.py`](file:///home/luckycelestial/Pavithran/QCatalyst/Q_Catalyst/pet_degradation_lab/preparation/pet_substrate/substrate_builder.py)).
- [x] GAFF2/RESP force-field parameterization with refined ester-aryl planar conjugation ($k_1=20.08\text{ kJ/mol}$) and ethylene glycol gauche effect dihedrals ([`preparation/forcefield/pet_gaff2_resp.xml`](file:///home/luckycelestial/Pavithran/QCatalyst/Q_Catalyst/pet_degradation_lab/preparation/forcefield/pet_gaff2_resp.xml)).
- [x] Force-field parameter validation (1-4 scaling factors, net charge = 0.000000 e, OpenMM compatibility) ([`preparation/forcefield/parameter_validator.py`](file:///home/luckycelestial/Pavithran/QCatalyst/Q_Catalyst/pet_degradation_lab/preparation/forcefield/parameter_validator.py)).
- [x] Explicit-solvent system builder with TIP3P water, 150 mM NaCl counterions, and serialized OpenMM `system.xml` and `integrator.xml` ([`preparation/solvated_system/system_builder.py`](file:///home/luckycelestial/Pavithran/QCatalyst/Q_Catalyst/pet_degradation_lab/preparation/solvated_system/system_builder.py)).
- [x] End-to-end reproducible CLI pipeline ([`scripts/prepare_systems.py`](file:///home/luckycelestial/Pavithran/QCatalyst/Q_Catalyst/pet_degradation_lab/scripts/prepare_systems.py)).
- [x] Independent test suite expanded to 23 passing tests ([`tests/test_milestone2_systems.py`](file:///home/luckycelestial/Pavithran/QCatalyst/Q_Catalyst/pet_degradation_lab/tests/test_milestone2_systems.py)).
- [x] Comprehensive simulation-readiness report ([`reports/milestone2_simulation_readiness.md`](file:///home/luckycelestial/Pavithran/QCatalyst/Q_Catalyst/pet_degradation_lab/reports/milestone2_simulation_readiness.md)).

### Planned for Milestone 3:
1. Multi-replica NVT/NPT equilibration protocols (restrained solute warmup, barostat coupling).
2. Production OpenMM molecular dynamics simulations on top candidates, FAST-PETase, and WT IsPETase.
3. Trajectory analysis: backbone RMSD, active-site loop RMSF, substrate contacts, and oxyanion hole stability.
4. QM/MM reaction barrier calculations for acylation transition state.
