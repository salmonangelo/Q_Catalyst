# Q-Catalyst Project Report
**Date:** 2026-10-29
**Time:** 14:15 (Approx.)

---

## 1. Project Overview
The **Q-Catalyst** project is a sophisticated, multi-stream research effort focused on developing a context-aware, mechanism-informed pipeline for prioritising PETase (PET-degrading enzyme) variants. It aims to bridge the gap between high-throughput AI screening and computationally intensive quantum chemical validation.

## 2. Project Goals
- Build a reproducible data-to-ranking pipeline for PETase variants.
- Predict activity and thermostability using protein language models.
- Quantify predictive uncertainty to identify reliable predictions.
- Select a budget-limited subset of candidates for classical chemistry.
- Compare classical chemistry descriptors with an AI-only baseline.
- Validate VQE against an exact classical reference for the same active-space Hamiltonian.
- Measure ranking quality, uncertainty calibration, compute cost, and the added value of chemistry.

## 3. High-level Architecture
The project is divided into four distinct workstreams:
- **Role 1 (Protein ML Lead):** Data curation, protein AI, uncertainty quantification, and AI-only benchmark arms.
- **Role 2 (Computational Chemistry Lead):** Structure preparation, PySCF electronic structure calculations, and classical CASCI references.
- **Role 3 (Quantum Lead):** Active-space selection, VQE implementation, and noise analysis.
- **Role 4 (Integration, UI and Pitch Lead):** Repository management, acquisition gate orchestration, evidence fusion, and final delivery (Dashboard).

## 4. Key Scientific Components
- **Chemistry Engine (`pyscf_engine.py`):** Performs RHF, DFT, and CASCI calculations using PySCF, supporting QM/MM electrostatic embedding.
- **Uncertainty Validation (`gate.py`):** Implements "Gate 0," a rigorous validation step using Spearman rank correlation and bootstrap confidence intervals to confirm uncertainty signals correlate with model errors before they are used for acquisition.
- **Protein AI (`esm_scoring.py`):** Provides modular zero-shot mutation scoring using ESM-1v with masked-marginal likelihood ratio scoring, including offline-cached weights and deterministic mocks for CI.

## 5. Scientific Integrity & Boundaries
- No claim of quantum advantage (small active spaces).
- No whole-protein VQE (focused on reduced clusters).
- No automatic trust in agent debate (evidence-based review only).
- Failures/missing results are not zero-valued.

## 6. Project Layout
The repository is structured to separate inputs, source code, and artifacts:
- `src/qcatalyst/`: Core logic organized by scientific stream (protein_ai, chemistry, quantum, uncertainty, acquisition, etc.).
- `data/`: Curated datasets and raw inputs.
- `configs/`: Shared project/stream parameters.
- `reports/`: Generated technical reports and documentation.
