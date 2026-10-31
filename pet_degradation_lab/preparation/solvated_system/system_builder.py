"""Solvated Explicit-Solvent System Builder for PET Degradation Lab.

Constructs:
- Solvated Apo and Holo (BHET-bound) complexes for:
  1. WT IsPETase (5XJH)
  2. FAST-PETase (7SH6)
  3. VAR-ENG-004 (Engineered Q-Catalyst variant)
- Adds explicit TIP3P water box with 10.0 Å (1.0 nm) buffer padding
- Neutralizes net formal charge with counterions
- Adjusts NaCl ionic strength to 0.15 M (150 mM physiological salt concentration)
- Applies PME (Particle Mesh Ewald) with 1.0 nm cutoff and HBonds constraints
- Tests potential energy convergence (relaxation check)
- Serializes OpenMM System, Integrator, and PDB coordinate files
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import openmm
import openmm.app as app
from openmm import unit


@dataclass
class SolvatedSystemMetrics:
    system_id: str
    is_holo: bool
    num_solute_atoms: int
    num_waters: int
    num_na_ions: int
    num_cl_ions: int
    total_particles: int
    net_charge: float
    ionic_strength_M: float
    box_dimensions_nm: List[float]
    potential_energy_kj_mol: float
    is_stable: bool
    pdb_path: str
    system_xml_path: str
    integrator_xml_path: str
    summary: str


class SolvatedSystemBuilder:
    """Builds explicit solvent systems with TIP3P water, 150 mM NaCl, and serialized topologies."""

    def __init__(
        self,
        ff_xml_path: Optional[Path] = None,
        padding_nm: float = 1.0,
        ionic_strength_M: float = 0.15,
        temperature_K: float = 300.0,
    ):
        self.padding = padding_nm * unit.nanometers
        self.ionic_strength = ionic_strength_M * unit.molar
        self.temperature = temperature_K * unit.kelvin
        if ff_xml_path is None:
            ff_xml_path = Path(__file__).parent.parent / "forcefield" / "pet_gaff2_resp.xml"
        self.ff_xml_path = ff_xml_path
        self.forcefield = app.ForceField(
            "amber14/protein.ff14SB.xml",
            "amber14/tip3p.xml",
            str(self.ff_xml_path),
        )

    def build_system(
        self,
        protein_pdb_path: Path,
        system_id: str,
        output_dir: Path,
        ligand_pdb_path: Optional[Path] = None,
        max_minimization_steps: int = 100,
    ) -> SolvatedSystemMetrics:
        """Constructs, solvates, neutralizes, and serializes an OpenMM system."""
        output_dir.mkdir(parents=True, exist_ok=True)

        prot_pdb = app.PDBFile(str(protein_pdb_path))
        modeller = app.Modeller(prot_pdb.topology, prot_pdb.positions)

        is_holo = ligand_pdb_path is not None and ligand_pdb_path.exists()
        if is_holo:
            lig_pdb = app.PDBFile(str(ligand_pdb_path))
            modeller.add(lig_pdb.topology, lig_pdb.positions)

        num_solute_atoms = modeller.topology.getNumAtoms()

        # Solvate with TIP3P, neutralize, and add 150 mM NaCl
        modeller.addSolvent(
            self.forcefield,
            model="tip3p",
            padding=self.padding,
            ionicStrength=self.ionic_strength,
            neutralize=True,
        )

        total_particles = modeller.topology.getNumAtoms()

        # Count waters and ions
        num_waters = 0
        num_na = 0
        num_cl = 0
        for r in modeller.topology.residues():
            if r.name in ["HOH", "WAT"]:
                num_waters += 1
            elif r.name in ["NA", "Na+"]:
                num_na += 1
            elif r.name in ["CL", "Cl-"]:
                num_cl += 1

        # Periodic box dimensions
        box_vectors = modeller.topology.getPeriodicBoxVectors()
        box_dims = [
            box_vectors[0][0].value_in_unit(unit.nanometers),
            box_vectors[1][1].value_in_unit(unit.nanometers),
            box_vectors[2][2].value_in_unit(unit.nanometers),
        ]

        # Create OpenMM System with PME
        system = self.forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
        )

        # Verify net charge from particles
        nb_force = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
        net_charge = 0.0
        for i in range(system.getNumParticles()):
            q, sig, eps = nb_force.getParticleParameters(i)
            net_charge += q.value_in_unit(unit.elementary_charge)
        net_charge = round(net_charge, 4)

        # Integrator: LangevinMiddleIntegrator for standard NVT/NPT test relaxation
        integrator = openmm.LangevinMiddleIntegrator(
            self.temperature,
            1.0 / unit.picoseconds,
            0.002 * unit.picoseconds,
        )

        # Context and brief minimization to verify finite potential energy
        simulation = app.Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        simulation.minimizeEnergy(maxIterations=max_minimization_steps)
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        pot_energy_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        is_stable = not np.isnan(pot_energy_kj) and (pot_energy_kj < 0.0)

        # Save coordinates
        pdb_path = output_dir / f"{system_id}_solvated.pdb"
        with open(pdb_path, "w") as f:
            app.PDBFile.writeFile(modeller.topology, state.getPositions(), f)

        # Serialize OpenMM System and Integrator XMLs
        system_xml_path = output_dir / f"{system_id}_system.xml"
        with open(system_xml_path, "w") as f:
            f.write(openmm.XmlSerializer.serialize(system))

        integrator_xml_path = output_dir / f"{system_id}_integrator.xml"
        with open(integrator_xml_path, "w") as f:
            f.write(openmm.XmlSerializer.serialize(integrator))

        summary = (
            f"System {system_id} built ({'Holo' if is_holo else 'Apo'}): {total_particles} particles "
            f"({num_solute_atoms} solute, {num_waters} waters, {num_na} Na+, {num_cl} Cl-), "
            f"Box: {box_dims[0]:.2f}x{box_dims[1]:.2f}x{box_dims[2]:.2f} nm, "
            f"Net Charge: {net_charge:.2f} e, Potential Energy: {pot_energy_kj:.1f} kJ/mol"
        )

        metrics = SolvatedSystemMetrics(
            system_id=system_id,
            is_holo=is_holo,
            num_solute_atoms=num_solute_atoms,
            num_waters=num_waters,
            num_na_ions=num_na,
            num_cl_ions=num_cl,
            total_particles=total_particles,
            net_charge=net_charge,
            ionic_strength_M=0.15,
            box_dimensions_nm=box_dims,
            potential_energy_kj_mol=round(pot_energy_kj, 2),
            is_stable=is_stable,
            pdb_path=str(pdb_path),
            system_xml_path=str(system_xml_path),
            integrator_xml_path=str(integrator_xml_path),
            summary=summary,
        )

        metadata_path = output_dir / f"{system_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

        return metrics
