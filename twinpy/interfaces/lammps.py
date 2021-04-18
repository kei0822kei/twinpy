#!/usr/bin/env python

"""
Interfaces for lammps.
"""

import numpy as np
from phonopy import Phonopy
from phonolammps import Phonolammps
from lammpkits.lammps.static import LammpsStatic


def get_lammps_relax(cell:tuple,
                     pair_style:str,
                     pair_coeff:str=None,
                     pot_file:str=None,
                     is_relax_lattice:bool=True,
                     run_lammps:bool=False) -> LammpsStatic:
    """
    Get lammpkits before run_lammps is called.

    Args:
        cell: Original cell.
        pair_style: Key 'pair_style' setting for lammps input.
        pair_coeff: Key 'pair_coeff' setting for lammps input.
        pot_file: Potential file path from potentials directory.
                  This setting becomes activated only when pair_coeff is None.
        is_relax_lattice: If True, lattice is relaxed.
        run_lammps: If True, run lamms.
    """
    lmp_stc = LammpsStatic()
    lmp_stc.add_structure(cell=cell)
    if pair_coeff is None:
        lmp_stc.add_potential_from_string(pair_style=pair_style,
                                          pair_coeff=pair_coeff)
    else:
        lmp_stc.add_potential_from_database(pair_style=pair_style,
                                            pot_file=potfile)
    lmp_stc.add_variables(add_energy=True)
    lmp_stc.add_relax_settings(is_relax_lattice=is_relax_lattice)
    if run_lammps:
        lmp_stc.run_lammps()

    return lmp_stc


def get_relax_analyzer_from_lammps_relax(lammps_static:LammpsStatic):
    """
    Get relax analyzer from lammps.

    Args:
        lammps_static: LammpsStatic class object.
    """
    from twinpy.analysis.relax_analyzer import RelaxAnalyzer

    if not lammps_static.is_run_finished:
        lammps_static.run_lammps()
    initial_cell = lammps_static.get_initial_cell()
    final_cell = lammps_static.get_final_cell()
    forces = lammps_static.get_forces()
    energy = lammps_static.get_energy()
    rlx_analyzer = RelaxAnalyzer(
            initial_cell=initial_cell,
            final_cell=final_cell,
            forces=forces,
            energy=energy,
            )

    return rlx_analyzer


def get_phonon_from_lammps(lammps_input:list,
                           supercell_matrix:np.array=np.eye(3),
                           primitive_matrix:np.array=np.eye(3)) -> Phonopy:
    """
    Get Phonopy class object from PhonoLammps.
    """
    phlammps = Phonolammps(lammps_input=lammps_input,
                           supercell_matrix=supercell_matrix,
                           primitive_matrix=primitive_matrix)
    unitcell = phlammps.get_unitcell()
    force_constants = phlammps.get_force_constants()
    phonon = Phonopy(unitcell=unitcell,
                     supercell_matrix=supercell_matrix)
    phonon.set_force_constants(force_constants)

    return phonon
