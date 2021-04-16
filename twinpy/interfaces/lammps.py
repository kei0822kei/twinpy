#!/usr/bin/env python

"""
Interfaces for lammps.
"""

import numpy as np
from phonolammps import PhonoLammps
from phonopy import Phonopy
from lammpkits.static import LammpsStatic


def get_lammpkits_before_run(cell:tuple,
                             pair_style:str,
                             pair_coeff:str=None,
                             pot_file:str=None,
                             is_relax_lattice:bool=True) -> LammpsStatic:
    """
    Get lammpkits before run_lammps is called.

    Args:
        cell: Original cell.
        pair_style: Key 'pair_style' setting for lammps input.
        pair_coeff: Key 'pair_coeff' setting for lammps input.
        pot_file: Potential file path from potentials directory.
                  This setting becomes activated only when pair_coeff is None.
        is_relax_lattice: If True, lattice is relaxed.
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

    return lmp_stc


def get_relax_analyzer_from_lammps(cell:tuple,
                                   pair_style:str,
                                   pair_coeff:str=None,
                                   pot_file:str=None,
                                   is_relax_lattice:bool=True):
    """
    Get relax analyzer from lammps.

    Args:
        cell: Original cell.
        pair_style: Key 'pair_style' setting for lammps input.
        pair_coeff: Key 'pair_coeff' setting for lammps input.
        pot_file: Potential file path from potentials directory.
                  This setting becomes activated only when pair_coeff is None.
        is_relax_lattice: If True, lattice is relaxed.

    Todo:
        Consider standardization (original cell).
    """
    from twinpy.analysis.relax_analyzer import RelaxAnalyzer

    lmp_stc = get_lammpkits_before_run(cell=cell,
                                       pair_style=pair_style,
                                       pair_coeff=pair_coeff,
                                       pot_file=pot_file,
                                       is_relax_lattice=is_relax_lattice)
    initial_cell = lmp_stc.get_initial_cell()
    final_cell = lmp_stc.get_final_cell()
    forces = lmp_stc.get_forces()
    energy = lmp_stc.get_energy()
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
    phlammps = Phonolammps(strings,
                       supercell_matrix=supercell_matrix,
                       primitive_matrix=primitive_matrix)
    phonon = Phonopy(unitcell,
                     supercell_matrix)
    unitcell = phlammps.get_unitcell()
    force_constants = phlammps.get_force_constants()
    phonon = Phonopy(unitcell=unitcell,
                     supercell_matrix=supercell_matrix)
    phonon.set_force_constants(force_constants)

    return phonon


def get_phonon_analyzer_from_lammps(lammps_input:list,
                                    supercell_matrix:np.array=np.eye(3),
                                    primitive_matrix:np.array=np.eye(3),
                                    relax_analyzer=None):
    """
    Get PhononAnalyzer class object from PhonoLammps.
    """
    from twinpy.analysis.phonon_analyzer import PhononAnalyzer

    phonon = get_phonon_from_lammps(
            lammps_input=lammps_input,
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            )
    ph_analyzer = PhononAnalyzer(phonon=phonon,
                                 relax_analyzer=relax_analyzer)

    return ph_analyzer
