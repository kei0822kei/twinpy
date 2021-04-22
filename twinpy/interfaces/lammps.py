#!/usr/bin/env python

"""
Interfaces for lammps.
"""

import numpy as np
from phonopy import Phonopy
from phonolammps import Phonolammps
from lammpkits.lammps.static import LammpsStatic
from twinpy.structure.standardize import StandardizeCell
from twinpy.analysis.relax_analyzer import RelaxAnalyzer
from twinpy.analysis.phonon_analyzer import PhononAnalyzer
from twinpy.analysis.twinboundary_analyzer import TwinBoundaryAnalyzer



def get_lammps_relax(cell:tuple,
                     pair_style:str,
                     pair_coeff:str=None,
                     pot_file:str=None,
                     is_relax_lattice:bool=True,
                     run_lammps:bool=True) -> LammpsStatic:
    """
    Get lammpkits before run_lammps is called.

    Args:
        cell: Input cell.
        pair_style: Key 'pair_style' setting for lammps input.
        pair_coeff: Key 'pair_coeff' setting for lammps input.
        pot_file: Potential file path from potentials directory.
                  This setting becomes activated only when pair_coeff is None.
        is_relax_lattice: If True, lattice is relaxed.
        run_lammps: If True, run lamms.
    """
    lmp_stc = LammpsStatic()
    lmp_stc.add_structure(cell=cell)
    if pair_coeff:
        lmp_stc.add_potential_from_string(pair_style=pair_style,
                                          pair_coeff=pair_coeff)
    else:
        lmp_stc.add_potential_from_database(pair_style=pair_style,
                                            pot_file=pot_file)
    lmp_stc.add_thermo(thermo=100)
    lmp_stc.add_variables(add_energy=True,
                          add_stress=True)
    lmp_stc.add_relax_settings(is_relax_lattice=is_relax_lattice)
    if run_lammps:
        lmp_stc.run_lammps()

    return lmp_stc


def get_relax_analyzer_from_lammps_static(lammps_static:LammpsStatic,
                                          original_cell:tuple=None,
                                          no_standardize:bool=False):
    """
    Get relax analyzer from lammps.

    Args:
        lammps_static: LammpsStatic class object.
        original_cell: Original cell.
        no_standardize: See docstring in RelaxAnalyzer.
                        If no_standardize is True, input 'original_cell' is
                        ignored and original_cell and input_cell becomes
                        identical.
    """
    if not lammps_static.is_run_finished:
        lammps_static.run_lammps()
    initial_cell = lammps_static.get_initial_cell()
    final_cell = lammps_static.get_final_cell()
    forces = lammps_static.get_forces()
    energy = lammps_static.get_energy()
    rlx_analyzer = RelaxAnalyzer(
            initial_cell=initial_cell,
            final_cell=final_cell,
            original_cell=original_cell,
            forces=forces,
            energy=energy,
            no_standardize=no_standardize,
            )

    return rlx_analyzer


def get_phonon_analyzer_from_lammps_static(
        lammps_static:LammpsStatic,
        supercell_matrix:np.array,
        primitive_matrix:np.array=np.identity(3),
        original_cell:tuple=None,
        no_standardize:bool=False,
        ):
    """
    Get phonon analyzer from lammps.

    Args:
        lammps_static: LammpsStatic class object.
        original_cell: Original cell.
        no_standardize: See docstring in RelaxAnalyzer.
                        If no_standardize is True, input 'original_cell' is
                        ignored and original_cell and input_cell becomes
                        identical.
    """
    rlx_analyzer = get_relax_analyzer_from_lammps_static(
                       lammps_static=lammps_static,
                       original_cell=original_cell,
                       no_standardize=no_standardize,
                       )
    ph_lammps_input = lammps_static.get_lammps_input_for_phonolammps()
    ph_lmp = Phonolammps(lammps_input=ph_lammps_input,
                         supercell_matrix=supercell_matrix,
                         primitive_matrix=primitive_matrix)
    phonon = get_phonon_from_phonolammps(ph_lmp)
    ph_analyzer = PhononAnalyzer(phonon=phonon,
                                 relax_analyzer=rlx_analyzer)

    return ph_analyzer


def get_twinboundary_analyzer_from_lammps(
        twinboundary_structure,
        pair_style:str,
        supercell_matrix,
        pair_coeff:str=None,
        pot_file:str=None,
        is_relax_lattice:bool=True,
        move_atoms_into_unitcell:bool=True,
        no_standardize:bool=True,
        hexagonal_relax_analyzer:RelaxAnalyzer=None,
        hexagonal_phonon_analyzer:PhononAnalyzer=None,
        ):
    """
    Set twinboundary_analyzer from lammps.
    """
    def _get_phonon_analyzer(cell,
                             original_cell,
                             is_rlx_lat,
                             no_std,
                             sup_mat):
        lmp_stc = get_lammps_relax(
                      cell=_cell,
                      pair_style=pair_style,
                      pair_coeff=pair_coeff,
                      pot_file=pot_file,
                      is_relax_lattice=is_rlx_lat,
                      run_lammps=True,
                      )
        rlx_analyzer = get_relax_analyzer_from_lammps_static(
                           lammps_static=lmp_stc,
                           original_cell=original_cell,
                           no_standardize=True,
                           )
        ph_lammps_input = lmp_stc.get_lammps_input_for_phonolammps()
        ph_lmp = Phonolammps(lammps_input=ph_lammps_input,
                             supercell_matrix=sup_mat,
                             primitive_matrix=np.identity(3))
        phonon = get_phonon_from_phonolammps(ph_lmp)
        ph_analyzer = PhononAnalyzer(phonon=phonon,
                                     relax_analyzer=rlx_analyzer)
        return ph_analyzer

    primitive_matrix = np.identity(3)
    cell = twinboundary_structure.get_cell_for_export(
            get_lattice=False,
            move_atoms_into_unitcell=move_atoms_into_unitcell,
            )
    std = StandardizeCell(cell)

    if no_standardize:
        _original_cell = None
        _cell = std.cell
    else:
        _original_cell = std.cell
        _cell = std.get_standardized_cell(to_primitive=False,
                                          no_idealize=False,
                                          no_sort=True)
    ph_analyzer = _get_phonon_analyzer(
                      cell=_cell,
                      original_cell=_original_cell,
                      is_rlx_lat=is_relax_lattice,
                      no_std=no_standardize,
                      sup_mat=supercell_matrix,
                      )
    tb_analyzer = TwinBoundaryAnalyzer(
                      twinboundary_structure=twinboundary_structure,
                      twinboundary_phonon_analyzer=ph_analyzer,
                      hexagonal_relax_analyzer=hexagonal_relax_analyzer,
                      hexagonal_phonon_analyzer=hexagonal_phonon_analyzer,
                      )
    return tb_analyzer


def get_phonon_from_phonolammps(phonolammps) -> Phonopy:
    """
    Get Phonopy class object from PhonoLammps.

    Args:
        phonolammps: Phonlammps class object.
    """
    unitcell = phonolammps.get_unitcell()
    force_constants = phonolammps.get_force_constants()
    supercell_matrix = phonolammps.get_supercell_matrix()
    phonon = Phonopy(unitcell=unitcell,
                     supercell_matrix=supercell_matrix)
    phonon.set_force_constants(force_constants)

    return phonon
