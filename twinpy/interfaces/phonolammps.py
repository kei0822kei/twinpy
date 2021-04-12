#!/usr/bin/env python

"""
Interfaces for PhonoLammps.
"""

import numpy as np
from phonolammps import PhonoLammps
from phonopy import Phonopy


def get_phonon_from_lammps(self,
                           lammps_input:list,
                           supercell_matrix:np.array=np.eye(3),
                           primitive_matrix:np.array=np.eye(3)):
    """
    Get Phonopy class object from PhonoLammps.
    """
    phlammps = Phonolammps(strings,
                       supercell_matrix=supercell_matrix,
                       primitive_matrix=primitive_matrix)
    phonon = Phonopy(unitcell,
                 supercell_matrix)
    phonon.set_force_constants(force_constants)

    return phonon
