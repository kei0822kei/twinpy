#!/usr/bin/env pythoo
# -*- coding: utf-8 -*-

"""
Analize twinboudnary relax calculation.
"""
import numpy as np
from twinpy.structure.diff import get_structure_diff
from twinpy.structure.twinboundary import TwinBoundaryStructure
from twinpy.structure.standardize import StandardizeCell
from twinpy.plot.base import get_plot_properties_for_trajectory
from twinpy.plot.band import bands_plot
from twinpy.common.kpoints import get_mesh_offset_from_direct_lattice
from phonopy import Phonopy


class TwinBoundaryAnalyzer():
    """
    Analize shear result.
    """

    def __init__(
           self,
           twinboundary_structure:TwinBoundaryStructure,
           hexagonal_phonon:Phonopy,
           twinboundary_phonon:Phonopy,
           ):
        """
        Args:
            twinboundary_structure:TwinBoundaryStructure object.
            hexagonal_phonon: hexagonal Phonopy object
            twinboundary_phonon: twinboundary Phonopy object
        """
        self._twinboundary_structure = twinboundary_structure
        self._hexagonal_phonon = hexagonal_phonon
        self._twinboundary_phonon = twinboundary_phonon
        self._standardize = None
        self._set_standardize()
        self._hexagonal_to_original_rotation_matrix = None
        self._twinboundary_to_original_rotation_matrix = None
        self._set_rotation_matrices()

    def _set_standardize(self):
        """
        Set standardize.
        """
        cell = self._twinboundary_structure.get_cell_for_export()
        self._standardize = StandardizeCell(cell)

    def _set_rotation_matrices(self):
        """
        Set rotation matrix.
        """
        self._hexagonal_to_original_rotation_matrix = \
                self._twinboundary_structure.rotation_matrix
        self._twinboundary_to_original_rotation_matrix = \
                np.linalg.inv(self._standardize.rotation_matrix)

    @property
    def twinboundary_structure(self):
        """
        TwinBoundaryStructure object
        """
        return self._twinboundary_structure

    @property
    def hexagonal_phonon(self):
        """
        Bulk phonon.
        """
        return self._hexagonal_phonon

    @property
    def twinboundary_phonon(self):
        """
        Twinboundary phonon.
        """
        return self._twinboundary_phonon

    @property
    def hexagonal_to_original_rotation_matrix(self):
        """
        Hexagonal to original rotation matrix.
        """
        return self._hexagonal_to_original_rotation_matrix

    @property
    def twinboundary_to_original_rotation_matrix(self):
        """
        Twinboundary to original rotation matrix.
        """
        return self._twinboundary_to_original_rotation_matrix

    def run_mesh(self, interval:float=0.1):
        """
        Run mesh for both hexagonal and twinboundary phonon.

        Args:
            interval (float): mesh interval
        """
        phonons = (self._hexagonal_phonon, self._twinboundary_phonon)
        structure_types = ['hexagonal', 'twinboundary']
        for structure_type, phonon in zip(structure_types, phonons):
            lattice = phonon.primitive.get_cell()
            kpt = get_mesh_offset_from_direct_lattice(
                    lattice=lattice,
                    interval=interval,
                    )
            print("run mesh with {} ({})".format(
                kpt['mesh'], structure_type))
            phonon.run_mesh
            phonon.set_mesh(
                mesh=kpt['mesh'],
                shift=None,
                is_time_reversal=True,
                is_mesh_symmetry=False,  # necessary for calc ellipsoid
                is_eigenvectors=True,
                is_gamma_center=False,
                run_immediately=True)

    def get_thermal_displacement_matrices(
            self,
            t_step:int=100,
            t_max:int=1000,
            t_min:int=0,
            with_original_cart:bool=True,
            def_cif:bool=False,
            ):
        """
        Get ThermalDisplacementMatrices object for
        both hexagonal and twinboundary.

        Args:
            t_step (int): temperature interval
            t_max (int): max temperature
            t_min (int): minimum temperature
            with_original_cart (bool): if True, use twinboundary
                                       original frame
            def_cif (bool): if True, use cif definition

        Todo:
            I do not know how to rotate 4d array (temp, atoms, 3, 3).
        """
        phonons = (self._hexagonal_phonon, self._twinboundary_phonon)
        tdm_matrices = []
        rotation_matrices = (self._hexagonal_to_original_rotation_matrix,
                             self._twinboundary_to_original_rotation_matrix)
        for phonon, rotation_matrix in zip(phonons, rotation_matrices):
            phonon.set_thermal_displacement_matrices(
                t_step=t_step,
                t_max=t_max,
                t_min=t_min,
                freq_min=None,
                freq_max=None,
                t_cif=None)
            tdm = phonon.thermal_displacement_matrices
            if def_cif:
                matrix = tdm.thermal_displacement_matrices_cif
            else:
                matrix = tdm.thermal_displacement_matrices
            if with_original_cart:
                rot_matrices = []
                shape = matrix.shape
                lst = []
                for i in range(shape[0]):
                    atom_lst = []
                    for j in range(shape[1]):
                        mat = np.dot(rotation_matrix,
                                     np.dot(matrix[i,j],
                                            rotation_matrix.T))
                        atom_lst.append(mat)
                    lst.append(atom_lst)
                tdm_matrices.append(np.array(lst))
            else:
                tdm_matrices.append(tdm.thermal_displacement_matrices)
        return tuple(tdm_matrices)
