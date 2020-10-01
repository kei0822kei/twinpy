#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analize relax calculation.
I did not name this file as vasp_analyzer.py
because this also can be used in the case lammps is used.
"""
import numpy as np
from twinpy.structure.base import check_same_cells
from twinpy.structure.diff import get_structure_diff
from twinpy.structure.standardize import StandardizeCell


class RelaxAnalyzer():
    """
    Analize relax result.
    """

    def __init__(
           self,
           initial_cell:tuple,
           final_cell:tuple,
           original_cell:tuple=None,
           forces:np.array=None,
           stress:np.array=None,
           ):
        """
        Args:
            initial_cell (tuple): Initial cell for vasp.
            final_cell (tuple): Final cell of vasp
            original_cell (tuple): Original cell whose standardized cell
                                   is initail_cell.
            forces (np.array): Forces acting on atoms in the final cell.
            stress (np.array): Forces acting on the final cell.
        """
        self._initial_cell = initial_cell
        self._final_cell = final_cell
        self._original_cell = None
        self._standardize = None
        self._final_cell_in_original_frame = None
        if original_cell is not None:
            self.set_original_cell(original_cell=original_cell)
        self._forces = forces
        self._stress = stress

    @property
    def initial_cell(self):
        """
        Initial cell.
        """
        return self._initial_cell

    @property
    def final_cell(self):
        """
        Final cell.
        """
        return self._final_cell

    def set_original_cell(self, original_cell:tuple):
        """
        Set original cell.

        Args:
            original_cell (tuple): Original cell whose standardized cell
                                   is initail_cell.
        """
        if not len(original_cell[2]) == len(self._initial_cell[2]):
            raise RuntimeError(
                    "The number of atoms changes between "
                    "original cell and initial cell, "
                    "which is not supported.")

        std = StandardizeCell(cell=original_cell)
        primitive_cell = std.get_standardized_cell(
                to_primitive=True,
                no_idealize=False,
                symprec=1e-5,
                get_sort_list=False)
        print(primitive_cell)
        print(self._initial_cell)
        if not check_same_cells(first_cell=primitive_cell,
                                second_cell=self._initial_cell):
            raise RuntimeError("Standardized original cell does not "
                               "the same as initial cell.")
        self._original_cell = original_cell
        self._standardize = std
        self._set_final_cell_in_original_frame()

    def _set_final_cell_in_original_frame(self):
        """
        Set final cell in original frame.

        Note:
            For variable definitions in this definition,
            see Eq.(1.8) and (1.17) in Crystal Structure documention.
            Note that by crystal body rotation, fractional
            coordinate of atom positions are not changed.
        """
        def __get_final_atoms_in_original_frame(prim_atoms,
                                                conv_to_prim_matrix,
                                                transformation_matrix,
                                                origin_shift):
            X_p = prim_atoms.T
            P_c = conv_to_prim_matrix
            P = transformation_matrix
            p = origin_shift.reshape(3,1)

            X_s = np.dot(P_c, X_p)
            X = np.dot(np.linalg.inv(P), X_s) \
                    - np.dot(np.linalg.inv(P), p)
            orig_atoms = np.round(X.T, decimals=8) % 1.
            return orig_atoms

        lattice = self._original_cell[0]
        conv_to_prim_matrix = \
            self._standardize.conventional_to_primitive_matrix
        transformation_matrix = \
            self._standardize.transformation_matrix
        origin_shift = self._standardize.origin_shift
        scaled_positions = __get_final_atoms_in_original_frame(
                prim_atoms=self._final_cell[1],
                conv_to_prim_matrix=conv_to_prim_matrix,
                transformation_matrix=transformation_matrix,
                origin_shift=origin_shift,
                )
        symbols = self._original_cell[2]
        final_orig_cell = (lattice, scaled_positions, symbols)

        self._final_cell_in_original_frame = final_orig_cell

    def _check_original_cell_is_set(self):
        """
        Check original cell is set.

        Raises:
            RuntimeError: Original cell is not set.
        """
        if self._original_cell is None:
            raise RuntimeError("Original cell is not set.")

    @property
    def original_cell(self):
        """
        Original cell.
        """
        return self._original_cell

    @property
    def standardize(self):
        """
        StandardizeCell class object.
        Which is for original and initial cell.
        """
        return self._standardize

    @property
    def final_cell_in_original_frame(self):
        """
        Final cell in original frame.
        """
        return self._final_cell_in_original_frame

    @property
    def forces(self):
        """
        Forces acting on atoms.
        """
        return self._forces

    @property
    def stress(self):
        """
        Stress acting on atoms.
        """
        return self._stress

    def get_diff(self,
                 is_original_frame:bool=False):
        """
        Get structure diffs between initial cell and final cell.

        Args:
            is_original_frame (bool): If True, get diff in original frame.

        Notes:
            When you use is_original_frame=True, you have to set
            original_cell before running this function.
        """
        if is_original_frame:
            self._check_original_cell_is_set()
            initial_cell = self._original_cell
            final_cell = self._final_cell_in_original_frame
        else:
            initial_cell = self._initial_cell
            final_cell = self._final_cell

        diff = get_structure_diff(cells=(initial_cell, final_cell),
                                  base_index=0,
                                  include_base=False)

        return diff
