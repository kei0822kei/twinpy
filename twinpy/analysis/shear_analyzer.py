#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analize shear calculation.
"""
import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Supercell
from twinpy.structure.diff import get_structure_diff
from twinpy.structure.standardize import StandardizeCell
from twinpy.common.plot import (bands_plot,
                                get_plot_properties_from_trajectory)


def is_cells_are_same(first_cell:tuple,
                      second_cell:tuple) -> bool:
    """
    Check first cell and second cell are same.

    Args:
        first_cell (tuple): first cell
        second_cell (tuple): second cell

    Returns:
        bool: return True if two cells are same
    """
    is_lattice_same = np.allclose(first_cell[0], second_cell[0])
    is_scaled_positions_same = np.allclose(first_cell[1], second_cell[1])
    is_symbols_same = (first_cell[2] == second_cell[2])
    is_same = (is_lattice_same
               and is_scaled_positions_same
               and is_symbols_same)
    return is_same


class ShearAnalyzer():
    """
    Analize shear result.
    """

    def __init__(
           self,
           original_cells:list,
           input_cells:list,
           relax_cells:list,
           ):
        """
        Args:
            original_cells (list): primitivie original cells, which is output
                               cells of ShearStructure class
            input_cells (list): input cells for vasp
            relax_eells (list): relax cells of vasp
        """
        self._original_cells = original_cells
        self._input_cells = input_cells
        self._relax_cells = relax_cells
        self._standardizes = None
        self._set_standardizes()
        self._phonons = None

    @property
    def original_cells(self):
        """
        Original cells, which are output of ShearStructure.
        """
        return self._original_cells

    @property
    def input_cells(self):
        """
        Relax input cells.
        """
        return self._input_cells

    @property
    def relax_cells(self):
        """
        Relax output cells.
        """
        return self._relax_cells

    def _set_standardizes(self):
        """
        Set standardizes.
        """
        to_primitive = True
        no_idealize = False
        symprec = 1e-5
        no_sort = False
        get_sort_list = False

        standardizes = [ StandardizeCell(cell) for cell in self._original_cells ]

        for i, standardize in enumerate(standardizes):
            std_cell = standardize.get_standardized_cell(
                    to_primitive=to_primitive,
                    no_idealize=no_idealize,
                    symprec=symprec,
                    no_sort=no_sort,
                    get_sort_list=get_sort_list,
                    )
            cells_are_same = is_cells_are_same(self._input_cells[i],
                                               std_cell)
            if not cells_are_same:
                raise RuntimeError("standardized cell do not match "
                                   "with input cell")

        self._standardizes = standardizes

    @property
    def standardizes(self):
        """
        Standardizes.
        """
        return self._standardizes

    def set_phonons(self, phonons):
        """
        Set phonons.
        """
        self._phonons = phonons

    @property
    def phonons(self):
        """
        Phonons.
        """
        return self._phonons

    def get_relax_cells_with_original_basis(self):
        """
        get relax cells with original basis
        """
        # def __get_idx(lst, val):
        #     a = np.round(lst, decimals=8)
        #     b = np.round(val, decimals=8)
        #     idx = [ i for i in range(len(a)) if np.all(a[i] == b) ]
        #     assert len(idx) == 1
        #     return idx[0]

        if self._structure_type == 'base':
            return self._relax_cells

        orig_relax_cells = []
        for i, relax_cell in enumerate(self._relax_cells):
            R = self._standardizes[i].rotation_matrix
            P_c = self._standardizes[i].conventional_to_primitive_matrix
            P = self._standardizes[i].transformation_matrix
            p = self._standardizes[i].origin_shift

            before_rotate_lattice = \
                    np.dot(np.linalg.inv(R),
                           np.transpose(relax_cell[0])).T

            supercell_matrix = None
            if self._structure_type == 'primitive':
                supercell_matrix = np.dot(np.linalg.inv(P_c), P)
            else:
                supercell_matrix = P
            np.testing.assert_allclose(
                    np.round(supercell_matrix, decimals=8) % 1,
                    np.zeros(supercell_matrix.shape),
                    err_msg="supercell_matrix must int matrix")
            supercell_matrix = \
                    np.round(supercell_matrix, decimals=8).astype(int)

            before_rotate_atoms = np.dot(np.linalg.inv(P),
                                         np.transpose(relax_cell[1])).T
            before_rotate_atoms_input = np.dot(np.linalg.inv(P),
                                               np.transpose(self._input_cells[i][1])).T
            before_rotate_shift = np.dot(np.linalg.inv(P), p)
            fixed_atoms = (before_rotate_atoms - before_rotate_shift) % 1
            fixed_atoms_input = (before_rotate_atoms_input - before_rotate_shift) % 1

            cell = (before_rotate_lattice,
                    fixed_atoms,
                    relax_cell[2])
            cell_input = (before_rotate_lattice,
                          fixed_atoms_input,
                          relax_cell[2])
            unitcell = PhonopyAtoms(cell=cell[0],
                                    scaled_positions=cell[1],
                                    symbols=cell[2])
            unitcell_input = PhonopyAtoms(cell=cell_input[0],
                                          scaled_positions=cell_input[1],
                                          symbols=cell_input[2])
            supercell = Supercell(unitcell=unitcell,
                                  supercell_matrix=supercell_matrix)
            supercell_input = Supercell(unitcell=unitcell_input,
                                        supercell_matrix=supercell_matrix)

            # atom order is different from original, so fix it
            # print(i)
            # print(supercell_input.get_scaled_positions())
            # print(self._original_cells[i][1])
            # idxes = [ __get_idx(self._original_cells[i][1], val)
            #           for val in supercell_input.get_scaled_positions() ]
            # fixed_atoms = supercell.get_scaled_positions()[idxes]

            # orig_relax_cell = (supercell.get_cell(),
            #                    fixed_atoms,
            #                    supercell.get_chemical_symbols())
            orig_relax_cell = (supercell.get_cell(),
                               supercell.get_scaled_positions(),
                               supercell.get_chemical_symbols())
            np.testing.assert_allclose(
                    np.round(orig_relax_cell[0], decimals=8),
                    np.round(self._original_cells[i][0], decimals=8))


            orig_relax_cells.append(orig_relax_cell)

        return orig_relax_cells

    def get_relax_diffs(self):
        """
        get structure diffs between original and relax structures
        """
        output_cells = self.get_relax_cells_with_original_basis()
        diffs = [ get_structure_diff(input_cell, output_cell)
                  for input_cell, output_cell
                  in zip(self._original_cells, output_cells) ]
        return diffs

    def get_shear_diffs(self):
        """
        get structure diffs between original and sheared structures
        """
        output_cells = self.get_relax_cells_with_original_basis()
        diffs = get_structure_diff(*output_cells)
        return diffs

    def plot_bands(self,
                   fig,
                   with_dos=False,
                   mesh=None,
                   band_labels=None,
                   segment_qpoints=None,
                   is_auto=False,
                   xscale=20,
                   npoints=51,
                   labels=None,):
        """
        plot phonon bands

        Args:
            arg1 (str): description
            arg2 (np.array): (3x3 numpy array) description

        Returns:
            dict: description

        Raises:
            ValueError: description

        Examples:
            description

            >>> print_test ("test", "message")
              test message

        Note:
            description
        """
        cs, alphas, linewidths, linestyles = \
                get_plot_properties_from_trajectory(
                        plot_nums=len(self._phonons))
        bands_plot(fig=fig,
                   phonons=self._phonons,
                   original_cells=self._original_cells,
                   with_dos=with_dos,
                   mesh=mesh,
                   band_labels=band_labels,
                   segment_qpoints=segment_qpoints,
                   is_auto=is_auto,
                   xscale=xscale,
                   npoints=npoints,
                   cs=cs,
                   alphas=alphas,
                   linewidths=linewidths,
                   linestyles=linestyles,
                   labels=labels,
                   )
