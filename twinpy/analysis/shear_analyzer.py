#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analize shear
"""
import warnings
import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Supercell
from twinpy.structure.base import get_cell_from_phonopy_structure
from twinpy.structure.diff import get_structure_diff
from twinpy.structure.standardize import StandardizeCell
from twinpy.common.plot import (bands_plot,
                                get_plot_properties_from_trajectory)

class ShearAnalyzer():
    """
    analize shear result

       .. attribute:: att1

          Optional comment string.


       .. attribute:: att2

          Optional comment string.

    """

    def __init__(
           self,
           structure_type:str,
           orig_cells:list,
           input_cells:list,
           symprec:float=1e-5,
       ):
        """
        Args:
            structure_type (str): 'base', 'primitive' or 'conventional'
            orig_cells (list): (primitivie) original cells,
                                orig_cells=(cell1, cell2, ...)
            input_cells (list): input cells for vasp
        """
        self._check_input_cells(structure_type=structure_type,
                                orig_cells=orig_cells,
                                input_cells=input_cells,
                                symprec=symprec)
        self._structure_type = structure_type
        self._orig_cells = orig_cells
        self._input_cells = input_cells
        self._symprec = symprec
        self._standardizes = None
        self._set_standardizes()
        self._relax_cells = None
        self._phonons = None

    def _check_structure_type(self, structure_type):
        valid_structure_type = ['base', 'primitive', 'convnetional']
        if not structure_type in valid_structure_type:
            raise ValueError("illigal inputs structure_tyle: {}" \
                             .format(structure_type))

    def _check_input_cells(self,
                           structure_type,
                           orig_cells,
                           input_cells,
                           symprec):
        self._check_structure_type(structure_type=structure_type)
        if structure_type == 'base':
            input_cells_ = orig_cells
        else:
            if structure_type == 'primitive':
                to_primitive = True
            else:
                to_primitive = False
            input_cells_ = [ StandardizeCell(cell).get_standardized_cell(
                             to_primitive=to_primitive,
                             no_idealize=False,
                             symprec=symprec)
                             for cell in orig_cells ]
        are_same_lattices = [ np.allclose(input_cells_[i][0],
                                          input_cells[i][0])
                              for i in range(len(input_cells)) ]
        if False in are_same_lattices:
            raise RuntimeError("could not be detected relation "
                               "between original cell and input cell")

    def _set_standardizes(self):
        self._standardizes = \
                [ StandardizeCell(cell) for cell in self._orig_cells ]

    @property
    def standardizes(self):
        """
        standardizes
        """
        return self._standardizes

    def set_relax_cells(self, relax_cells:list):
        """
        Args:
            relax_cells (list): relax cells
        """
        self._relax_cells = relax_cells

    def set_phonons(self, phonons):
        """
        set phonons
        """
        self._phonons = phonons

    @property
    def phonons(self):
        """
        phonons
        """
        return self._phonons

    @property
    def original_cells(self):
        """
        original cells
        """
        return self._orig_cells

    @property
    def input_cells(self):
        """
        relax input cells
        """
        return self._input_cells

    @property
    def relax_cells(self):
        """
        relax output cells
        """
        return self._relax_cells

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
            # print(self._orig_cells[i][1])
            # idxes = [ __get_idx(self._orig_cells[i][1], val)
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
                    np.round(self._orig_cells[i][0], decimals=8))


            orig_relax_cells.append(orig_relax_cell)

        return orig_relax_cells

    def get_relax_diffs(self):
        """
        get structure diffs between original and relax structures
        """
        output_cells = self.get_relax_cells_with_original_basis()
        diffs = [ get_structure_diff(input_cell, output_cell)
                  for input_cell, output_cell
                  in zip(self._orig_cells, output_cells) ]
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
                   orig_cells=self._orig_cells,
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
