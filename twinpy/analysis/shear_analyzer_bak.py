#!/usr/bin/env pythoo
# -*- coding: utf-8 -*-

"""
Analize shear calculation.
"""
import numpy as np
from twinpy.structure.diff import get_structure_diff
from twinpy.structure.standardize import StandardizeCell
from twinpy.plot.base import get_plot_properties_for_trajectory
from twinpy.plot.band import bands_plot


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
            relax_cells (list): relax cells of vasp

        Raises:
            RuntimeError: The number of atoms changes between original cells
                          and input cells, which is not supported.

        Todo:
            Currently not supported the case the number of original_cells
            and input_cells changes because it is difficult to construct
            the relax cells in the original frame. But future fix this
            problem. One solution is to make attribute
            'self._original_primitive' which contains two atoms
            in the unit cell and original basis.
        """
        def __check_number_of_atoms_not_changed(original_cells,
                                                input_cells):
            for original_cell, input_cell in zip(original_cells, input_cells):
                if not len(original_cell) == len(input_cell):
                    raise RuntimeError(
                            "The number of atoms changes between "
                            "original cells and input cells, "
                            "which is not supported.")

        __check_number_of_atoms_not_changed(original_cells, input_cells)
        self._original_cells = original_cells
        self._input_cells = input_cells
        self._relax_cells = None
        self._set_relax_cells(input_cells, relax_cells)
        self._standardizes = None
        self._set_standardizes()
        self._relax_cells_original_frame = None
        self._set_relax_cells_in_original_frame()
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

    def _set_relax_cells(self, input_cells, relax_cells):
        """
        Check lattice does not change in relaxation and
        set relax_cells.
        """
        for input_cell, relax_cell in zip(input_cells, relax_cells):
            np.testing.assert_allclose(input_cell[0], relax_cell[0],
                                       atol=1e-5)
        self._relax_cells = relax_cells

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

        standardizes = [ StandardizeCell(cell)
                             for cell in self._original_cells ]

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

    def _set_relax_cells_in_original_frame(self) -> list:
        """
        Set relax cells in original frame which is not
        conventional and its angles are close to (90., 90., 120.).

        Returns:
            list: relax cells in original frame

        Note:
            For variable definitions in this definition,
            see Eq.(1.8) and (1.17) in Crystal Structure documention.
            Note that by crystal body rotation, fractional
            coordinate of atom positions are not changed.
        """
        def __get_relax_atoms_in_original_frame(prim_atoms,
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

        relax_orig_cells = []
        for i in range(len(self._relax_cells)):
            lattice = self._original_cells[i][0]
            conv_to_prim_matrix = \
                self._standardizes[i].conventional_to_primitive_matrix
            transformation_matrix = \
                self._standardizes[i].transformation_matrix
            origin_shift = self._standardizes[i].origin_shift
            scaled_positions = __get_relax_atoms_in_original_frame(
                    prim_atoms=self._relax_cells[i][1],
                    conv_to_prim_matrix=conv_to_prim_matrix,
                    transformation_matrix=transformation_matrix,
                    origin_shift=origin_shift,
                    )
            symbols = self._original_cells[i][2]
            relax_orig_cell = (lattice, scaled_positions, symbols)
            relax_orig_cells.append(relax_orig_cell)

        self._relax_cells_in_original_frame = relax_orig_cells

    @property
    def relax_cells_in_original_frame(self):
        """
        Relax cells in original frame
        """
        return self._relax_cells_in_original_frame

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

    def get_relax_diffs(self):
        """
        Get structure diffs between input and relax cells
        IN ORIGINAL FRAME.
        """
        diffs = []
        for input_cell, relax_cell in \
                zip(self._original_cells, self._relax_cells_in_original_frame):
            cells = (input_cell, relax_cell)
            diff = get_structure_diff(cells=cells,
                                      base_index=0,
                                      include_base=False)
            diffs.append(diff)

        return diffs

    def get_shear_diffs(self):
        """
        Get structure diffs between original and sheared structures
        IN ORIGINAL FRAME.
        """
        cells = self._relax_cells_in_original_frame
        diffs = get_structure_diff(cells=cells,
                                   base_index=0,
                                   include_base=True)
        return diffs

    def plot_bands(self,
                   fig,
                   with_dos=False,
                   mesh=None,
                   band_labels=None,
                   segment_qpoints=None,
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
                get_plot_properties_for_trajectory(
                        plot_nums=len(self._phonons))
        transformation_matrices = \
                [ std.transformation_matrix for std in self._standardizes ]
        bands_plot(fig=fig,
                   phonons=self._phonons,
                   transformation_matrices=transformation_matrices,
                   band_labels=band_labels,
                   segment_qpoints=segment_qpoints,
                   xscale=xscale,
                   npoints=npoints,
                   with_dos=with_dos,
                   mesh=mesh,
                   cs=cs,
                   alphas=alphas,
                   linewidths=linewidths,
                   linestyles=linestyles,
                   labels=labels,
                   )
