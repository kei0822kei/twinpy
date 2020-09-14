#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Symmetry Analyzer
"""

import numpy as np
import spglib
from phonopy.structure.atoms import symbol_map, atom_data
from phonopy.interface.vasp import sort_positions_by_symbols


def get_standardized_cell(cell:tuple,
                          to_primitive:bool,
                          no_idealize:bool,
                          symprec:float=1.e-5,
                          no_sort:bool=False,
                          get_sort_list:list=False,
                          ) -> tuple:
    """
    Get standardized cell.

    Args:
        cell (tuple): input cell
        to_primitive (bool): True => primitive,
                             False => conventional
        no_idealize (bool): True => not rotate crystal body,
                            False => rotate crystal body
        symprec (float): symmetry tolerance, for more detail
                         see spglib documentation
        no_sort (bool): does not change atoms order
        get_sort_list (bool): When no_sort=True, return sort list

    Returns:
        tuple: standardized cell

    Note:
        For the other arguments, see StandardizeCell.get_standardized_cell.
    """
    std = StandardizeCell(cell)
    std_cell = std.get_standardized_cell(to_primitive=to_primitive,
                                         no_idealize=False,
                                         symprec=symprec,
                                         no_sort=no_sort,
                                         get_sort_list=False)
    return std_cell


def get_atomic_numbers(symbols:list) -> list:
    """
    Get atomic numbers from symbols.

    Returns:
        list: list of element numbers
    """
    numbers = [ symbol_map[symbol] for symbol in symbols ]
    return numbers


def get_chemical_symbols(numbers) -> list:
    """
    Get chemical symbols from atomic numbers.

    Returns:
        list: list of symbols
    """
    symbols = [ atom_data[num][1] for num in numbers ]
    return symbols


def get_conventional_to_primitive_matrix(centering:str) -> np.array:
    """
    get conventional to primitive matrix, P_c

    Args:
        centering (str): choose from  'P', 'A', 'C', 'R', 'I' and 'F'

    Raises:
        RuntimeError: irrigal centering specified

    Returns:
        np.array: conventional to primitive matrix
    """
    if centering == 'P':
        P_c = np.array([[ 1. ,  0. ,  0.  ],
                        [ 0. ,  1. ,  0.  ],
                        [ 0. ,  0. ,  1.  ]])
    elif centering == 'A':
        P_c = np.array([[ 1. ,  0. ,  0.  ],
                        [ 0. ,  0.5, -0.5 ],
                        [ 0. ,  0.5,  0.5 ]])
    elif centering == 'C':
        P_c = np.array([[ 0.5,  0.5,  0.  ],
                        [-0.5,  0.5,  0.  ],
                        [ 0. ,  0. ,  1.  ]])
    elif centering == 'R':
        P_c = np.array([[ 2/3, -1/3, -1/3 ],
                        [ 1/3,  1/3, -2/3 ],
                        [ 1/3 , 1/3,  1/3 ]])
    elif centering == 'I':
        P_c = np.array([[-0.5,  0.5,  0.5 ],
                        [ 0.5, -0.5,  0.5 ],
                        [ 0.5,  0.5, -0.5 ]])
    elif centering == 'F':
        P_c = np.array([[ 0. ,  0.5,  0.5 ],
                        [ 0.5,  0. ,  0.5 ],
                        [ 0.5,  0.5,  0.  ]])
    else:
        print(centering)
        raise RuntimeError("irrigal centering specified")
    return P_c


class StandardizeCell():
    """
    Symmetry analyzer
    """

    def __init__(self, cell):
        """
        Args:
            cell: cell = (lattice, scaled_positions, symbols)
        """
        self._cell = cell
        self._atomic_numbers = get_atomic_numbers(self._cell[2])
        self._cell_for_spglib = (self._cell[0],
                                 self._cell[1],
                                 self._atomic_numbers)
        self._dataset = spglib.get_symmetry_dataset(self._cell_for_spglib)
        self._conv_to_prim_matrix = get_conventional_to_primitive_matrix(
                self._dataset['international'][0])
        self._rotation_matrix = self._dataset['std_rotation_matrix']
        self._transformation_matrix = self._dataset['transformation_matrix']
        self._origin_shift = self._dataset['origin_shift']
        self._check_spg_output()

    @property
    def cell(self):
        """
        Cell
        """
        return self._cell

    @property
    def origin_shift(self):
        """
        Origin shift, p.
        """
        return self._origin_shift

    @property
    def rotation_matrix(self):
        """
        Rotation matrix.
        """
        return self._rotation_matrix

    @property
    def transformation_matrix(self):
        """
        Transformation matrix, P.
        """
        return self._transformation_matrix

    @property
    def conventional_to_primitive_matrix(self):
        """
        Conventional to primitive matrix, P_c.
        """
        return self._conv_to_prim_matrix

    def _check_spg_output(self):
        """
        check spglib output

        Raises:
            AssertionError: M != M_s P
            AssertionError: M_p != M_s P_c
            AssertionError: M_bar_s != R M_s
            AssertionError: M_bar_p != R M_p
            AssertionError: x_s != P x + p
            AssertionError: x_p != P_c^{-1} x_s
        """
        def __check_lattice_matrix(P, P_c, R, M, M_s, M_p,
                                   M_bar_s, M_bar_p):
            # check M = M_s P
            np.testing.assert_allclose(
                    M,
                    np.dot(M_s, P),
                    atol=atol
                    )
            # check M_p = M_s P_c
            np.testing.assert_allclose(
                    M_p,
                    np.dot(M_s, P_c),
                    atol=atol
                    )
            # check: M_bar_s = R M_s
            np.testing.assert_allclose(
                    M_bar_s,
                    np.dot(R, M_s),
                    atol=atol
                    )
            # check: M_bar_p = R M_p
            np.testing.assert_allclose(
                    M_bar_p,
                    np.dot(R, M_p),
                    atol=atol
                    )

        def __check_atom_positions(P, p, P_c, x, x_s, x_p):
            # check x_s = P x + p
            x_s_ = np.round((np.dot(P, np.transpose(x)).T + p),
                            decimals=8) % 1
            for i in range(len(x_s_)):
                # assert x_s_[i] in x_s, \
                assert np.round(x_s_[i], decimals=8) \
                       in np.round(x_s, decimals=8), \
                       "x_s != Px+p, check script"
            # check x_p = P_c^{-1} x_s
            x_p_ = (np.dot(np.linalg.inv(P_c), np.transpose(x_s)).T) % 1
            for i in range(len(x_p_)):
                assert np.round(x_p_[i], decimals=8) \
                       in np.round(x_p, decimals=8), \
                       'x_p != P_c^{-1} x_s, check script'

        atol = 1e-5

        conv_orig = self.get_standardized_cell(to_primitive=False,
                                               no_idealize=True)
        prim_orig = self.get_standardized_cell(to_primitive=True,
                                               no_idealize=True)
        conv = self.get_standardized_cell(to_primitive=False,
                                          no_idealize=False)
        prim = self.get_standardized_cell(to_primitive=True,
                                          no_idealize=False)

        x = self._cell[1]
        x_s = conv[1]
        x_p = prim[1]
        M = np.transpose(self._cell[0])
        M_s = np.transpose(conv_orig[0])
        M_p = np.transpose(prim_orig[0])
        M_bar_s = np.transpose(conv[0])
        M_bar_p = np.transpose(prim[0])
        R = self._rotation_matrix
        P = self._transformation_matrix
        P_c = self._conv_to_prim_matrix
        p = self._origin_shift

        __check_lattice_matrix(P, P_c, R, M, M_s, M_p,
                               M_bar_s, M_bar_p)
        __check_atom_positions(P, p, P_c, x, x_s, x_p)

    def get_standardized_cell(self,
                              to_primitive:bool,
                              no_idealize:bool,
                              symprec:float=1e-5,
                              no_sort:bool=False,
                              get_sort_list:list=False):
        """
        Get stadandardized cell.

        Args:
            to_primitive (bool): True => primitive,
                                 False => conventional
            no_idealize (bool): True => not rotate crystal body,
                                False => rotate crystal body
            symprec (float): symmetry tolerance, for more detail
                             see spglib documentation
            no_sort (bool): does not change atoms order
            get_sort_list (bool): When no_sort=True, return sort list
        """
        spg_cell = spglib.standardize_cell(self._cell_for_spglib,
                                           to_primitive=to_primitive,
                                           no_idealize=no_idealize,
                                           symprec=symprec)
        symbols = get_chemical_symbols(spg_cell[2])
        std_cell = (spg_cell[0], spg_cell[1], symbols)
        if no_sort:
            return std_cell
        else:
            num_atoms, unique_symbols, scaled_positions, sort_list = \
                sort_positions_by_symbols(
                        symbols=std_cell[2],
                        positions=std_cell[1])
            symbols = []
            for num, symbol in zip(num_atoms, unique_symbols):
                symbols.extend([symbol] * num)

            sort_std_cell = (std_cell[0], scaled_positions, symbols)
            if get_sort_list:
                return (sort_std_cell, sort_list)
            else:
                return sort_std_cell

    # def convert_to_original_frame(self, cell:tuple, input_is_primitive:bool) -> tuple:
    #     """
    #     If you input relaxed cell,
    #     you can get relaxed cell in original coordinate.
    #     """
    #     M_bar_std = cell[0].T
    #     R = self._rotation_matrix
    #     M_std = np.dot(np.linalg.inv(R), M_bar_std)
