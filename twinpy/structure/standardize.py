#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Symmetry Analyzer
"""

import numpy as np
import spglib
from phonopy.structure.atoms import symbol_map, atom_data
from phonopy.interface.vasp import sort_positions_by_symbols

def get_atomic_numbers(symbols):
    """
    get atomic numbers from symbols
    """
    numbers = [ symbol_map[symbol] for symbol in symbols ]
    return numbers

def get_chemical_symbols(numbers):
    """
    get chemical symbols from atomic numbers
    """
    symbols = [ atom_data[num][0] for num in numbers ]
    return symbols

def get_conventional_to_primitive_matrix(centering:str):
    """
    get conventional to primitive matrix, P_c

    Args:
        centering (str): choose from  'P', 'A', 'C', 'R', 'I' and 'F'
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
    symmetry analyzer

       .. attribute:: att1

          Optional comment string.


       .. attribute:: att2

          Optional comment string.

    """

    def __init__(self, cell):
        """
        init

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
        cell
        """
        return self._cell

    @property
    def origin_shift(self):
        """
        origin shift, p
        """
        return self._origin_shift

    @property
    def rotation_matrix(self):
        """
        rotation matrix
        """
        return self._rotation_matrix

    @property
    def transformation_matrix(self):
        """
        transformation matrix, P
        """
        return self._transformation_matrix

    @property
    def conventional_to_primitive_matrix(self):
        """
        conventional to primitive matrix, P_c
        """
        return self._conv_to_prim_matrix

    def _check_spg_output(self):
        """
        check spglib output
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
            x_s_ = (np.dot(P, np.transpose(x)).T + p) % 1
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


        atol = 1e-8

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
                              symprec:float=1.e-5,
                              no_sort:bool=False,
                              get_sort_list:list=False):
        """
        get stadandardized cell

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
        std_cell =  spglib.standardize_cell(self._cell_for_spglib,
                                            to_primitive=to_primitive,
                                            no_idealize=no_idealize)
        if no_sort:
            return std_cell
        else:
            num_atoms, unique_symbols, scaled_positions, sort_list = \
                sort_positions_by_symbols(
                        symbols=get_chemical_symbols(std_cell[2]),
                        positions=std_cell[1])
            symbols = []
            for num, symbol in zip(num_atoms, unique_symbols):
                symbols.extend([symbol] * num)

            sort_std_cell = (std_cell[0], scaled_positions, symbols)
            if get_sort_list:
                return (sort_std_cell, sort_list)
            else:
                return sort_std_cell
