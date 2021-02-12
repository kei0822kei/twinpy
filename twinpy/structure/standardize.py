#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module deals with crystal symmetry.
"""

import numpy as np
import spglib
from phonopy.interface.vasp import sort_positions_by_symbols
from twinpy.structure.base import (get_numbers_from_symbols,
                                   get_symbols_from_numbers)


def get_standardized_cell(cell:tuple,
                          to_primitive:bool,
                          no_idealize:bool,
                          symprec:float=1e-5,
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
                                         no_idealize=no_idealize,
                                         symprec=symprec,
                                         no_sort=no_sort,
                                         get_sort_list=get_sort_list)
    return std_cell


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
        self._atomic_numbers = get_numbers_from_symbols(self._cell[2])
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
                            decimals=7) % 1
            for _x in x_s_:
                # assert x_s_[i] in x_s, \
                assert np.round(_x, decimals=7) \
                       in np.round(x_s, decimals=7), \
                       "x_s != Px+p, check script"
            # check x_p = P_c^{-1} x_s
            x_p_ = (np.dot(np.linalg.inv(P_c), np.transpose(x_s)).T) % 1
            for _x in x_p_:
                assert np.round(_x, decimals=7) \
                       in np.round(x_p, decimals=7), \
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
        symbols = get_symbols_from_numbers(spg_cell[2])
        std_cell = (spg_cell[0], spg_cell[1], symbols)

        if no_sort:
            return std_cell

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

        return sort_std_cell

    def convert_kpoints(self, kpoints:tuple, kpoints_type:str) -> tuple:
        """
        Convert input kpoints to required kpoints.

        Args:
            kpoints (tuple): Contains (mesh, offset).
            kpoints_type (str): Input kpoints type.
                                Choose 'original' or 'primitive'.
        Notes:
            'primitive' in this function means 'primitive idealized'.
        """
        def __get_kpt_transformation_matrix():
            M_p = self.get_standardized_cell(to_primitive=True,
                                             no_idealize=True)[0].T
            M = self.cell[0].T
            mat = np.dot(M_p.T,
                         np.linalg.inv(M).T)
            return mat

        convert_mat = __get_kpt_transformation_matrix()
        if kpoints_type == 'original':
            mesh = np.round(np.abs(np.dot(convert_mat, kpoints[0])),
                            decimals=4).astype(int).tolist()
            offset = np.round(np.abs(np.dot(convert_mat, kpoints[1])),
                              decimals=4).tolist()
            kpts = {
                    'original': kpoints,
                    'primitive': (mesh, offset),
                    }
        elif kpoints_type == 'primitive':
            mesh = np.round(np.abs(np.dot(np.linalg.inv(convert_mat),
                                          kpoints[0])),
                            decimals=4).astype(int)
            offset = np.round(np.abs(np.dot(np.linalg.inv(convert_mat),
                                            kpoints[1])),
                              decimals=4)
            kpts = {
                    'original': (mesh, offset),
                    'primitive': kpoints,
                    }
        else:
            raise RuntimeError("input kpoints_type: {}, which is prohibited."
                               .format(kpoints_type))

        return kpts
