#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HexagonalStructure
"""

import numpy as np
import spglib
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive, Supercell
from typing import Union
from twinpy.common.utils import get_ratio
from twinpy.properties.hexagonal import get_atom_positions
from twinpy.properties.twinmode import (get_shear_strain_function,
                                        get_twin_indices)
from twinpy.lattice.lattice import Lattice
from twinpy.lattice.hexagonal_plane import HexagonalPlane

def is_hcp(lattice, positions, symbols):
    """
    check input structure is Hexagonal Close-Packed structure

    Args:
        lattice (np.array): lattice
        positions: atom fractional positions
        symbols (list): atomic symbols

    Raises:
        AssertionError: input symbols are not unique
        AssertionError: input structure is not
                        Hexagonal Close-Packed structure
    """
    assert (len(set(symbols)) == 1 and len(symbols) == 2), \
        "symbols is not unique or the number of atoms are not two"
    dataset = spglib.get_symmetry_dataset((lattice,
                                           positions,
                                           [0, 0]))
    spg_symbol = dataset['international']
    wyckoffs = dataset['wyckoffs']
    assert spg_symbol == 'P6_3/mmc', \
            "space group of input structure is {} not 'P6_3/mmc'" \
            .format(spg_symbol)
    assert wyckoffs in [['c'] * 2, ['d'] * 2]

def _get_supercell_matrix(indices):
    tf1 = np.array(get_ratio(indices['m'].three))
    tf2 = np.array(get_ratio(indices['eta1'].three))
    tf3 = np.array(get_ratio(indices['eta2'].three))
    supercell_matrix = np.vstack([tf1, tf2, tf3]).T
    return supercell_matrix

class HexagonalStructure():
    """
    deals with hexagonal close-packed structure
    """

    def __init__(
           self,
           a:float,
           c:float,
           symbol:str,
           wyckoff:str='c',
           lattice:np.array=None,
        ):
        """
        Args:
            a (str): the norm of a axis
            c (str): the norm of c axis
            symbol (str): element symbol
            lattice (np.array): if None, default lattice is set
            wyckoff (str): No.194 Wycoff position ('c' or 'd')

        Raises:
            AssertionError: either a or c is negative value
            AssertionError: wyckoff is not 'c' and 'd'
            ValueError: lattice is not None (future fix)
        """
        assert a > 0. and c > 0., "input 'a' and 'c' must be positive value"
        if lattice is None:
            lattice = np.array([[  1.,           0., 0.],
                                [-0.5, np.sqrt(3)/2, 0.],
                                [  0.,           0., 1.]]) * np.array([a,a,c])
        else:
            raise ValueError("lattice option is currently invalid"
                             " (write future)")
        atoms_from_lp = get_atom_positions(wyckoff)
        symbols = [symbol] * 2
        self._r = c / a
        self._symbol = symbol
        self._wyckoff = wyckoff
        self._atoms_from_lattice_points = \
                get_atom_positions(wyckoff=self._wyckoff)
        self._hexagonal_lattice = Lattice(lattice)
        self._indices = None
        self._parent_matrix = np.eye(3)
        self._shear_strain_funcion = None
        self._shear_strain_ratio = 0.
        self._build = None

    def _parent_not_set(self):
        if self._parent_matrix is None:
            raise RuntimeError("parent_matrix is not set"
                               "run set_parent first")

    @property
    def r(self):
        """
        r ( = c / a )
        """
        return self._r

    @property
    def symbol(self):
        """
        symbol
        """
        return self._symbol

    @property
    def wyckoff(self):
        """
        wyckoff position
        """
        return self._wyckoff

    @property
    def atoms_from_lattice_points(self):
        """
        atoms from lattice points
        """
        return self._atoms_from_lattice_points

    @property
    def hexagonal_lattice(self):
        """
        hexagonal lattice
        """
        return self._hexagonal_lattice

    @property
    def indices(self):
        """
        indices of twinmode
        """
        return self._indices

    @property
    def parent_matrix(self):
        """
        parent matrix
        """
        return self._parent_matrix

    @property
    def shear_strain_function(self):
        """
        shear shear strain function
        """
        return self._shear_strain_funcion

    @property
    def shear_strain_ratio(self):
        """
        shear shear strain ratio
        """
        return self._shear_strain_ratio

    @property
    def build(self):
        """
        build structure
        """
        return self._build

    def _get_shear_matrix(self):
        plane = HexagonalPlane(lattice=self._hexagonal_lattice.lattice,
                               four=self._indices['K1'].four)
        d = plane.get_distance_from_plane(self._indices['eta2'].three)
        gamma = self._shear_strain_funcion(self._r)
        norm_eta1 = plane.get_cartesian(self._indices['eta1'].three)
        s = gamma * d / norm_eta1

        shear_matrix = np.eye(3)
        shear_matrix[1,2] = self._shear_strain_ratio * s
        return shear_matrix

    def set_parent(self, twinmode:str):
        """
        set parent

        Args:
            twinmode (str): twinmode

        Note:
            set attribute 'indices'
            set attribute 'parent_matrix'
            set attribute 'shear_function'
        """
        self._indices = get_twin_indices(twinmode=twinmode,
                                         lattice=self._hexagonal_lattice,
                                         wyckoff=self._wyckoff)
        self._parent_matrix = _get_supercell_matrix(self._indices)
        self._shear_strain_funcion = get_shear_strain_function(twinmode)

    def set_shear_ratio(self, ratio):
        """
        set shear ratio

        Args:
            ratio (float): the ratio of shear value

        Note:
            set attribute 'shear_strain_ratio'
        """
        self._shear_strain_ratio = ratio

    def build(self, is_primitive=False):
        """
        build structure

        Args:
            style (str): determine the structure style,
              choose from 'tuple'
        """
        lattice_points = np.array([[0.,0.,0.]])
        unitcell = PhonopyAtoms(symbols=['X'],
                        cell=self._hexagonal_lattice.lattice,
                        scaled_positions=lattice_points)

        if is_primitive:
            shear_matrix = self._get_shear_matrix()
            shear_lattice = \
                    np.dot(self._hexagonal_lattice.lattice.T,
                           np.dot(self._parent_matrix,
                                  np.dot(shear_matrix,
                                         np.linalg.inv(self._parent_matrix))))
            self._build = (shear_lattice,
                           lattice_points,
                           self._atoms_from_lattice_points,
                           self._symbol)
        else:
            # super_lattice = Supercell(unitcell=unitcell,
            #                           supercell_matrix=self._parent_matrix)
            raise ValueError("future write is the case of 'is_primitive=False'")
