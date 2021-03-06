#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module deals with hexagonal twin structure.
"""

from pprint import pprint
import numpy as np
from phonopy.structure.atoms import atom_data, symbol_map
from twinpy.properties.hexagonal import (get_hcp_atom_positions,
                                         check_cell_is_hcp)
from twinpy.properties.twinmode import TwinIndices
from twinpy.structure.lattice import CrystalLattice


def get_numbers_from_symbols(symbols:list):
    """
    Get atomic numbers from symbols.

    Args:
        symbols: Atomic symbols.
    """
    numbers = [ symbol_map[symbol] for symbol in symbols ]
    return numbers


def get_symbols_from_numbers(numbers:list):
    """
    Get symbols from atomic numbers.

    Args:
        numbers: Atomic numbers.
    """
    symbols = [ atom_data[number][1] for number in numbers ]
    return symbols


def check_same_cells(first_cell:tuple,
                     second_cell:tuple,
                     raise_error:bool=False,
                     atol:float=1e-6) -> bool:
    """
    Check first cell and second cell are same.

    Args:
        first_cell: First cell.
        second_cell: Second cell.
        raise_error: If True, raise assrtion error.

    Returns:
        bool: Return True if two cells are same.
    """
    is_lattice_same = np.allclose(first_cell[0], second_cell[0], atol=atol)
    is_scaled_positions_same = np.allclose(
            first_cell[1], second_cell[1], atol=atol)
    is_symbols_same = (first_cell[2] == second_cell[2])
    is_same = (is_lattice_same
               and is_scaled_positions_same
               and is_symbols_same)
    if not is_same and raise_error:
        np.testing.assert_allclose(first_cell[0], second_cell[0], atol=atol)
        np.testing.assert_allclose(first_cell[1], second_cell[1], atol=atol)
        assert (first_cell[2] == second_cell[2])

    return is_same


def get_atom_positions_from_lattice_points(lattice_points:np.array,
                                           atoms_from_lp:np.array) -> np.array:
    """
    Get atom positions by embedding primitive atoms to lattice points.
    Both lattice points and atom positions must be cartesian coordinates.

    Args:
        lattice_points: Lattice points.
        atoms_from_lp: Atoms from lattice_points.

    Returns:
        np.array: atom positions
    """
    scaled_positions = []
    for lattice_point in lattice_points:
        atoms = lattice_point + atoms_from_lp
        scaled_positions.extend(atoms.tolist())
    return np.array(scaled_positions)


class _BaseTwinStructure():
    """
    Base structure class which is inherited
    ShearStructure class and TwinBoundaryStructure class.
    """

    def __init__(
           self,
           lattice:np.array,
           symbol:str,
           twinmode:str,
           wyckoff:str='c',
           ):
        """
        Setup.

        Args:
            lattice: Lattice.
            symbol: Element symbol.
            twinmode: Twin mode.
            wyckoff: No.194 Wycoff letter ('c' or 'd').

        Todo:
            Check it is best to use 'deepcopy'.
        """
        atoms_from_lp = get_hcp_atom_positions(wyckoff)
        symbols = [symbol] * 2
        crylat = CrystalLattice(lattice=lattice)
        check_cell_is_hcp(cell=(lattice, atoms_from_lp, symbols))
        self._hexagonal_lattice = lattice
        self._a, _, self._c = crylat.abc
        self._r = self._c / self._a
        self._symbol = symbol
        self._wyckoff = wyckoff
        self._atoms_from_lattice_points = \
                get_hcp_atom_positions(wyckoff=self._wyckoff)
        self._natoms = 2
        self._twinmode = None
        self._indices = None
        self._set_twinmode(twinmode=twinmode)
        self._xshift = None
        self._yshift = None
        self._expansion_ratios = np.ones(3)
        self._output_structure = \
                {'lattice': self._hexagonal_lattice,
                 'lattice_points': {
                     'white': np.array([0.,0.,0.])},
                 'atoms_from_lattice_points': {
                     'white': self._atoms_from_lattice_points},
                 'symbols': [self._symbol] * 2}

    def _set_twinmode(self, twinmode:str):
        """
        Set parent.

        Args:
            twinmode: Twinmode.
        """
        self._twinmode = twinmode
        self._indices = TwinIndices(twinmode=self._twinmode,
                                    lattice=self._hexagonal_lattice,
                                    wyckoff=self._wyckoff)

    @property
    def r(self):
        """
        Lattice ratio r  = c / a .
        """
        return self._r

    @property
    def symbol(self):
        """
        Symbol.
        """
        return self._symbol

    @property
    def wyckoff(self):
        """
        Wyckoff letter.
        """
        return self._wyckoff

    @property
    def atoms_from_lattice_points(self):
        """
        Atoms from lattice points.
        """
        return self._atoms_from_lattice_points

    @property
    def hexagonal_lattice(self):
        """
        Hexagonal lattice.
        """
        return self._hexagonal_lattice

    @property
    def xshift(self):
        """
        Structure x shift.
        """
        return self._xshift

    @property
    def yshift(self):
        """
        Structure y shift.
        """
        return self._yshift

    @property
    def natoms(self):
        """
        Number of atoms.
        """
        return self._natoms

    @property
    def twinmode(self):
        """
        Twinmode.
        """
        return self._twinmode

    @property
    def indices(self):
        """
        Indices of twinmode.
        """
        return self._indices

    @property
    def expansion_ratios(self):
        """
        Expansion ratios which is applied when you run 'get_cell_for_export'.
        """
        return self._expansion_ratios

    def set_expansion_ratios(self, expansion_ratios:np.array):
        """
        Set expansion ratios which is applied
        when you run 'get_cell_for_export'.

        Args:
        """
        _expansion_ratios = expansion_ratios.copy()
        if not isinstance(expansion_ratios, np.ndarray):
            _expansion_ratios = np.array(_expansion_ratios)
        assert _expansion_ratios.shape == (3,), \
                 "Shape of expansion_ratios is {}, which must be (3,)".format(
                         np.array(_expansion_ratios).shape)
        self._expansion_ratios = _expansion_ratios

    @property
    def output_structure(self):
        """
        Built structure.
        """
        return self._output_structure

    def get_cell_for_export(self,
                            get_lattice:bool=False,
                            move_atoms_into_unitcell:bool=True,
                            ) -> tuple:
        """
        Get cell for export.

        Args:
            get_lattice: Get lattice points not crystal structure.
            move_atoms_into_unitcell: if True, move atoms to unitcell.

        Returns:
            tuple: Output cell.

        Notes:
            Lattice matrix is expanded using expansion_ratios.
            You have to run set_expansion_ratios before run this function
            if you want to expand lattice.
        """
        _dummy = {'white': 'H', 'white_tb': 'Li',
                  'black': 'He', 'black_tb': 'Be'}
        scaled_positions = []
        if get_lattice:
            symbols = []
            for color in self._output_structure['lattice_points']:
                posi = self._output_structure['lattice_points'][color]
                sym = [_dummy[color]] * len(posi)
                scaled_positions.extend(posi.tolist())
                symbols.extend(sym)
            print("replacing lattice points to elements:")
            print("    'white'   : 'H'")
            print("    'white_tb': 'Li'")
            print("    'black'   : 'He'")
            print("    'black_tb': 'Be'")
        else:
            for color in self._output_structure['lattice_points']:
                posi = get_atom_positions_from_lattice_points(
                    self._output_structure['lattice_points'][color],
                    self._output_structure['atoms_from_lattice_points'][color])
                scaled_positions.extend(posi.tolist())
            scaled_positions = np.round(np.array(scaled_positions), decimals=8)

            if move_atoms_into_unitcell:
                scaled_positions %= 1.

            symbols = self._output_structure['symbols']

        lattice = np.transpose(
                      np.transpose(self._output_structure['lattice'])
                      * self._expansion_ratios)

        return (lattice,
                scaled_positions,
                symbols)
