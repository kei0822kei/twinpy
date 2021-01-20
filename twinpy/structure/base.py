#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HexagonalStructure
"""

import numpy as np
import spglib
from phonopy.structure.atoms import PhonopyAtoms, atom_data, symbol_map
from phonopy.structure.cells import Supercell
from twinpy.properties.hexagonal import get_atom_positions
from twinpy.properties.twinmode import TwinIndices
from twinpy.lattice.lattice import get_hexagonal_lattice_from_a_c, Lattice


def get_numbers_from_symbols(symbols):
    """
    Get atomic numbers from symbols.

    Args:
        symbols (list): Atomic symbols.
    """
    numbers = [ symbol_map[symbol] for symbol in symbols ]
    return numbers


def get_symbols_from_numbers(numbers):
    """
    Get symbols from atomic numbers.

    Args:
        numbers (list): Atomic numbers.
    """
    symbols = [ atom_data[number][1] for number in numbers ]
    return symbols


def check_same_cells(first_cell:tuple,
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


def get_atom_positions_from_lattice_points(lattice_points:np.array,
                                           atoms_from_lp:np.array) -> np.array:
    """
    Get atom positions from lattice points.
    Both lattice points and atom positions must be cartesian coordinates.

    Args:
        lattice_points (np.array): lattice points
        atoms_from_lp (np.array): atoms from lattice_points

    Returns:
        np.array: atom positions
    """
    scaled_positions = []
    for lattice_point in lattice_points:
        atoms = lattice_point + atoms_from_lp
        scaled_positions.extend(atoms.tolist())
    return np.array(scaled_positions)


def get_lattice_points_from_supercell(lattice:np.array,
                                      dim:np.array) -> np.array:
    """
    Get lattice points from supercell.

    Args:
        lattice (np.array): lattice
        dim (np.array): dimension, its shape is (3,) or (3,3)

    Returns:
        np.array: lattice points
    """
    unitcell = PhonopyAtoms(symbols=['H'],
                            cell=lattice,
                            scaled_positions=np.array([[0.,0.,0]]),
                            )
    super_lattice = Supercell(unitcell=unitcell,
                              supercell_matrix=reshape_dimension(dim))
    lattice_points = super_lattice.scaled_positions
    return lattice_points


def reshape_dimension(dim:np.array) -> np.array:
    """
    If dim.shape == (3,), reshape to (3,3) numpy array.

    Raises:
        ValueError: input dim is not (3,) and (3,3) array

    Returns:
        np.array: 3x3 dimention matrix
    """
    if np.array(dim).shape == (3,3):
        dim_matrix = np.array(dim)
    elif np.array(dim).shape == (3,):
        dim_matrix = np.diag(dim)
    else:
        raise ValueError("input dim is not (3,) and (3,3) array")
    return dim_matrix


class _BaseStructure():
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
        Args:
            lattice (np.array): lattice
            symbol (str): element symbol
            twinmode (str): twin mode
            wyckoff (str): No.194 Wycoff position ('c' or 'd')
        """
        atoms_from_lp = get_atom_positions(wyckoff)
        symbols = [symbol] * 2
        is_hcp(lattice=lattice,
               scaled_positions=atoms_from_lp,
               symbols=symbols)
        self._hcp_lattice = Lattice(lattice)
        self._a, _, self._c = self._hcp_lattice.abc
        self._r = self._c / self._a
        self._symbol = symbol
        self._wyckoff = wyckoff
        self._atoms_from_lattice_points = \
                get_atom_positions(wyckoff=self._wyckoff)
        self._hexagonal_lattice = Lattice(lattice)
        self._natoms = 2
        self._twinmode = None
        self._indices = None
        self._set_twinmode(twinmode=twinmode)
        self._xshift = None
        self._yshift = None
        self._output_structure = \
                {'lattice': lattice,
                 'lattice_points': {
                     'white': np.array([0.,0.,0.])},
                 'atoms_from_lattice_points': {
                     'white': self._atoms_from_lattice_points},
                 'symbols': [self._symbol] * 2}

    def _set_twinmode(self, twinmode:str):
        """
        Set parent.

        Args:
            twinmode (str): twinmode

        Note:
            set attribute 'twinmode'
            set attribute 'indices'
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
        Wyckoff position.
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
            get_lattice (str): get lattice points not crystal structure
            move_atoms_into_unitcell (bool): if True, move atoms to unitcell

        Returns:
            tuple: output cell
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
            scaled_positions = np.array(scaled_positions)

            if move_atoms_into_unitcell:
                scaled_positions = scaled_positions % 1.

            symbols = self._output_structure['symbols']
        return (self._output_structure['lattice'],
                scaled_positions,
                symbols)
