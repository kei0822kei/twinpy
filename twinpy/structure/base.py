#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HexagonalStructure
"""

import numpy as np
from scipy.linalg import sqrtm
import spglib
from phonopy.structure.atoms import PhonopyAtoms, symbol_map
from phonopy.structure.cells import Primitive, Supercell
from pymatgen.core.structure import Structure
from typing import Sequence, Union
from twinpy.common.utils import get_ratio
from twinpy.properties.hexagonal import get_atom_positions
from twinpy.properties.twinmode import TwinIndices
from twinpy.lattice.lattice import Lattice
from twinpy.lattice.hexagonal_plane import HexagonalPlane
from twinpy.file_io import write_poscar
from twinpy.structure.standardize import StandardizeCell

def is_hcp(lattice:np.array,
           symbols:Sequence[str],
           positions:np.array=None,
           scaled_positions:np.array=None,
           get_wyckoff:bool=False):
    """
    check input structure is Hexagonal Close-Packed structure

    Args:
        lattice (np.array): lattice
        symbols: atomic symbols
        positions (np.array): atom cartesian positions
        scaled_positions (np.array): atom fractional positions
        get_wyckoff (bool): if True, return wyckoff letter, which is 'c' or 'd'

    Raises:
        AssertionError: both positions and scaled_positions are specified
        AssertionError: input symbols are not unique
        AssertionError: input structure is not
                        Hexagonal Close-Packed structure
    """
    if positions is not None and scaled_positions is not None:
        raise AssertionError("both positions and scaled_positions "
                             "are specified")

    assert (len(set(symbols)) == 1 and len(symbols) == 2), \
        "symbols is not unique or the number of atoms are not two"

    if positions is not None:
        scaled_positions = np.dot(np.linalg.inv(lattice.T), positions.T).T
    dataset = spglib.get_symmetry_dataset((lattice,
                                           scaled_positions,
                                           [0, 0]))
    spg_symbol = dataset['international']
    wyckoffs = dataset['wyckoffs']
    assert spg_symbol == 'P6_3/mmc', \
            "space group of input structure is {} not 'P6_3/mmc'" \
            .format(spg_symbol)
    assert wyckoffs in [['c'] * 2, ['d'] * 2]
    if get_wyckoff:
        return wyckoffs[0]

def get_hexagonal_structure_from_pymatgen(pmgstructure):
    """
    get HexagonalStructure object from pyamtgen structure

    Args:
        pmgstructure: pymatgen structure object
    """
    lattice = pmgstructure.lattice.matrix
    scaled_positions = pmgstructure.frac_coords
    symbols = [ specie.value for specie in pmgstructure.species ]
    wyckoff = is_hcp(lattice=lattice,
                     scaled_positions=scaled_positions,
                     symbols=symbols,
                     get_wyckoff=True)
    return (lattice, scaled_positions, symbols[0], wyckoff)

def get_atom_positions_from_lattice_points(lattice_points:np.array,
                                           atoms_from_lp:np.array):
    """
    get atom positions from lattice points

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

def get_lattice_points_from_supercell(lattice, dim) -> np.array:
    """
    get lattice points from supercell

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

def reshape_dimension(dim):
    """
    if dim.shape == (3,), reshape to (3,3) numpy array

    Raises:
        ValueError: input dim is not (3,) and (3,3) array
    """
    if np.array(dim).shape == (3,3):
        dim_matrix =np.array(dim)
    elif np.array(dim).shape == (3,):
        dim_matrix = np.diag(dim)
    else:
        raise ValueError("input dim is not (3,) and (3,3) array")
    return dim_matrix

def get_phonopy_structure(cell,
                          structure_type:str='base',
                          symprec:float=1e-5):
    """
    return phonopy structure

    Args:
        cell: tuple (lattice, scaled_positions, symbols)
        structure_type (str): 'base', 'primitive' or 'conventional'
        symprec (float): used when searching conventional unitcell
    """
    if structure_type not in ['base', 'primitive', 'conventional']:
        msg = "structure_type must be 'base', 'primitive' or 'conventional'"
        raise RuntimeError(msg)

    fixed_cell = None
    if structure_type == 'base':
        fixed_cell = cell
    else:
        if structure_type == 'primitive':
            to_primitive = True
        else:
            to_primitive = False
        std = StandardizeCell(cell)
        fixed_cell = std.get_standardized_cell(to_primitive=to_primitive)
    ph_structure = PhonopyAtoms(cell=fixed_cell[0],
                                scaled_positions=fixed_cell[1],
                                symbols=fixed_cell[2])
    return ph_structure

def get_cell_from_phonopy_structure(ph_structure):
    """
    get cell from phonopy structure
    """
    lattice = ph_structure.get_cell()
    scaled_positions = ph_structure.get_scaled_positions()
    symbols = ph_structure.get_chemical_symbols()
    return (lattice, scaled_positions, symbols)

class _BaseStructure():
    """
    base structure class which is inherited
    ShearStructure class and TwinBoundaryStructure class
    """
    def __init__(
           self,
           lattice:np.array,
           symbol:str,
           twinmode:str,
           wyckoff:str='c',
        ):
        """
        initialize

        Args:
            lattice (np.array): lattice
            symbol (str): element symbol
            wyckoff (str): No.194 Wycoff position ('c' or 'd')

        Raises:
            AssertionError: wyckoff is not 'c' and 'd'
            ValueError: lattice is not None (future fix)
        """
        atoms_from_lp = get_atom_positions(wyckoff)
        symbols = [symbol] * 2
        is_hcp(lattice=lattice,
               scaled_positions=atoms_from_lp,
               symbols=symbols)
        self._hcp_lattice = Lattice(lattice)
        self._a, _, self._c = self._hcp_lattice.get_abc()
        self._r = self._c / self._a
        self._symbol = symbol
        self._wyckoff = wyckoff
        self._atoms_from_lattice_points = \
                get_atom_positions(wyckoff=self._wyckoff)
        self._hexagonal_lattice = Lattice(lattice)
        self._natoms = 2
        self._dim = None
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
        set parent

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
        r ( = c / a )
        """
        return self._r

    @property
    def hcp_lattice(self):
        """
        base HCP lattice
        """
        return self._hcp_lattice

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
    def dim(self):
        """
        dimension
        """
        return self._dim

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
    def xshift(self):
        """
        x shift
        """
        return self._xshift

    @property
    def yshift(self):
        """
        x shift
        """
        return self._yshift

    @property
    def natoms(self):
        """
        number of atoms
        """
        return self._natoms

    @property
    def twinmode(self):
        """
        twinmode
        """
        return self._twinmode

    @property
    def indices(self):
        """
        indices of twinmode
        """
        return self._indices

    @property
    def output_structure(self):
        """
        built structure
        """
        return self._output_structure

    def get_structure_for_export(self,
                                 get_lattice=False,
                                 move_atoms_into_unitcell=True):
        """
        get structure for export

        Args:
            get_lattice (str): get lattice points not crystal structure
            move_atoms_into_unitcell (bool): if True, move atoms to unitcell

        Returns:
            tuple: output cell
        """
        _dummy = {'white': 'H', 'white_tb': 'H',
                  'black': 'He', 'black_tb': 'He', 'grey': 'Li'}
        scaled_positions = []
        if get_lattice:
            symbols = []
            for color in self._output_structure['lattice_points']:
                posi = self._output_structure['lattice_points'][color]
                sym = [_dummy[color]] * len(posi)
                scaled_positions.extend(posi.tolist())
                symbols.extend(sym)
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

    def get_pymatgen_structure(self,
                               get_lattice=False,
                               move_atoms_into_unitcell=True):
        """
        get pymatgen structure
        """
        lattice, scaled_positions, species = \
                self.get_structure_for_export(
                        get_lattice=get_lattice,
                        move_atoms_into_unitcell=move_atoms_into_unitcell)
        return Structure(lattice=lattice,
                         coords=scaled_positions,
                         species=species)

    def write_poscar(self,
                     filename:str='POSCAR',
                     get_lattice=False,
                     move_atoms_into_unitcell=True):
        """
        write poscar

        Args:
            filename (str): output filename
        """
        lattice, scaled_positions, symbols = \
                self.get_structure_for_export(
                        get_lattice=get_lattice,
                        move_atoms_into_unitcell=move_atoms_into_unitcell)
        write_poscar(lattice=lattice,
                     scaled_positions=scaled_positions,
                     symbols=symbols,
                     filename=filename)

    # def get_phonopy_structure(self, structure_type:str='base', symprec:float=1e-5):
    #     """
    #     return phonopy structure

    #     Args:
    #         structure_type (str): 'base', 'primitive' or 'conventional'
    #         symprec (float): used when searching conventional unitcell
    #     """
    #     cell = self.get_structure_for_export(get_lattice=False)
    #     ph_structure = get_phonopy_structure(cell, structure_type, symprec)
    #     return ph_structure
