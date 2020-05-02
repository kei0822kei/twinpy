#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HexagonalStructure
"""

import numpy as np
from scipy.linalg import sqrtm
import spglib
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive, Supercell
from pymatgen.core.structure import Structure
from typing import Sequence, Union
from twinpy.common.utils import get_ratio
from twinpy.properties.hexagonal import get_atom_positions
from twinpy.properties.twinmode import (get_shear_strain_function,
                                        get_twin_indices)
from twinpy.lattice.lattice import Lattice
from twinpy.lattice.hexagonal_plane import HexagonalPlane
from twinpy.file_io import write_poscar

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
    return HexagonalStructure(lattice=lattice,
                              symbol=symbols[0],
                              wyckoff=wyckoff)


def get_hexagonal_structure_from_a_c(a:float,
                                     c:float,
                                     symbol:str=None,
                                     wyckoff:str='c'):
    """
    get HexagonalStructure class object from a and c axes

    Args:
        a (str): the norm of a axis
        c (str): the norm of c axis
        symbol (str): element symbol
        wyckoff (str): No.194 Wycoff position ('c' or 'd')

    Raises:
        AssertionError: either a or c is negative value
    """
    assert a > 0. and c > 0., "input 'a' and 'c' must be positive value"
    lattice = np.array([[  1.,           0., 0.],
                        [-0.5, np.sqrt(3)/2, 0.],
                        [  0.,           0., 1.]]) * np.array([a,a,c])
    return HexagonalStructure(lattice=lattice,
                              symbol=symbol,
                              wyckoff=wyckoff)


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
           lattice:np.array,
           symbol:str,
           wyckoff:str='c',
        ):
        """
        Args:
            lattice (np.array): lattice
            symbol (str): element symbol
            wyckoff (str): No.194 Wycoff position ('c' or 'd')

        Raises:
            AssertionError: wyckoff is not 'c' and 'd'
            ValueError: lattice is not None (future fix)
        """
        norms = np.linalg.norm(lattice, axis=1)
        atoms_from_lp = get_atom_positions(wyckoff)
        symbols = [symbol] * 2
        is_hcp(lattice=lattice,
               scaled_positions=atoms_from_lp,
               symbols=symbols)
        self._a = norms[0]
        self._c = norms[2]
        self._r = self._c / self._a
        self._symbol = symbol
        self._wyckoff = wyckoff
        self._atoms_from_lattice_points = \
                get_atom_positions(wyckoff=self._wyckoff)
        self._hexagonal_lattice = Lattice(lattice)
        self._indices = None
        self._parent_matrix = np.eye(3)
        self._shear_strain_funcion = None
        self._shear_strain_ratio = 0.
        self._output_structure = (lattice,
                                  np.array([0.,0.,0.]),
                                  self._atoms_from_lattice_points,
                                  [self._symbol] * 2)

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
    def output_structure(self):
        """
        built structure
        """
        return self._output_structure

    def _get_shear_matrix(self):
        s = self._get_shear_value()
        shear_matrix = np.eye(3)
        shear_matrix[1,2] = self._shear_strain_ratio * s
        return shear_matrix

    def _get_shear_value(self):
        plane = HexagonalPlane(lattice=self._hexagonal_lattice.lattice,
                               four=self._indices['K1'].four)
        d = plane.get_distance_from_plane(self._indices['eta2'].three)
        gamma = self._shear_strain_funcion(self._r)
        norm_eta1 = np.linalg.norm(plane.get_cartesian(self._indices['eta1'].three))
        s = abs(gamma) * d / norm_eta1
        return s

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

    def get_shear_properties(self) -> dict:
        """
        get various properties related to shear

        Note:
            key variable:
              - shear value (s)
              - shear ratio (alpha)
              - strain matrix (S)
              - deformation gradient tensor (F)
              - right Cauchy-Green tensor (C)
              - left Cauchy-Green tensor (b)
              - matrial stretch tensor (U)
              - spatial stretch tensor (V)
              - rotation (R)
            for more detail and definition, see documentation
        """
        H = self.hexagonal_lattice.lattice.T
        M = self._parent_matrix
        S = self._get_shear_matrix()
        F = np.eye(3) + \
            np.dot(H,
                   np.dot(M,
                          np.dot(S,
                                 np.dot(np.linalg.inv(M),
                                        np.linalg.inv(H)))))
        s = self._get_shear_value()
        alpha = self._shear_strain_ratio
        C = np.dot(F.T, F)
        b = np.dot(F, F.T)
        U = sqrtm(C)
        V = sqrtm(b)
        R = np.dot(F, np.linalg.inv(U))
        R_ = np.dot(np.linalg.inv(V), F)
        np.testing.assert_allclose(R, R_)
        np.testing.assert_allclose(np.linalg.norm(R, axis=0),
                                   np.array([1,1,1]))
        np.testing.assert_allclose(np.linalg.norm(R, axis=1),
                                   np.array([1,1,1]))
        return {
                 'shear_value': s,
                 'shear_ratio': alpha,
                 'strain_matrix': S,
                 'deformation_gradient_tensor': F,
                 'material_stretch_tensor': U,
                 'spatial_stretch_tensor': V,
                 'right_Cauchy': C,
                 'left_Cauchy': b,
                 'rotation':R,
               }

    def run(self, is_primitive=False):
        """
        build structure

        Args:
            is_primitive (bool): if True, primitive structure is build
              choose from 'tuple'

        Note:
            the structure built is set self.output_structure
        """
        shear_matrix = self._get_shear_matrix()
        if is_primitive:
            lattice_points = np.array([[0.,0.,0.]])
            atoms_from_lattice_points = self.atoms_from_lattice_points.copy()
            shear_lattice = \
                np.dot(self._hexagonal_lattice.lattice.T,
                       np.dot(self._parent_matrix,
                              np.dot(shear_matrix,
                                     np.linalg.inv(self._parent_matrix)))).T
        else:
            unitcell = PhonopyAtoms(symbols=['H'],
                            cell=self._hexagonal_lattice.lattice,
                            scaled_positions=np.array([[0.,0.,0]]),
                            )
            super_lattice = Supercell(unitcell=unitcell,
                                      supercell_matrix=self._parent_matrix)
            shear_lattice = \
                np.dot(self._hexagonal_lattice.lattice.T,
                       np.dot(self._parent_matrix,
                              shear_matrix)).T
            lattice_points = super_lattice.scaled_positions
            atoms_from_lattice_points = np.dot(
                    np.linalg.inv(self._parent_matrix),
                    self._atoms_from_lattice_points.T,
                    ).T
        symbols = [self._symbol] * len(lattice_points) \
                                 * len(self._atoms_from_lattice_points)
        self._output_structure = (shear_lattice,
                                  lattice_points,
                                  atoms_from_lattice_points,
                                  symbols)

    def get_pymatgen_structure(self):
        """
        get pymatgen structure
        """
        scaled_positions = get_atom_positions_from_lattice_points(
                self._output_structure[1],
                self._output_structure[2])
        return Structure(lattice=self._hexagonal_lattice.lattice,
                         coords=scaled_positions,
                         species=[self._symbol] * len(scaled_positions))

    def get_poscar(self, filename:str='POSCAR'):
        """
        get poscar

        Args:
            filename (str): output filename
        """
        frac_coords = get_atom_positions_from_lattice_points(
                self._output_structure[1],
                self._output_structure[2])
        write_poscar(lattice=self.output_structure[0],
                     frac_coords=np.array(frac_coords),
                     symbols=self._output_structure[3],
                     filename=filename)
