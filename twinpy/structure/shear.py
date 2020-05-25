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
from twinpy.properties.twinmode import TwinIndices
from twinpy.lattice.lattice import Lattice
from twinpy.lattice.hexagonal_plane import HexagonalPlane
from twinpy.file_io import write_poscar
from twinpy.structure.hexagonal import (get_lattice_points_from_supercell,
                                        _StructureBase)

class ShearStructure(_StructureBase):
    """
    shear structure class
    """

    def __init__(
           self,
           lattice:np.array,
           symbol:str,
           wyckoff:str='c',
        ):
        """
        initialize

        Note:
            to see detail, visit _BaseStructure class
        """
        super().__init__(lattice=lattice,
                         symbol=symbol,
                         wyckoff=wyckoff)
        self._dim = np.ones(3, dtype=int)
        self._xshift = 0.
        self._yshift = 0.
        self._shear_strain_ratio = 0.

    @property
    def dim(self):
        """
        dimension
        """
        return self._dim

    def set_dim(self, dim):
        """
        set dimension
        """
        self._dim = dim

    @property
    def shear_strain_ratio(self):
        """
        shear shear strain ratio
        """
        return self._shear_strain_ratio

    def set_shear_ratio(self, ratio):
        """
        set shear ratio
        """
        self._shear_strain_ratio = ratio

    @property
    def xshift(self):
        """
        x shift
        """
        return self._xshift

    def set_xshift(self, xshift):
        """
        setter of x shift
        """
        self._xshift = xshift

    @property
    def yshift(self):
        """
        x shift
        """
        return self._yshift

    def set_yshift(self, yshift):
        """
        setter of x shift
        """
        self._yshift = yshift

    def get_shear_value(self):
        """
        get shear value
        """
        plane = HexagonalPlane(lattice=self._hexagonal_lattice.lattice,
                               four=self._indices['K1'].four)
        d = plane.get_distance_from_plane(self._indices['eta2'].three)
        gamma = self._shear_strain_funcion(self._r)
        norm_eta1 = np.linalg.norm(
                plane.get_cartesian(self._indices['eta1'].three))
        s = gamma * d / norm_eta1
        return s

    def get_shear_matrix(self):
        """
        get shear matrix
        """
        s = self.get_shear_value()
        shear_matrix = np.eye(3)
        shear_matrix[1,2] = self._shear_strain_ratio * s
        return shear_matrix

    def get_shear_properties(self) -> dict:
        """
        get various properties related to shear

        Note:
            key variables are
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
        S = self.get_shear_matrix(self._shear_strain_ratio)
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

    def run(self):
        """
        build structure

        Note:
            the built structure is set to self.output_structure
        """
        shear_matrix = self.get_shear_matrix(self._shear_strain_ratio)
        supercell_matrix = self._parent_matrix * self._dim
        unit_lattice = PhonopyAtoms(symbols=['H'],
                cell=self._hexagonal_lattice.lattice,
                scaled_positions=np.array([[self._xshift,self._yshift,0]]))
        super_lattice = Supercell(unitcell=unit_lattice,
                                  supercell_matrix=supercell_matrix)
        lattice_points = get_lattice_points_from_supercell(
                lattice=self._hexagonal_lattice.lattice,
                dim=supercell_matrix)
        shear_lattice = \
            np.dot(self._hexagonal_lattice.lattice.T,
                   np.dot(supercell_matrix,
                          shear_matrix)).T
        atoms_from_lattice_points = np.dot(
                np.linalg.inv(supercell_matrix),
                self._atoms_from_lattice_points.T,
                ).T
        symbols = [self._symbol] * len(lattice_points) \
                                 * len(self._atoms_from_lattice_points)
        structure = {'lattice': shear_lattice,
                     'lattice_points': {
                         'white': lattice_points},
                     'atoms_from_lattice_points': {
                         'white': atoms_from_lattice_points},
                     'symbols': symbols}
        self.set_output_structure(structure)
        self._natoms = len(self._output_structure['symbols'])
