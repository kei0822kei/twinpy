#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hexagonal shear structure
"""

import numpy as np
from scipy.linalg import sqrtm
from twinpy.lattice.hexagonal_plane import HexagonalPlane
from twinpy.structure.base import (get_lattice_points_from_supercell,
                                   _BaseStructure)


class ShearStructure(_BaseStructure):
    """
    Shear structure class.
    """

    def __init__(
           self,
           lattice:np.array,
           symbol:str,
           shear_strain_ratio:float,
           twinmode:str,
           wyckoff:str='c',
           ):
        """
        Args:
            shear_strain_ratio (float): shear strain ratio

        Note:
            to see detail, visit _BaseStructure class
        """
        super().__init__(lattice=lattice,
                         symbol=symbol,
                         twinmode=twinmode,
                         wyckoff=wyckoff)
        self._dim = None
        self._shear_strain_ratio = shear_strain_ratio

    @property
    def dim(self):
        """
        Dimension.
        """
        return self._dim

    @property
    def shear_strain_ratio(self):
        """
        Shear strain ratio.
        """
        return self._shear_strain_ratio

    @property
    def output_structure_primitive(self):
        """
        Built structure primitive basis
        not standardized.
        """
        return self._output_structure

    def get_gamma(self):
        """
        Get gamma.

        Returns:
            float: gamma used for computing shear value
                   more detail, see documentaion
        """
        shear_strain_function = self._indices.get_shear_strain_function()
        gamma = shear_strain_function(self._r)
        return gamma

    def get_shear_value(self):
        """
        Get shear value.
        """
        ratio = self._shear_strain_ratio
        plane = HexagonalPlane(lattice=self._hexagonal_lattice.lattice,
                               four=self._indices.indices['K1'].four)
        d = plane.get_distance_from_plane(self._indices.indices['eta2'].three)
        gamma = self.get_gamma()
        norm_eta1 = np.linalg.norm(
                plane.get_cartesian(self._indices.indices['eta1'].three))
        s = ratio * gamma * d / norm_eta1
        return s

    def get_shear_matrix(self):
        """
        Get shear matrix.
        """
        s = self.get_shear_value()
        shear_matrix = np.eye(3)
        shear_matrix[1,2] = s
        return shear_matrix

    def get_shear_properties(self) -> dict:
        """
        Get various properties related to shear.

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

        Todo:
            FUTURE EDITED
        """
        H = self.hexagonal_lattice.lattice.T
        M = self._parent_matrix
        S = self.get_shear_matrix()
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

    def get_shear_lattice(self,
                          is_primitive:bool=False,
                          dim:np.array=np.ones(3, dtype='intc'),
                          xshift:float=0.,
                          yshift:float=0.) -> tuple:
        """
        Get shear lattice.

        Args:
            is_primitive (bool): If primitive, multiplied M^(-1)
            dim (np.array): dimension
            xshift (float): x shift
            yshift (float): y shift

        Returns:
            tuple: shear lattice
        """
        shear_matrix = self.get_shear_matrix()
        parent_matrix = self._indices.get_supercell_matrix_for_parent()
        supercell_matrix = parent_matrix * dim
        shear_lattice = \
            np.dot(self._hexagonal_lattice.lattice.T,
                   np.dot(supercell_matrix,
                          shear_matrix)).T
        lattice_points = get_lattice_points_from_supercell(
                lattice=self._hexagonal_lattice.lattice,
                dim=supercell_matrix)
        lattice_points += np.array([xshift, yshift, 0]) / np.array(dim)

        if is_primitive:
            shear_lattice = np.dot(shear_lattice.T,
                                   np.linalg.inv(supercell_matrix)).T
            lattice_points = np.array([xshift, yshift, 0.])

        lattice_points = np.round(lattice_points, decimals=8) % 1
        symbols = ['white'] * len(lattice_points)
        return (shear_lattice, lattice_points, symbols)

    def run(self,
            is_primitive:bool=False,
            dim:np.array=np.ones(3, dtype='intc'),
            xshift:float=0.,
            yshift:float=0.):
        """
        Build structure.

        Args:
            is_primitive (bool): If primitive, multiplied M^(-1)
            dim (np.array): dimension
            xshift (float): x shift
            yshift (float): y shift

        Note:
            the built structure is set to self.output_structure
        """
        if type(dim) is list:
            dim = np.array(dim, dtype='intc')

        shear_lattice, lattice_points, _ = \
                self.get_shear_lattice(is_primitive=is_primitive,
                                       dim=dim,
                                       xshift=xshift,
                                       yshift=yshift)

        if not is_primitive:
            parent_matrix = self._indices.get_supercell_matrix_for_parent()
            supercell_matrix = parent_matrix * dim
            atoms_from_lp = np.dot(np.linalg.inv(supercell_matrix),
                                   self._atoms_from_lattice_points.T).T
        else:
            atoms_from_lp = self._atoms_from_lattice_points

        symbols = [self._symbol] * len(lattice_points) \
                                 * len(self._atoms_from_lattice_points)
        structure = {'lattice': shear_lattice,
                     'lattice_points': {
                         'white': lattice_points},
                     'atoms_from_lattice_points': {
                         'white': atoms_from_lp},
                     'symbols': symbols}

        self._output_structure = structure
        self._dim = dim
        self._xshift = xshift
        self._yshift = yshift


def get_shear(lattice:np.array,
              symbol:str,
              twinmode:str,
              wyckoff:str='c',
              xshift:float=0.,
              yshift:float=0.,
              dim:np.array=np.ones(3, dtype='intc'),
              shear_strain_ratio:float=0.,
              is_primitive:bool=False,
              ) -> ShearStructure:
    """
    Get shear structure object.

    Args:
        lattice (np.array): lattice
        symbol (str): element symbol
        wyckoff (str): No.194 Wycoff position ('c' or 'd')
        xshift (float): x shift
        yshift (float): y shift
        dim (np.array): dimension
        shear_strain_ratio (float): shear strain ratio
        is_primitive (bool): If primitive, multiplied M^(-1)
    """
    shear = ShearStructure(lattice=lattice,
                           symbol=symbol,
                           twinmode=twinmode,
                           shear_strain_ratio=shear_strain_ratio,
                           wyckoff=wyckoff)
    shear.run(dim=dim,
              xshift=xshift,
              yshift=yshift,
              is_primitive=is_primitive)
    return shear
