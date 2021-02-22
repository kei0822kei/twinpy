#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module deals with hexagonal shear structure.
"""

import numpy as np
from twinpy.properties.hexagonal import HexagonalPlane
from twinpy.structure.lattice import get_lattice_points_from_supercell
from twinpy.structure.base import _BaseTwinStructure


class ShearStructure(_BaseTwinStructure):
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
        Setup.

        Args:
            lattice: Lattice.
            symbol: Element symbol.
            twinmode: Twin mode.
            wyckoff: No.194 Wycoff letter ('c' or 'd').
            shear_strain_ratio: Shear strain ratio.

        Note:
            To see detail, visit _BaseStructure class.
        """
        super().__init__(lattice=lattice,
                         symbol=symbol,
                         twinmode=twinmode,
                         wyckoff=wyckoff)
        self._dim = None
        self._shear_strain_ratio = shear_strain_ratio
        self._output_structure = None

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

    def get_gamma(self) -> float:
        """
        Get gamma.

        Returns:
            float: Gamma used for computing shear value
                   more detail, see documentaion.
        """
        shear_strain_function = self._indices.get_shear_strain_function()
        gamma = shear_strain_function(self._r)
        return gamma

    def get_shear_value(self):
        """
        Get shear value.
        """
        ratio = self._shear_strain_ratio
        plane = HexagonalPlane(lattice=self._hexagonal_lattice,
                               four=self._indices.indices['K1'].four)
        d = plane.get_plane_interval()
        gamma = self.get_gamma()
        norm_eta1 = \
                np.linalg.norm(self._indices.indices['eta1'].get_cartesian())
        z_norm = d * self._indices.layers
        s = ratio * gamma * z_norm / norm_eta1
        return s

    def get_shear_matrix(self):
        """
        Get shear matrix.
        """
        s = self.get_shear_value()
        shear_matrix = np.eye(3)
        shear_matrix[1,2] = s
        return shear_matrix

    # def get_shear_properties(self) -> dict:
    #     """
    #     Get various properties related to shear.

    #     Note:
    #         key variables are
    #         - shear value (s)
    #         - shear ratio (alpha)
    #         - strain matrix (S)
    #         - deformation gradient tensor (F)
    #         - right Cauchy-Green tensor (C)
    #         - left Cauchy-Green tensor (b)
    #         - matrial stretch tensor (U)
    #         - spatial stretch tensor (V)
    #         - rotation (R)
    #         for more detail and definition, see documentation

    #     Todo:
    #         FUTURE EDITED
    #     """
    #     from scipy.linalg import sqrtm
    #     H = self.hexagonal_lattice.lattice.T
    #     M = self._parent_matrix
    #     S = self.get_shear_matrix()
    #     F = np.eye(3) + \
    #         np.dot(H,
    #                np.dot(M,
    #                       np.dot(S,
    #                              np.dot(np.linalg.inv(M),
    #                                     np.linalg.inv(H)))))
    #     s = self._get_shear_value()
    #     alpha = self._shear_strain_ratio
    #     C = np.dot(F.T, F)
    #     b = np.dot(F, F.T)
    #     U = sqrtm(C)
    #     V = sqrtm(b)
    #     R = np.dot(F, np.linalg.inv(U))
    #     R_ = np.dot(np.linalg.inv(V), F)
    #     np.testing.assert_allclose(R, R_)
    #     return {
    #              'shear_value': s,
    #              'shear_ratio': alpha,
    #              'strain_matrix': S,
    #              'deformation_gradient_tensor': F,
    #              'material_stretch_tensor': U,
    #              'spatial_stretch_tensor': V,
    #              'right_Cauchy': C,
    #              'left_Cauchy': b,
    #              'rotation':R,
    #            }

    def get_shear_lattice(self,
                          is_primitive:bool=False,
                          dim:np.array=np.ones(3, dtype='intc'),
                          xshift:float=0.,
                          yshift:float=0.) -> tuple:
        """
        Get shear lattice.

        Args:
            is_primitive: If primitive, multiplied M^(-1).
            dim: Dimension.
            xshift: x shift.
            yshift: y shift.

        Returns:
            tuple: Shear lattice.
        """
        shear_matrix = self.get_shear_matrix()
        parent_matrix = self._indices.get_supercell_matrix_for_parent()
        supercell_matrix = parent_matrix * dim
        shear_lattice = \
            np.dot(self._hexagonal_lattice.T,
                   np.dot(supercell_matrix,
                          shear_matrix)).T
        lattice_points = get_lattice_points_from_supercell(
                lattice=self._hexagonal_lattice,
                dim=supercell_matrix)
        lattice_points += np.array([xshift, yshift, 0]) / np.array(dim)

        if is_primitive:
            shear_lattice = np.dot(shear_lattice.T,
                                   np.linalg.inv(supercell_matrix)).T
            lattice_points = np.array([[xshift, yshift, 0.]])
        else:
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
            is_primitive: If primitive, multiplied M^(-1).
            dim: Dimension.
            xshift: x shift
            yshift: y shift

        Note:
            The built structure is set to self.output_structure.
        """
        if isinstance(dim, list):
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
              expansion_ratios:np.array=np.ones(3),
              is_primitive:bool=False,
              ) -> ShearStructure:
    """
    Get shear structure object.

    Args:
        lattice: Lattice.
        symbol: Element symbol.
        twinmode: Twin mode.
        wyckoff: No.194 Wycoff letter ('c' or 'd').
        xshift: Structure x shift.
        yshift: Structure y shift.
        dim: Supercell dimension.
        shear_strain_ratio: Shear strain ratio.
        expansion_ratios: Expansion ratios.
        is_primitive: If primitive, by multiplying M^(-1)
                      toward conventional structure.
    """
    shear = ShearStructure(lattice=lattice,
                           symbol=symbol,
                           twinmode=twinmode,
                           shear_strain_ratio=shear_strain_ratio,
                           wyckoff=wyckoff)
    shear.set_expansion_ratios(expansion_ratios)
    shear.run(dim=dim,
              xshift=xshift,
              yshift=yshift,
              is_primitive=is_primitive)
    return shear
