#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HexagonalStructure
"""

import numpy as np
from scipy.linalg import sqrtm
from copy import deepcopy
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
from twinpy.structure.base import _BaseStructure
from twinpy.structure.shear import ShearStructure
from twinpy.file_io import write_poscar

def get_twinboundary(lattice:np.array,
                     symbol:str,
                     twinmode:str,
                     wyckoff:str='c',
                     twintype:int=1,
                     xshift:float=0.,
                     yshift:float=0.,
                     dim:np.array=np.ones(3, dtype='intc'),
                     shear_strain_ratio:float=0.,
                     make_tb_flat=True,
                     ):
    """
    set shear structure object

    Args:
        lattice (np.array): lattice
        symbol (str): element symbol
        twinmode (str): twinmode
        wyckoff (str): No.194 Wycoff position ('c' or 'd')
        twintype (int): twintype, choose from 1 and 2
        xshift (float): x shift
        yshift (float): y shift
        dim (np.array): dimension
        shear_strain_ratio (float): shear twinboundary ratio
        make_tb_flat (bool): whether make twin boundary flat
    """
    tb = TwinBoundaryStructure(lattice=lattice,
                               symbol=symbol,
                               wyckoff=wyckoff)
    tb.set_parent(twinmode)
    tb.run(dim=dim,
           twintype=twintype,
           xshift=xshift,
           yshift=yshift,
           shear_strain_ratio=shear_strain_ratio,
           make_tb_flat=make_tb_flat,
           )
    return tb

class TwinBoundaryStructure(_BaseStructure):
    """
    twinboundary structure class
    """

    def __init__(
           self,
           lattice:np.array,
           symbol:str,
           twinmode:str,
           twintype:int,
           wyckoff:str='c',
        ):
        """
        initialize

        Note:
            to see detail, visit _BaseStructure class
        """
        super().__init__(lattice=lattice,
                         symbol=symbol,
                         twinmode=twinmode,
                         wyckoff=wyckoff)
        self._dim = np.ones(3, dtype=int)
        self._shear_strain_ratio = None
        self._twintype = twintype
        self._rotation_matrix = None
        self._set_rotation_matrix()

    def _set_rotation_matrix(self):
        indices = self._indices.get_indices()
        rotaion_matrix = np.array([
                indices['m'].get_cartesian(normalize=True),
                indices['eta1'].get_cartesian(normalize=True),
                indices['k1'].get_cartesian(normalize=True),
                ]).T
        self._rotation_matrix = rotaion_matrix

    @property
    def rotation_matrix(self):
        """
        rotation matrix
        """
        return self._rotation_matrix

    @property
    def shear_strain_ratio(self):
        """
        shear twinboundary ratio
        """
        return self._shear_strain_ratio

    @property
    def twintype(self):
        """
        twin type
        """
        return self._twintype

    def get_dichromatic_operation(self):
        """
        get dichromatic operation
        """
        twintype = self._twintype
        if twintype == 1:
            W = np.array([[ 1, 0, 0],
                          [ 0, 1, 0],
                          [ 0, 0,-1]])
        elif twintype == 2:
            W = np.array([[-1, 0, 0],
                          [ 0, 1, 0],
                          [ 0, 0,-1]])
        else:
            msg = "twintype must be 1 or 2"
            raise RuntimeError(msg)
        return W

    def _get_parent(self, zdim):
        """
        get parent
        """
        parent = ShearStructure(lattice=self._hcp_lattice.lattice,
                                symbol=self._symbol,
                                ratio=0.,
                                twinmode=self._twinmode,
                                wyckoff=self._wyckoff)
        parent.run(dim=(1,1,zdim))
        return parent

    def _get_parent_lattice_points_atoms(self, parent_output):
        """
        get parent structure for twinboudnary
        """
        M = parent_output['lattice'].T
        lattice_points = parent_output['lattice_points']['white']
        additional_points = \
                lattice_points[np.isclose(lattice_points[:,2], 0)] + \
                    np.array([0.,0.,1.])
        lattice_points = np.vstack((lattice_points, additional_points))
        lp_p_cart = np.dot(M, lattice_points.T).T
        atoms_p_cart = np.dot(
                M, parent_output['atoms_from_lattice_points']['white'].T).T
        return (lp_p_cart, atoms_p_cart)

    def _get_twin_lattice_points_atoms(self,
                                       dichromatic_operation,
                                       rotation,
                                       lp_p_cart,
                                       atoms_p_cart):
        """
        get twin structure for twinboudnary
        """
        W = dichromatic_operation
        R = rotation
        lp_t_cart = np.dot(R,
                          np.dot(W,
                                 np.dot(np.linalg.inv(R),
                                        lp_p_cart.T))).T
        atoms_t_cart = np.dot(R,
                              np.dot(W,
                                     np.dot(np.linalg.inv(R),
                                            atoms_p_cart.T))).T
        return (lp_t_cart, atoms_t_cart)

    def _get_shear_tb_lattice(self,
                              tb_lattice,
                              shear_strain_ratio):
        """
        return shear twinboudnary lattice
        """
        lat = deepcopy(tb_lattice)
        e_b = lat[1] / np.linalg.norm(lat[1])
        shear_func = self._indices.get_shear_strain_function()
        lat[2] += np.linalg.norm(lat[2]) * shear_func(self._r) * shear_strain_ratio * e_b
        return lat

    def _get_twinboundary_structure(self,
                                    tb_lattice,
                                    lp_p_cart,
                                    lp_t_cart,
                                    atoms_p_cart,
                                    atoms_t_cart,
                                    make_tb_flat,
                                    dim,
                                    xshift,
                                    yshift,
                                    shear_strain_ratio):
        """
        return twinbounadry structure
        """
        white_lp = np.dot(np.linalg.inv(tb_lattice.T), lp_p_cart.T).T % 1
        black_lp = np.dot(np.linalg.inv(tb_lattice.T), lp_t_cart.T).T % 1
        shear_tb_lat = self._get_shear_tb_lattice(tb_lattice,
                                                  shear_strain_ratio)

        if make_tb_flat:
            black_tb_ix  = [ bl2 for bl2 in np.isclose(white_lp[:,2], 0.5) ]
            white_tb_ix = [ bl1 or bl3 for bl1, bl3 in \
                            zip(np.isclose(black_lp[:,2], 0),
                                np.isclose(black_lp[:,2], 1)) ]

            white_tb_lp = white_lp[white_tb_ix]
            black_tb_lp = black_lp[black_tb_ix]
            w_ix  = [ bl1 or bl2 or bl3 for bl1, bl2, bl3 in \
                             zip(np.isclose(white_lp[:,2], 0),
                                 np.isclose(white_lp[:,2], 0.5),
                                 np.isclose(white_lp[:,2], 1)) ]
            b_ix  = [ bl1 or bl2 or bl3 for bl1, bl2, bl3 in \
                             zip(np.isclose(black_lp[:,2], 0),
                                 np.isclose(black_lp[:,2], 0.5),
                                 np.isclose(black_lp[:,2], 1)) ]
            white_lp = white_lp[[ not bl for bl in w_ix ]]
            black_lp = black_lp[[ not bl for bl in b_ix ]]

            # atoms from lattice points
            white_atoms = np.dot(np.linalg.inv(tb_lattice.T),
                                 atoms_p_cart.T).T
            black_atoms = np.dot(np.linalg.inv(tb_lattice.T),
                                 atoms_t_cart.T).T
            atms = len(white_atoms)
            white_tb_atoms = np.hstack((white_atoms[:,:2],
                                        np.zeros(atms).reshape(atms,1)))
            black_tb_atoms = np.hstack((black_atoms[:,:2],
                                        np.zeros(atms).reshape(atms,1)))
            nlp = len(white_lp)+len(black_lp)+len(white_tb_lp)+len(black_tb_lp)
            symbols = [self._symbol] * nlp \
                                     * len(self._atoms_from_lattice_points)
            return {'lattice': shear_tb_lat,
                    'lattice_points': {
                         'white': white_lp + np.array([xshift/(dim[0]*2),
                                                       yshift/(dim[1]*2),
                                                       0.]),
                         'white_tb': white_tb_lp + np.array([xshift/(dim[0]*2),
                                                             yshift/(dim[1]*2),
                                                             0.]),
                         'black': black_lp + np.array([-xshift/(dim[0]*2),
                                                       -yshift/(dim[1]*2),
                                                       0.]),
                         'black_tb': black_tb_lp + np.array([-xshift/(dim[0]*2),
                                                             -yshift/(dim[1]*2),
                                                             0.]),
                                      },
                    'atoms_from_lattice_points': {
                         'white': white_atoms,
                         'white_tb': white_tb_atoms,
                         'black': black_atoms,
                         'black_tb': black_tb_atoms,
                                                 },
                    'symbols': symbols}

        else:
            grey_ix  = [ bl2 for bl2 in np.isclose(white_lp[:,2], 0.5) ]
            grey_ix_ = [ bl1 or bl3 for bl1, bl3 in \
                             zip(np.isclose(black_lp[:,2], 0),
                                 np.isclose(black_lp[:,2], 1)) ]

            white_lp = white_lp[[ not bl for bl in grey_ix ]]
            black_lp = black_lp[[ not bl for bl in grey_ix_ ]]
            white_atoms = np.dot(np.linalg.inv(tb_lattice.T), atoms_p_cart.T).T
            black_atoms = np.dot(np.linalg.inv(tb_lattice.T), atoms_t_cart.T).T
            symbols = [self._symbol] * (len(white_lp)+len(black_lp)) \
                                     * len(self._atoms_from_lattice_points)
            return {'lattice': shear_tb_lat,
                    'lattice_points': {
                         'white': white_lp + np.array([xshift/(dim[0]*2),
                                                       self._yshift/(dim[1]*2),
                                                       0.]),
                         'black': black_lp + np.array([-xshift/(dim[0]*2),
                                                       -self._yshift/(dim[1]*2),
                                                       0.]),
                                      },
                    'atoms_from_lattice_points': {
                         'white': white_atoms,
                         'black': black_atoms,
                                                 },
                    'symbols': symbols}

    def run(self,
            dim=np.ones(3, dtype='intc'),
            xshift=0.,
            yshift=0.,
            shear_strain_ratio=0.,
            make_tb_flat=True):
        """
        build structure

        Note:
            the structure built is set self.output_structure
        """
        parent = self._get_parent(zdim=dim[2])
        lp_p_cart, atoms_p_cart = self._get_parent_lattice_points_atoms(
                        parent_output=parent.get_output_stucture())
        W = self.get_dichromatic_operation()
        R = self._rotation_matrix
        lp_t_cart, atoms_t_cart = \
                self._get_twin_lattice_points_atoms(
                        dichromatic_operation=W,
                        rotation=R,
                        lp_p_cart=lp_p_cart,
                        atoms_p_cart=atoms_p_cart,
                        )
        tb_c = lp_p_cart[-1] - lp_t_cart[-1]
        tb_lattice = np.array([parent.output_structure['lattice'][0],
                               parent.output_structure['lattice'][1],
                               tb_c])
        self._output_structure = \
                self._get_twinboundary_structure(
                        tb_lattice=tb_lattice,
                        lp_p_cart=lp_p_cart,
                        lp_t_cart=lp_t_cart,
                        atoms_p_cart=atoms_p_cart,
                        atoms_t_cart=atoms_t_cart,
                        make_tb_flat=make_tb_flat,
                        dim=dim,
                        xshift=xshift,
                        yshift=yshift,
                        shear_strain_ratio=shear_strain_ratio)
        self._natoms = len(self.output_structure['symbols'])
        self._dim = dim
        self._twintype = twintype
        self._xshift = xshift
        self._yshift = yshift
        self._shear_strain_ratio = shear_strain_ratio
