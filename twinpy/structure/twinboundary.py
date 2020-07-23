#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HexagonalStructure
"""

import numpy as np
from copy import deepcopy
from twinpy.structure.base import _BaseStructure
from twinpy.structure.shear import ShearStructure


class TwinBoundaryStructure(_BaseStructure):
    """
    Twinboundary structure class.
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
        Args:
            twintype (int): twin type choose 1 or 2

        Note:
            to see detail, visit _BaseStructure class
        """
        super().__init__(lattice=lattice,
                         symbol=symbol,
                         twinmode=twinmode,
                         wyckoff=wyckoff)
        self._shear_strain_ratio = None
        self._twintype = twintype
        self._rotation_matrix = None
        self._set_rotation_matrix()
        self._dichromatic_operation = None
        self._set_dichromatic_operation()

    def _set_rotation_matrix(self):
        """
        Set rotation matrix
        """
        indices = self._indices.get_indices()
        rotation_matrix = np.array([
                indices['m'].get_cartesian(normalize=True),
                indices['eta1'].get_cartesian(normalize=True),
                indices['k1'].get_cartesian(normalize=True),
                ])
        self._rotation_matrix = rotation_matrix

    @property
    def rotation_matrix(self):
        """
        Rotation matrix.
        """
        return self._rotation_matrix

    @property
    def shear_strain_ratio(self):
        """
        Shear twinboundary ratio.
        """
        return self._shear_strain_ratio

    @property
    def twintype(self):
        """
        Twin type.
        """
        return self._twintype

    def _set_dichromatic_operation(self):
        """
        Set dichromatic operation.

        Raises:
            ValueError: twintype != 1 nor 2
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
            raise ValueError(msg)
        self._dichromatic_operation = W

    @property
    def dichromatic_operation(self):
        """
        Dichromatic operation.
        """
        return self._dichromatic_operation

    def _get_parent_structure(self,
                              dim:np.array):
        """
        Get parent.
        """
        parent = ShearStructure(lattice=self._hcp_lattice.lattice,
                                symbol=self._symbol,
                                shear_strain_ratio=0.,
                                twinmode=self._twinmode,
                                wyckoff=self._wyckoff)
        parent.run(dim=dim)
        output_structure = parent.output_structure
        return output_structure

    def _get_shear_twobundary_lattice(self,
                                      tb_lattice:np.array,
                                      shear_strain_ratio:float):
        """
        Get shear twinboudnary lattice.
        """
        lat = deepcopy(tb_lattice)
        e_b = lat[1] / np.linalg.norm(lat[1])
        shear_func = self._indices.get_shear_strain_function()
        lat[2] += np.linalg.norm(lat[2]) \
                  * shear_func(self._r) \
                  * shear_strain_ratio \
                  * e_b
        return lat

    def get_twinboudnary_lattice(self,
                                 dim:np.array=np.ones(3, dtype='intc'),
                                 xshift:float=0.,
                                 yshift:float=0.):
        """
        Get twinboundary lattice.

        Args:
            dim (np.array): dimension
            xshift (float): x shift
            yshift (float): y shift
        """
        multi_dim = np.array(dim) * np.array([1,1,2])
        p_orig_structure = self._get_parent_structure(dim=multi_dim)
        p_frame = np.dot(self._rotation_matrix,
                         p_orig_structure['lattice'].T).T
        p_cart_points = np.dot(p_frame.T,
                               p_orig_structure['lattice_points']['white'].T).T
        t_frame = np.dot(self._dichromatic_operation,
                         p_frame.T).T
        t_cart_points = np.dot(self._dichromatic_operation,
                               p_cart_points.T).T
        tb_frame = np.array([
            p_frame[0],
            p_frame[1],
            (p_frame[2] - t_frame[2]) / 2
            ])

        p_frac_points = np.dot(np.linalg.inv(tb_frame.T),
                               p_cart_points.T).T
        p_frac_points = np.round(p_frac_points, decimals=8) % 1.
        t_frac_points = np.dot(np.linalg.inv(tb_frame.T),
                               t_cart_points.T).T
        t_frac_points = np.round(t_frac_points, decimals=8) % 1.

        p_ix = np.where(p_frac_points[:,2] < 0.5)[0]
        t_ix = np.where(t_frac_points[:,2] >= 0.5)[0]
        p_shift = np.array([-xshift/2/dim[0], -yshift/2/dim[1], 0.])
        t_shift = np.array([ xshift/2/dim[0],  yshift/2/dim[1], 0.])
        scaled_positions = np.vstack([p_frac_points[p_ix]+p_shift,
                                      t_frac_points[t_ix]+t_shift])

        num = len(p_ix)
        symbols = ['white'] * num + ['black'] * num

        return (tb_frame, scaled_positions, symbols)

    def _make_twinboundary_flat(self,
                                output_structure:dict):
        """
        Make twinboundary flat.
        """
        white_lp = output_structure['lattice_points']['white']
        black_lp = output_structure['lattice_points']['black']
        white_ix = np.where(white_lp[:,2] != 0.)[0]
        white_tb_ix = np.where(white_lp[:,2] == 0.)[0]
        black_ix = np.where(black_lp[:,2] != 0.5)[0]
        black_tb_ix = np.where(black_lp[:,2] == 0.5)[0]
        lattice_points = {'white': white_lp[white_ix],
                          'white_tb': white_lp[white_tb_ix],
                          'black': black_lp[black_ix],
                          'black_tb': black_lp[black_tb_ix]}
        atoms_from_lp = output_structure['atoms_from_lattice_points']
        atoms_from_lp['white_tb'] = \
                atoms_from_lp['white'] * np.array([1.,1.,0. ])
        atoms_from_lp['black_tb'] = \
                atoms_from_lp['black'] * np.array([1.,1.,0. ])
        flat_structure = {
                'lattice': output_structure['lattice'],
                'lattice_points': lattice_points,
                'atoms_from_lattice_points': atoms_from_lp,
                'symbols': output_structure['symbols'],
                }
        return flat_structure

    def run(self,
            dim:np.array=np.ones(3, dtype='intc'),
            xshift:float=0.,
            yshift:float=0.,
            shear_strain_ratio:float=0.,
            make_tb_flat:bool=False,
            ):
        """
        Build structure.

        Args:
            dim (np.array): dimension
            xshift (float): x shift
            yshift (float): y shift
            shear_strain_ratio (float): shear strain ratio
            make_tb_flat (bool): whether make twin boundary flat

        Note:
            the structure built is set self.output_structure
        """
        W = self._dichromatic_operation
        R = self._rotation_matrix
        orig_cart_atoms = np.dot(self._hexagonal_lattice.lattice.T,
                                 self._atoms_from_lattice_points.T).T
        parent_cart_atoms = np.dot(R, orig_cart_atoms.T).T
        twin_cart_atoms = np.dot(W, parent_cart_atoms.T).T
        tb_frame, lat_points, dichs = self.get_twinboudnary_lattice(
                dim=dim, xshift=xshift, yshift=yshift)
        parent_frac_atoms = np.dot(np.linalg.inv(tb_frame.T),
                                   parent_cart_atoms.T).T
        twin_frac_atoms = np.dot(np.linalg.inv(tb_frame.T),
                                 twin_cart_atoms.T).T

        white_ix = [ i for i in range(len(dichs)) if dichs[i] == 'white' ]
        black_ix = [ i for i in range(len(dichs)) if dichs[i] == 'black' ]
        symbols = [ self._symbol ] * len(lat_points) * len(parent_frac_atoms)
        tb_shear_frame = self._get_shear_twinboundary_lattice(
                tb_lattice=tb_frame,
                shear_strain_ratio=shear_strain_ratio)
        output_structure = \
                {'lattice': tb_shear_frame,
                 'lattice_points': {
                     'white': lat_points[white_ix],
                     'black': lat_points[black_ix],
                                   },
                 'atoms_from_lattice_points': {
                      'white': parent_frac_atoms,
                      'black': twin_frac_atoms,
                                              },
                 'symbols': symbols}

        if make_tb_flat:
            output_structure = self._make_twinboundary_flat(output_structure)

        self._output_structure = output_structure
        self._natoms = len(self._output_structure['symbols'])
        self._dim = dim
        self._xshift = xshift
        self._yshift = yshift
        self._shear_strain_ratio = shear_strain_ratio


def get_twinboundary(lattice:np.array,
                     symbol:str,
                     twinmode:str,
                     wyckoff:str='c',
                     twintype:int=1,
                     xshift:float=0.,
                     yshift:float=0.,
                     dim:np.array=np.ones(3, dtype='intc'),
                     shear_strain_ratio:float=0.,
                     make_tb_flat:bool=False,
                     ):
    """
    Get twinboudnary structure object.

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
                               twinmode=twinmode,
                               twintype=twintype,
                               wyckoff=wyckoff)
    tb.run(dim=dim,
           xshift=xshift,
           yshift=yshift,
           shear_strain_ratio=shear_strain_ratio,
           make_tb_flat=make_tb_flat)
    return tb
