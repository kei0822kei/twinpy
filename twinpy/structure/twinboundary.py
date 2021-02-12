#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module deals with hexagonal twinboundary structure.
"""

from copy import deepcopy
import math
import numpy as np
from twinpy.structure.base import _BaseTwinStructure
from twinpy.structure.shear import ShearStructure


class TwinBoundaryStructure(_BaseTwinStructure):
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
            twintype: Twin type choose 1 or 2.

        Note:
            To see detail, visit _BaseTwinStructure class.
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
        self._layers = None
        self._delta = None

    def _set_rotation_matrix(self):
        """
        Set rotation matrix.
        """
        indices = self._indices.indices
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
            ValueError: Twintype is neither equal 1 nor 2.
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
            msg = "Twintype must be 1 or 2."
            raise ValueError(msg)
        self._dichromatic_operation = W

    @property
    def dichromatic_operation(self):
        """
        Dichromatic operation.
        """
        return self._dichromatic_operation

    @property
    def layers(self):
        """
        The number of layers.
        """
        return self._layers

    @property
    def delta(self):
        """
        Delta.
        """
        return self._delta

    def _get_parent_structure(self,
                              dim:np.array) -> dict:
        """
        Get parent.
        """
        parent = ShearStructure(lattice=self._hexagonal_lattice,
                                symbol=self._symbol,
                                shear_strain_ratio=0.,
                                twinmode=self._twinmode,
                                wyckoff=self._wyckoff)
        parent.run(dim=dim)
        output_structure = parent.output_structure
        return output_structure

    def _get_shear_twinboundary_lattice(self,
                                        tb_lattice:np.array,
                                        shear_strain_ratio:float) -> np.array:
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
                                 layers,
                                 delta,
                                 xshift:float=0.,
                                 yshift:float=0.):
        """
        Get twinboundary lattice.

        Args:
            layers: The number of layers in bulk.
            delta: Additional interval both sites of twin boundary.
            xshift: x shift.
            yshift: y shift.
        """
        def _add_delta(lat_cell:tuple,
                       delta:float):
            """
            Add delta interval to the twin boundary lattce.
            """
            ld = np.array([0., 0., delta])
            frame = lat_cell[0].copy()
            frame[2,:] = frame[2,:] + 4 * ld

            multi = {
                    'white_tb': 0,
                    'white': 1,
                    'black_tb': 2,
                    'black': 3,
                    }
            orig_cart_coords = np.dot(lat_cell[0].T, lat_cell[1].T).T.tolist()
            frac_coords = []
            for i in range(len(lat_cell[1])):
                cart_coord = orig_cart_coords[i] + multi[lat_cell[2][i]] * ld
                frac_coord = np.dot(cart_coord, np.linalg.inv(frame))
                frac_coords.append(frac_coord)

            return (frame, np.array(frac_coords), lat_cell[2])

        tot_layers = layers + 1
        prim_layers = self._indices.layers
        zdim = math.ceil(tot_layers / prim_layers)
        multi_dim = np.array([1,1,2*zdim])
        zratio = tot_layers / (zdim * prim_layers)
        p_orig_structure = self._get_parent_structure(dim=multi_dim)

        p_frame = np.dot(self._rotation_matrix,
                         p_orig_structure['lattice'].T).T
        t_frame = np.dot(self._dichromatic_operation,
                         p_frame.T).T
        tb_frame = np.array([
            p_frame[0],
            p_frame[1],
            (p_frame[2] - t_frame[2]) / 2
            ])
        crop_tb_frame = tb_frame * np.array([1., 1., zratio])

        p_cart_points = np.dot(p_frame.T,
                               p_orig_structure['lattice_points']['white'].T).T
        p_frac_points = np.round(np.dot(np.linalg.inv(crop_tb_frame.T),
                                        p_cart_points.T).T, decimals=8)
        t_cart_points = np.dot(self._dichromatic_operation,
                               p_cart_points.T).T
        t_frac_points = np.round(np.dot(np.linalg.inv(crop_tb_frame.T),
                                        t_cart_points.T).T, decimals=8)

        crop_p_tb_frac_points = np.array([ frac for frac in p_frac_points
                                           if np.allclose(frac[2], 0.) ]) % 1
        crop_p_frac_points = np.array([ frac for frac in p_frac_points
                                        if 0. < frac[2] < 0.5 ]) % 1
        crop_t_tb_frac_points = np.array([ frac for frac in t_frac_points
                                           if np.allclose(frac[2], -0.5) ]) % 1
        crop_t_frac_points = np.array([ frac for frac in t_frac_points
                                        if -0.5 < frac[2] < 0. ]) % 1

        p_shift = np.array([-xshift/2, -yshift/2, 0.])
        t_shift = np.array([ xshift/2,  yshift/2, 0.])
        scaled_positions = np.vstack([crop_p_tb_frac_points+p_shift,
                                      crop_p_frac_points+p_shift,
                                      crop_t_tb_frac_points+t_shift,
                                      crop_t_frac_points+t_shift])

        symbols = ['white_tb'] * len(crop_p_tb_frac_points) \
                + ['white'] * len(crop_p_frac_points) \
                + ['black_tb'] * len(crop_t_tb_frac_points) \
                + ['black'] * len(crop_t_frac_points)

        lat_cell = (crop_tb_frame, scaled_positions, symbols)
        delta_lat_cell = _add_delta(lat_cell, delta)

        return delta_lat_cell

    def run(self,
            layers,
            delta=0.,
            xshift:float=0.,
            yshift:float=0.,
            shear_strain_ratio:float=0.,
            ):
        """
        Build structure.

        Args:
            layers: The number of layers.
            delta: Additional interval both sites of twin boundary.
            xshift: x shift.
            yshift: y shift.
            shear_strain_ratio: Shear strain ratio.

        Note:
            The structure built is set self.output_structure.
        """
        tb_frame, lat_points, dichs = self.get_twinboudnary_lattice(
                layers=layers, delta=delta, xshift=xshift, yshift=yshift)

        W = self._dichromatic_operation
        R = self._rotation_matrix
        orig_cart_atoms = np.dot(self._hexagonal_lattice.T,
                                 self._atoms_from_lattice_points.T).T
        parent_cart_atoms = np.dot(R, orig_cart_atoms.T).T
        twin_cart_atoms = np.dot(W, parent_cart_atoms.T).T
        parent_frac_atoms = np.dot(np.linalg.inv(tb_frame.T),
                                   parent_cart_atoms.T).T
        twin_frac_atoms = np.dot(np.linalg.inv(tb_frame.T),
                                 twin_cart_atoms.T).T

        white_tb_ix = \
                [ i for i in range(len(dichs)) if dichs[i] == 'white_tb' ]
        white_ix = \
                [ i for i in range(len(dichs)) if dichs[i] == 'white' ]
        black_tb_ix = \
                [ i for i in range(len(dichs)) if dichs[i] == 'black_tb' ]
        black_ix = \
                [ i for i in range(len(dichs)) if dichs[i] == 'black' ]
        symbols = [ self._symbol ] * len(lat_points) * len(parent_frac_atoms)
        tb_shear_frame = self._get_shear_twinboundary_lattice(
                tb_lattice=tb_frame,
                shear_strain_ratio=shear_strain_ratio)

        lattice_points = {
                'white_tb': lat_points[white_tb_ix],
                'white': lat_points[white_ix],
                'black_tb': lat_points[black_tb_ix],
                'black': lat_points[black_ix]
                }

        atoms_from_lp = {
                'white_tb': parent_frac_atoms * np.array([1., 1., 0.]),
                'white': parent_frac_atoms,
                'black_tb': twin_frac_atoms * np.array([1., 1., 0.]),
                'black': twin_frac_atoms
                }

        output_structure = {
                'lattice': tb_shear_frame,
                'lattice_points': lattice_points,
                'atoms_from_lattice_points': atoms_from_lp,
                'symbols': symbols,
                }

        self._output_structure = output_structure
        self._natoms = len(self._output_structure['symbols'])
        self._xshift = xshift
        self._yshift = yshift
        self._shear_strain_ratio = shear_strain_ratio
        self._layers = layers
        self._delta = delta


def get_twinboundary(lattice:np.array,
                     symbol:str,
                     twinmode:str,
                     layers:int,
                     wyckoff:str='c',
                     delta:float=0.,
                     twintype:int=1,
                     xshift:float=0.,
                     yshift:float=0.,
                     shear_strain_ratio:float=0.,
                     ) -> TwinBoundaryStructure:
    """
    Get twinboudnary structure object.

    Args:
        lattice: Lattice matrix.
        symbol: Element symbol.
        twinmode: Twinmode.
        layers: The number of layers.
        wyckoff: No.194 Wycoff position ('c' or 'd').
        delta: Additional interval both sites of twin boundary.
        twintype: Twintype, choose from 1 and 2.
        xshift: x shift.
        yshift: y shift.
        shear_strain_ratio (float): Shear twinboundary ratio.
    """
    tb = TwinBoundaryStructure(lattice=lattice,
                               symbol=symbol,
                               twinmode=twinmode,
                               twintype=twintype,
                               wyckoff=wyckoff)
    tb.run(layers=layers,
           delta=delta,
           xshift=xshift,
           yshift=yshift,
           shear_strain_ratio=shear_strain_ratio)
    return tb
