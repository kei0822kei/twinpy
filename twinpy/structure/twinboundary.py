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
            make_tb_flat:bool=True,
            ):
        """
        Build structure.

        Args:
            layers: The number of layers.
            delta: Additional interval both sites of twin boundary.
            xshift: x shift.
            yshift: y shift.
            shear_strain_ratio: Shear strain ratio.
            make_tb_flat: If True, atoms on the twin boundary plane are
                          projected to twin boundary.

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

        white_ix = \
                [ i for i in range(len(dichs)) if dichs[i] == 'white' ]
        white_tb_ix = \
                [ i for i in range(len(dichs)) if dichs[i] == 'white_tb' ]
        black_ix = \
                [ i for i in range(len(dichs)) if dichs[i] == 'black' ]
        black_tb_ix = \
                [ i for i in range(len(dichs)) if dichs[i] == 'black_tb' ]
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
                'white': parent_frac_atoms,
                'white_tb': parent_frac_atoms.copy(),
                'black': twin_frac_atoms,
                'black_tb': twin_frac_atoms.copy(),
                }

        if make_tb_flat:
            atoms_from_lp['white_tb'] *= np.array([1., 1., 0.])
            atoms_from_lp['black_tb'] *= np.array([1., 1., 0.])

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
                     expansion_ratios:np.array=np.ones(3),
                     make_tb_flat:bool=True,
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
        expansion_ratios: Expansion ratios.
        make_tb_flat: If True, atoms on the twin boundary plane are
                      projected to twin boundary.
    """
    tb = TwinBoundaryStructure(lattice=lattice,
                               symbol=symbol,
                               twinmode=twinmode,
                               twintype=twintype,
                               wyckoff=wyckoff)
    tb.set_expansion_ratios(expansion_ratios)
    tb.run(layers=layers,
           delta=delta,
           xshift=xshift,
           yshift=yshift,
           shear_strain_ratio=shear_strain_ratio,
           make_tb_flat=make_tb_flat)
    return tb


def plot_nearest_atomic_distance_of_twinboundary(
        lattice:np.array,
        symbol:str,
        twinmode:str,
        layers:int,
        wyckoff:str='c',
        delta:float=0.,
        twintype:int=1,
        xshift:float=0.,
        yshift:float=0.,
        shear_strain_ratio:float=0.,
        expansion_ratios:np.array=np.ones(3),
        make_tb_flat:bool=True,
        ):
    """
    Show nearest atomic distance of twinboundary
    by changing xshift and yshift.

    Args:
        lattice: Lattice matrix.
        symbol: Element symbol.
        twinmode: Twinmode.
        layers: The number of layers.
        wyckoff: No.194 Wycoff position ('c' or 'd').
        delta: Additional interval both sites of twin boundary.
        twintype: Twintype, choose from 1 and 2.
        shear_strain_ratio (float): Shear twinboundary ratio.
        expansion_ratios: Expansion ratios.
        make_tb_flat: If True, atoms on the twin boundary plane are
                      projected to twin boundary.
    """
    import matplotlib.pyplot as plt
    from twinpy.structure.bonding import get_nearest_atomic_distance
    from twinpy.structure.lattice import CrystalLattice

    tb = TwinBoundaryStructure(lattice=lattice,
                               symbol=symbol,
                               twinmode=twinmode,
                               twintype=twintype,
                               wyckoff=wyckoff)
    tb.set_expansion_ratios(expansion_ratios)

    x = np.arange(0, 1.025, 0.025)
    y = np.arange(0, 1.025, 0.025)

    X, Y = np.meshgrid(x, y)
    shape = X.shape
    print(shape)
    Z = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            tb.run(layers=layers,
                   delta=delta,
                   xshift=X[i,j],
                   yshift=Y[i,j],
                   shear_strain_ratio=shear_strain_ratio,
                   make_tb_flat=make_tb_flat)
            cell = tb.get_cell_for_export()
            Z[i,j] = get_nearest_atomic_distance(cell)

    a, b, _ = CrystalLattice(lattice=cell[0]).abc
    fig = plt.figure(figsize=(4,4*b/a))
    cont=plt.contour(X,Y,Z,  5, vmin=0,vmax=4, colors=['black'])
    cont.clabel(fmt='%1.1f')

    plt.xlabel('xshift')
    plt.ylabel('yshift')

    plt.pcolormesh(X,Y,Z, cmap='cool')
    pp=plt.colorbar (orientation="vertical")
    pp.set_label("Nearest Atomic Distance")

    plt.show()
