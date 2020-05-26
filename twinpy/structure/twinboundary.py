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
from twinpy.structure.base import _BaseStructure
from twinpy.structure.shear import ShearStructure
from twinpy.file_io import write_poscar

class TwinBoundaryStructure(_BaseStructure):
    """
    twinboundary structure class
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
    def twintype(self):
        """
        twin type
        """
        return self._twintype

    def set_twintype(self, twintype):
        """
        set twintype
        """
        assert twintype == 1 or twintype == 2, \
                "twintype must be 1 or 2"
        self._twintype = twintype

    def _get_dichromatic_operation(self):
        """
        get dichromatic operation
        """
        if self._twintype == 1:
            W = np.array([[ 1, 0, 0],
                          [ 0, 1, 0],
                          [ 0, 0,-1]])
        elif self._twintype == 2:
            W = np.array([[-1, 0, 0],
                          [ 0, 1, 0],
                          [ 0, 0,-1]])
        return W

    def run(self, dichromatic=False, make_tb_flat=True):
        """
        build structure

        Note:
            the structure built is set self.output_structure

            currently dichromaitc=True is broken, because I do not understand
            how to set grey lattice points and atoms

            Future Edited
        """
        indices = self._indices.get_indices()
        shear = ShearStructure(lattice=self._hcp_lattice.lattice,
                               symbol=self._symbol,
                               wyckoff=self._wyckoff)
        shear.set_shear_strain_ratio(ratio=0.)
        shear.run()
        W = self._get_dichromatic_operation()
        M = shear.output_structure['lattice'].T
        lattice_points = shear.output_structure['lattice_points']['white']
        additional_points = \
                lattice_points[np.isclose(lattice_points[:,2], 0)] + \
                    np.array([0.,0.,1.])
        lattice_points = np.vstack((lattice_points, additional_points))
        lp_p_cart = np.dot(M, lattice_points.T).T
        atoms_p_cart = \
                np.dot(M, shear.output_structure['atoms_from_lattice_points']['white'].T).T
        R = np.array([
                self.indices['m'].get_cartesian(normalize=True),
                self.indices['eta1'].get_cartesian(normalize=True),
                self.indices['k1'].get_cartesian(normalize=True),
                ]).T
        lp_t_cart = np.dot(R,
                          np.dot(W,
                                 np.dot(np.linalg.inv(R),
                                        lp_p_cart.T))).T
        atoms_t_cart = np.dot(R,
                              np.dot(W,
                                     np.dot(np.linalg.inv(R),
                                            atoms_p_cart.T))).T
        tb_c = lp_p_cart[-1] - lp_t_cart[-1]
        tb_lattice = np.array([shear.output_structure['lattice'][0],
                               shear.output_structure['lattice'][1],
                               tb_c])

        # lattice points
        white_lp = np.dot(np.linalg.inv(tb_lattice.T), lp_p_cart.T).T % 1
        black_lp = np.dot(np.linalg.inv(tb_lattice.T), lp_t_cart.T).T % 1

        if dichromatic:
            grey_ix  = [ bl1 or bl2 or bl3 for bl1, bl2, bl3 in \
                             zip(np.isclose(white_lp[:,2], 0),
                                 np.isclose(white_lp[:,2], 0.5),
                                 np.isclose(white_lp[:,2], 1)) ]
            grey_ix_ = [ bl1 or bl2 or bl3 for bl1, bl2, bl3 in \
                             zip(np.isclose(black_lp[:,2], 0),
                                 np.isclose(black_lp[:,2], 0.5),
                                 np.isclose(black_lp[:,2], 1)) ]
            assert grey_ix == grey_ix_, "some unexpected error occured, check script"
            grey_lp = white_lp[grey_ix]

            white_lp = white_lp[[ not bl for bl in grey_ix ]]
            black_lp = black_lp[[ not bl for bl in grey_ix_ ]]

            # atoms from lattice points
            white_atoms = np.dot(np.linalg.inv(tb_lattice.T), atoms_p_cart.T).T
            black_atoms = np.dot(np.linalg.inv(tb_lattice.T), atoms_t_cart.T).T

            grey_atoms = (white_atoms + black_atoms) / 2
            symbols = [self._symbol] * (len(white_lp)+len(black_lp)+len(grey_lp)) \
                                     * len(self._atoms_from_lattice_points)
            return {'lattice': tb_lattice,
                    'lattice_points': {
                         'white': white_lp + np.array([self._xshift/(self._dim[0]*2),
                                                       self._yshift/(self._dim[1]*2),
                                                       0.]),
                         'black': black_lp + np.array([-self._xshift/(self._dim[0]*2),
                                                       -self._yshift/(self._dim[1]*2),
                                                       0.]),
                         'grey': grey_lp,
                                      },
                    'atoms_from_lattice_points': {
                         'white': white_atoms,
                         'black': black_atoms,
                         'grey': grey_atoms,
                                                 },
                    'symbols': symbols}


        elif make_tb_flat:
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
            white_atoms = np.dot(np.linalg.inv(tb_lattice.T), atoms_p_cart.T).T
            black_atoms = np.dot(np.linalg.inv(tb_lattice.T), atoms_t_cart.T).T
            atms = len(white_atoms)
            white_tb_atoms = np.hstack((white_atoms[:,:2], np.zeros(atms).reshape(atms,1)))
            black_tb_atoms = np.hstack((black_atoms[:,:2], np.zeros(atms).reshape(atms,1)))

            symbols = [self._symbol] * (len(white_lp)+len(black_lp)+len(white_tb_lp)+len(black_tb_lp)) \
                                     * len(self._atoms_from_lattice_points)
            return {'lattice': tb_lattice,
                    'lattice_points': {
                         'white': white_lp + np.array([self._xshift/(self._dim[0]*2),
                                                       self._yshift/(self._dim[1]*2),
                                                       0.]),
                         'white_tb': white_tb_lp + np.array([self._xshift/(self._dim[0]*2),
                                                             self._yshift/(self._dim[1]*2),
                                                             0.]),
                         'black': black_lp + np.array([-self._xshift/(self._dim[0]*2),
                                                       -self._yshift/(self._dim[1]*2),
                                                       0.]),
                         'black_tb': black_tb_lp + np.array([-self._xshift/(self._dim[0]*2),
                                                             -self._yshift/(self._dim[1]*2),
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

            # atoms from lattice points
            white_atoms = np.dot(np.linalg.inv(tb_lattice.T), atoms_p_cart.T).T
            black_atoms = np.dot(np.linalg.inv(tb_lattice.T), atoms_t_cart.T).T

            symbols = [self._symbol] * (len(white_lp)+len(black_lp)) \
                                     * len(self._atoms_from_lattice_points)
            return {'lattice': tb_lattice,
                    'lattice_points': {
                         'white': white_lp + np.array([self._xshift/(self._dim[0]*2),
                                                       self._yshift/(self._dim[1]*2),
                                                       0.]),
                         'black': black_lp + np.array([-self._xshift/(self._dim[0]*2),
                                                       -self._yshift/(self._dim[1]*2),
                                                       0.]),
                                      },
                    'atoms_from_lattice_points': {
                         'white': white_atoms,
                         'black': black_atoms,
                                                 },
                    'symbols': symbols}

        self._natoms = len(self.output_structure['symbols'])
