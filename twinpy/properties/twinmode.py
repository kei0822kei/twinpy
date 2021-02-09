#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module deals with properties of twin modes.
"""

import numpy as np
from copy import deepcopy
from itertools import permutations
from twinpy.lattice.lattice import Lattice, check_hexagonal_lattice
from twinpy.lattice.hexagonal_plane import HexagonalPlane
from twinpy.lattice.hexagonal_direction import HexagonalDirection
from twinpy.properties.hexagonal import get_atom_positions
from twinpy.common.utils import get_ratio


def check_supported_twinmode(twinmode:str):
    """
    Check input twinmode is supported.

    Args:
        twinmode: Choose from '10-12', '10-11', '11-22' or '11-21'
                  (which are supported).

    Raises:
        AssertionError: Input twinmode is not supported.
    """
    supported_twinmodes = ['10-12', '10-11', '11-22', '11-21']
    assert twinmode in supported_twinmodes, '%s is not supported' % twinmode


def get_shear_strain_function(twinmode:str):
    """
    Get shear strain.

    Args:
        twinmode: Choose from '10-12', '10-11', '11-22' or '11-21'
                  (which are supported).

    Returns:
        function: Function which returns shear strain
                  input arg is r.
    """
    check_supported_twinmode(twinmode)
    if twinmode == '10-12':
        func = lambda r: \
                abs(r**2-3) / (r*np.sqrt(3))
    elif twinmode == '10-11':
        func = lambda r: \
                abs(4*r**2-9) / (4*r*np.sqrt(3))
    elif twinmode == '11-22':
        func = lambda r: \
                abs(2*(r**2-2)) / (3*r)
    elif twinmode == '11-21':
        func = lambda r: \
                1 / r
    return func


def get_number_of_layers(twinmode:str) -> int:
    """
    Get the number of layers.

    Args:
        twinmode: Choose from '10-12', '10-11', '11-22' or '11-21'
                  (which are supported).

    Returns:
        int: The number of layes.
    """
    check_supported_twinmode(twinmode)
    if twinmode == '10-12':
        layers = 4
    elif twinmode == '10-11':
        layers = 8
    elif twinmode == '11-22':
        layers = 6
    elif twinmode == '11-21':
        layers = 2
    return layers


def get_twin_indices_by_Yoo() -> dict:
    """
    Get twin indices Yoo showed in his paper.

    Returns:
        dict: Twin indices.
    """
    dataset = {
            '10-12': {
                        'S'    : np.array([  1.,  1., -2.,  0.]),
                        'K1'   : np.array([  1.,  0., -1.,  2.]),
                        'K2'   : np.array([  1.,  0., -1., -2.]),
                        'eta1' : np.array([  1.,  0., -1., -1.]),
                        'eta2' : np.array([ -1.,  0.,  1., -1.]),
                     },

            '10-11': {
                        'S'    : np.array([  1.,  1., -2.,  0.]),
                        'K1'   : np.array([  1.,  0., -1.,  1.]),
                        'K2'   : np.array([  1.,  0., -1., -3.]),
                        'eta1' : np.array([  1.,  0., -1., -2.]),
                        'eta2' : np.array([  3.,  0., -3.,  2.]),
                     },

            '11-21': {
                        'S'    : np.array([  1.,  0., -1.,  0.]),
                        'K1'   : np.array([  1.,  1., -2.,  1.]),
                        'K2'   : np.array([  0.,  0.,  0.,  2.]),
                        'eta1' : np.array([-1/3,-1/3, 2/3,  2.]),
                        'eta2' : np.array([ 1/3, 1/3,-2/3,  0.]),
                     },
            '11-22': {
                        'S'    : np.array([  1.,  0., -1.,  0.]),
                        'K1'   : np.array([  1.,  1., -2.,  2.]),
                        'K2'   : np.array([  1.,  1., -2., -4.]),
                        'eta1' : np.array([ 1/3, 1/3,-2/3, -1.]),
                        'eta2' : np.array([ 2/3, 2/3,-4/3,  1.]),
                     },
              }

    return dataset


class TwinIndices():
    """
    Deals with twin indices.
    """

    def __init__(
           self,
           lattice:Lattice,
           twinmode:str,
           wyckoff:str,
           ):
        """
        Get twin indices of input twinmode.
        Twinmode must be '10-11', '10-12', '11-21' or '11-22'.

        Args:
            twinmode: Currently supported \
                            '10-11', '10-12', '11-22' and '11-21'.
            lattice: Lattice class object.
            wyckoff: Wyckoff letter, 'c' or 'd'.

        Returns:
            dict: Twin indices.

        Note:
            You can find customs how to set the four indices
            in the paper 'DEFORMATION TWINNING' by Christian (1995).
            Abstruct is as bellow

              1. Vector m normal to shear plane,
                 eta1 and eta2 form right hand system.
              2. The angle between eta1 and eta2 are abtuse.
              3. The angle between eta1 and k2 are acute.
              4. The angle between eta2 and k1 are acute.
                 Additional rule:
              5. K1 plane depends on wyckoff 'c' or 'd'
                 because in the case of {10-12}, either (10-12) or (-1012)
                 is chosen based on which plane are nearer to the neighbor
                 atoms

            In this algorithm, indices are set in order of

              a. k1    condition(5)
              b. eta2  condition(4)
              c. eta1  condition(2)
              d. k2    condition(3)
              e. m     condition(1)

        Todo:
            _twin_indices_orig is a little bit
            different from the ones 1981. Yoo especially 'S'
        """
        check_hexagonal_lattice(lattice.lattice)
        check_supported_twinmode(twinmode)
        self._lattice = lattice
        self._twinmode = twinmode
        self._layers = get_number_of_layers(twinmode)
        self._wyckoff = wyckoff
        self._indices_Yoo = self._get_indices_Yoo(self._lattice.lattice,
                                                  self._twinmode)
        self._indices = deepcopy(self._indices_Yoo)
        self._set_K1()
        self._set_k1_k2()
        self._reset_indices()
        self._set_shear_plane()

    @property
    def lattice(self):
        """
        Lattice.
        """
        return self._lattice

    @property
    def twinmode(self):
        """
        Twinmode.
        """
        return self._twinmode

    @property
    def layers(self):
        """
        Number of layers.
        """
        return self._layers

    @property
    def wyckoff(self):
        """
        Wyckoff.
        """
        return self._wyckoff

    @property
    def indices_Yoo(self):
        """
        Indices Yoo showed.
        """
        return self._indices_Yoo

    @property
    def indices(self):
        """
        Indices.
        """
        return self._indices

    def _get_indices_Yoo(self, lattice, twinmode) -> dict:
        """
        Get specific twinmode indices which are found in Yoo's paper.

        Returns:
            dict: indices by Yoo
        """
        indices_ = get_twin_indices_by_Yoo()[twinmode]
        indices = {}
        for plane in ['S', 'K1', 'K2']:
            indices[plane] = HexagonalPlane(lattice,
                                            four=indices_[plane])
        for direction in ['eta1', 'eta2']:
            indices[direction] = HexagonalDirection(lattice,
                                                    four=indices_[direction])
        return indices

    def _set_K1(self):
        """
        Set K1.

        Note:
            There are two candidates for K1 plane.
            One is the K1 plane (defimed as (hkil) plane)
            which is found in Yoo's paper, and
            the other is (-h-k-il) plane.
            The plane which has shorter dictance from nearest atoms
            is set as K1 plane.
            If (-h-k-il) is determined, every other planes and directions
            are fixed by multiplied (-1, -1, -1, 1).
        """
        arr = np.array([-1,-1,-1,1])
        atoms = get_atom_positions(self.wyckoff)
        K1_1 = self._indices['K1']
        K1_2 = deepcopy(K1_1)
        K1_2.reset_indices(four=K1_2.four*arr)
        if K1_1.get_distance_from_plane(atoms[0]) > \
               K1_2.get_distance_from_plane(atoms[0]):
            for key in ['S', 'K1', 'K2', 'eta1', 'eta2']:
                self._indices[key].reset_indices(
                        four=self._indices[key].four*arr)

    def _set_k1_k2(self):
        """
        Set k1 and k2.
        """
        self._indices['k1'] = \
                self._indices['K1'].get_direction_normal_to_plane()
        self._indices['k2'] = \
                self._indices['K2'].get_direction_normal_to_plane()

    def _reset_indices(self):
        """
        Reset indices.

        Raises:
            AssertionError: k1 is not orthogonal to eta1
            AssertionError: k2 is not orthogonal to eta2

        Note:
            conditions are as bellow

            - if the angle between k1 and eta2 is obtuse, eta2 -> -eta2
            - if the angle between eta1 and eta2 is acute, eta1 -> -eta1
            - if the angle between eta1 and k2 is obtuse, k2 -> -k2
            - check k1 is orthogonal to eta1
            - check k2 is orthogonal to eta2
        """
        # reset eta2
        if self.lattice.dot(self._indices['k1'].three,
                            self._indices['eta2'].three) < 0:
            self._indices['eta2'].inverse()

        # reset eta1
        if self.lattice.dot(self._indices['eta1'].three,
                            self._indices['eta2'].three) > 0:
            self._indices['eta1'].inverse()

        # reset K2
        if self.lattice.dot(self._indices['eta1'].three,
                            self._indices['k2'].three) < 0:
            self._indices['K2'].inverse()
            self._indices['k2'].inverse()

        # check k1 is orthogonal to eta1
        np.testing.assert_allclose(
                actual=self.lattice.dot(self._indices['k1'].three,
                                        self._indices['eta1'].three),
                desired=0,
                atol=1e-6,
                err_msg="k1 is not orthogonal to eta1")

        # check k2 is orthogonal to eta2
        np.testing.assert_allclose(
                actual=self.lattice.dot(self._indices['k2'].three,
                                        self._indices['eta2'].three),
                desired=0,
                atol=1e-6,
                err_msg="k2 is not orthogonal to eta2")

    def _set_shear_plane(self):
        """
        Set shear plane.

        Raises:
            RuntimeError: Could not find shear plane.

        Note:
            Set shear plane which fulfill the following conditions
            from six candidates
            ((hkil), (hikl), (khil), (kihl), (ihkl), (ikhl)).
        """
        S_four = self._indices['S'].four
        perms = map(list, permutations((0,1,2)))
        trial_S_fours = [ [S_four[i] for i in lst+[3]]
                          for lst in perms ]
        normal_directions = ['k1', 'k2', 'eta1', 'eta2']
        flag = 1
        for trial_S_four in trial_S_fours:
            trial_S = HexagonalPlane(self.lattice.lattice,
                                     four=np.array(trial_S_four, dtype='intc'))
            trial_m = trial_S.get_direction_normal_to_plane()
            dots = [ self.lattice.dot(trial_m.three,
                                      self._indices[direction].three)
                     for direction in normal_directions ]
            if np.allclose(np.array(dots), np.zeros(4), atol=1e-4):
                flag = 0
                break
        if flag == 1:
            raise RuntimeError("could not find shear plane")

        triple_product = \
                np.dot(trial_m.get_cartesian(),
                       np.cross(self._indices['eta1'].get_cartesian(),
                                self._indices['eta2'].get_cartesian()))
        if triple_product < 0:
            trial_S.inverse()
            trial_m.inverse()
        self._indices['S'] = trial_S
        self._indices['m'] = trial_m

    def get_supercell_matrix_for_parent(self) -> np.array:
        """
        Get supercell matrix for creating parent matrix.

        Returns:
            np.array: Sueprcell matrix.

        Note:
            Create lattice basis with integerized m, eta1 and eta2.
        """
        tf1 = np.array(get_ratio(self._indices['m'].three))
        tf2 = np.array(get_ratio(self._indices['eta1'].three))
        tf3 = np.array(get_ratio(self._indices['eta2'].three))
        supercell_matrix = np.vstack([tf1, tf2, tf3]).T

        return supercell_matrix

    def get_shear_strain_function(self):
        """
        Get shear strain function.
        """
        return get_shear_strain_function(self._twinmode)
