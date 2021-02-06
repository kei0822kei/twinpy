#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is pytest for twinpy.structure.lattice.
"""

import numpy as np
from twinpy.structure.lattice import Lattice


def test_lattice(ti_cell_wyckoff_c):
    """
    Check Lattice.

    Todo:
        Write test for 'get_diff' and get_midpoint
        after refactering.
    """
    def _test_reciprocal_lattice(lat):
        _direct_lattice = lat.lattice
        _recip_lattice_expected = lat.reciprocal_lattice

        recip_bases = []
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            recip_bases.append(
                np.cross(_direct_lattice[j], _direct_lattice[k]) / lat.volume)
        _recip_lattice = np.array(recip_bases)
        np.testing.assert_allclose(_recip_lattice, _recip_lattice_expected)

    def _test_abc_angles_cos_angles_dot(lat):
        """
        check dot
        """
        v1 = np.array([1. ,0., 0.])
        v2 = np.array([0. ,1., 0.])
        _abc = lat.abc
        _cos_angles = lat.cos_angles
        _inner_product = lat.dot(first_coord=v1,
                                 second_coord=v2,
                                 is_cartesian=False)
        _inner_product_expected = \
                _abc[0] * _abc[1] * _cos_angles[2]
        np.testing.assert_allclose(_inner_product, _inner_product_expected)

    def _test_convert_coordinate(lat):
        """
        check convert_fractional_to_cartesian
        check convert_cartesian_to_fractional
        """
        a, _, c = lat.abc
        v1_frac = np.array([0., 0., 0.5])
        v2_frac = np.array([[0., 0., 0.5],
                            [1., 0., 0. ]])
        v1_cart_expected = np.array([0., 0., c/2])
        v2_cart_expected = np.array([[0., 0., c/2],
                                     [a,  0.,  0.]])
        _v1_cart = lat.convert_fractional_to_cartesian(frac_coords=v1_frac)
        _v2_cart = lat.convert_fractional_to_cartesian(frac_coords=v2_frac)
        _v1_frac = lat.convert_cartesian_to_fractional(cart_coords=_v1_cart)
        _v2_frac = lat.convert_cartesian_to_fractional(cart_coords=_v2_cart)
        np.testing.assert_allclose(_v1_cart, v1_cart_expected)
        np.testing.assert_allclose(_v2_cart, v2_cart_expected)
        np.testing.assert_allclose(_v1_frac, v1_frac)
        np.testing.assert_allclose(_v2_frac, v2_frac)

    def _test_get_norm(lat):
        _a_norm_expected = lat.abc[0]
        _a_frac = np.array([1., 0., 0.])
        _a_cart = np.array([_a_norm_expected,0.,0.])
        _a_norm_from_frac = lat.get_norm(coord=_a_frac,
                                         is_cartesian=False,
                                         with_periodic=False)
        _a_norm_from_cart = lat.get_norm(coord=_a_cart,
                                         is_cartesian=True,
                                         with_periodic=False)
        np.testing.assert_allclose(_a_norm_from_frac, _a_norm_expected)
        np.testing.assert_allclose(_a_norm_from_cart, _a_norm_expected)

    hex_lattice = ti_cell_wyckoff_c[0]
    hex_lat = Lattice(lattice=hex_lattice)

    _test_reciprocal_lattice(lat=hex_lat)
    _test_abc_angles_cos_angles_dot(lat=hex_lat)
    _test_convert_coordinate(lat=hex_lat)
    _test_get_norm(lat=hex_lat)
