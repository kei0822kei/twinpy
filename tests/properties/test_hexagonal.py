#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is pytest for twinpy.properties.hexagonal.
"""

from copy import deepcopy
import numpy as np
from twinpy.properties import hexagonal

a = 2.93
c = 4.65


def test_check_hexagonal_lattice(ti_cell_wyckoff_c):
    """
    Check check_hexagonal_lattice.
    """
    hexagonal_lattice = ti_cell_wyckoff_c[0]
    hexagonal.check_hexagonal_lattice(lattice=hexagonal_lattice)


def test_check_cell_is_hcp(ti_cell_wyckoff_c, ti_cell_wyckoff_d):
    """
    Check check_cell_is_hcp.
    """
    for cell in [ti_cell_wyckoff_c, ti_cell_wyckoff_d]:
        hexagonal.check_cell_is_hcp(cell=cell)


def test_convert_direction():
    """
    Check convert_direction_from_four_to_three
    and convert_direction_from_three_to_four.

    Note:
        Let basis vectors for hexagonal lattice be a_1, a_2 and c,
        a_1 = [1,0,0] = 1/3[2,-1,-1,0].
    """
    def _test_convert_direction_from_three_to_four(three, four_expected):
        _four = hexagonal.convert_direction_from_three_to_four(
                three=three)
        np.testing.assert_allclose(_four, four_expected)

    def _test_convert_direction_from_four_to_three(four, three_expected):
        _three = hexagonal.convert_direction_from_four_to_three(
                four=four)
        np.testing.assert_allclose(_three, three_expected)

    a_1_three = np.array([1.,0.,0.])
    a_1_four = np.array([2.,-1.,-1.,0.]) / 3.
    _test_convert_direction_from_three_to_four(three=a_1_three,
                                               four_expected=a_1_four)
    _test_convert_direction_from_four_to_three(four=a_1_four,
                                               three_expected=a_1_three)


def test_hexagonal_direction(ti_cell_wyckoff_c):
    """
    Check HexagonalDirection.
    """
    def _test_reset_indices(hex_dr, three):
        _hex_dr = deepcopy(hex_dr)
        _hex_dr.reset_indices(three=three)
        _three_expected = _hex_dr.three
        np.testing.assert_allclose(three, _three_expected)

    def _test_inverse(hex_dr):
        _inv_hex_dr = deepcopy(hex_dr)
        _inv_hex_dr.inverse()
        _three = hex_dr.three
        _inv_three = _inv_hex_dr.three
        np.testing.assert_allclose(_three, _inv_three*(-1.))

    def _test_get_cartesian(hex_dr, cart_expected):
        _cart = hex_dr.get_cartesian(normalize=False)
        _cart_normalized = hex_dr.get_cartesian(normalize=True)
        _norm = np.linalg.norm(_cart_normalized)
        np.testing.assert_allclose(_cart, cart_expected)
        np.testing.assert_allclose(_norm, 1.)

    lattice = ti_cell_wyckoff_c[0]
    three_a1 = np.array([1.,0.,0.])  # a_1
    three_c = np.array([0.,0.,1.])  # c
    a1_cart = np.array([a,0.,0.])  # cartesian coordinate for vector a_1
    hex_dr_a1 = hexagonal.HexagonalDirection(lattice=lattice, three=three_a1)

    _test_reset_indices(hex_dr=hex_dr_a1,
                        three=three_c)
    _test_inverse(hex_dr=hex_dr_a1)
    _test_get_cartesian(hex_dr=hex_dr_a1, cart_expected=a1_cart)


def test_convert_plane():
    """
    Check convert_plane_from_four_to_three
    and convert_plane_from_three_to_four.

    Note:
        (10-12) plane is equal to (102).
    """
    def _test_convert_plane_from_three_to_four(three, four_expected):
        _four = hexagonal.convert_plane_from_three_to_four(
                three=three)
        np.testing.assert_allclose(_four, four_expected)

    def _test_convert_plane_from_four_to_three(four, three_expected):
        _three = hexagonal.convert_plane_from_four_to_three(
                four=four)
        np.testing.assert_allclose(_three, three_expected)

    twin_three = np.array([1.,0.,2.])
    twin_four = np.array([1.,0.,-1.,2.])

    _test_convert_plane_from_three_to_four(three=twin_three,
                                           four_expected=twin_four)
    _test_convert_plane_from_four_to_three(four=twin_four,
                                           three_expected=twin_three)


def test_hexagonal_plane(ti_cell_wyckoff_c):
    """
    Check HexagonalPlane.
    """
    def _test_reset_indices(hex_pln, four):
        _hex_pln = deepcopy(hex_pln)
        _hex_pln.reset_indices(four=four)
        _four = _hex_pln.four
        np.testing.assert_allclose(_four, four)

    def _test_inverse(hex_pln):
        _inv_hex_pln = deepcopy(hex_pln)
        _inv_hex_pln.inverse()
        four = hex_pln.four
        _inv_four = _inv_hex_pln.four
        np.testing.assert_allclose(_inv_four, four*(-1))

    def _test_get_distance_from_plane(hex_pln, frac_coord, d_expected):
        _d = hex_pln.get_distance_from_plane(frac_coord=frac_coord)
        np.testing.assert_allclose(_d, d_expected)

    lattice = ti_cell_wyckoff_c[0]
    basal_four = np.array([0.,0.,0.,1.])
    twin_four = np.array([1.,0.,-1.,2.])
    hex_pln_basal = hexagonal.HexagonalPlane(lattice=lattice,
                                             four=basal_four)
    hex_pln_twin = hexagonal.HexagonalPlane(lattice=lattice,
                                            four=twin_four)
    c_three = np.array([0.,0.,1.])

    _test_reset_indices(hex_pln=hex_pln_twin,
                        four=basal_four)
    _test_inverse(hex_pln=hex_pln_twin)
    _test_get_distance_from_plane(hex_pln=hex_pln_basal,
                                  frac_coord=c_three,
                                  d_expected=c)
