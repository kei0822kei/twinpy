#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is pytest for twinpy.hexagonal
"""

import numpy as np
from twinpy.properties import hexagonal
from copy import deepcopy

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
        lattice, scaled_positions, symbols = cell
        hexagonal.check_cell_is_hcp(lattice=lattice,
                                    scaled_positions=scaled_positions,
                                    symbols=symbols)


def test_convert_direction():
    """
    Check convert_direction_from_four_to_three
    and convert_direction_from_three_to_four.

    Note:
        Let basis vectors for hexagonal lattice be a_1, a_2 and c,
        a_1 = [1,0,0] = 1/3[2,-1,-1,0].
    """
    a_1_three = np.array([1.,0.,0.])
    a_1_four = np.array([2.,-1.,-1.,0.]) / 3.

    # convert_direction_from_three_to_four
    _a_1_four = hexagonal.convert_direction_from_three_to_four(
            three=a_1_three)
    np.testing.assert_allclose(_a_1_four, a_1_four)

    # convert_direction_from_four_to_three
    _a_1_three = hexagonal.convert_direction_from_four_to_three(
            four=a_1_four)
    np.testing.assert_allclose(_a_1_three, a_1_three)


def test_hexagonal_direction(ti_cell_wyckoff_c):
    """
    Check HexagonalDirection.
    """
    lattice = ti_cell_wyckoff_c[0]
    three_a1 = np.array([1.,0.,0.])  # a_1
    three_c = np.array([0.,0.,1.])  # c
    hex_dr_a1 = hexagonal.HexagonalDirection(lattice=lattice, three=three_a1)

    # reset_indices
    _hex_dr_c = deepcopy(hex_dr_a1)
    _hex_dr_c.reset_indices(three=three_c)
    _three_c = _hex_dr_c.three
    np.testing.assert_allclose(_three_c, three_c)

    # inverse
    _inv_hex_dr_a1 = deepcopy(hex_dr_a1)
    _inv_hex_dr_a1.inverse()
    _inv_a1_three = _inv_hex_dr_a1.three
    np.testing.assert_allclose(_inv_a1_three, (-1.)*three_a1)

    # get_cartesian
    a1_cart = np.array([a,0.,0.])
    a1_cart_norm = np.array([1.,0.,0.])
    _a1_cart = hex_dr_a1.get_cartesian(normalize=False)
    np.testing.assert_allclose(_a1_cart, a1_cart)
    _a1_cart_norm = hex_dr_a1.get_cartesian(normalize=True)
    np.testing.assert_allclose(_a1_cart_norm, a1_cart_norm)


def test_convert_plane():
    """
    Check convert_plane_from_four_to_three
    and convert_plane_from_three_to_four.

    Note:
        (10-12) plane is equal to (102).
    """
    plane_three = np.array([1.,0.,2.])
    plane_four = np.array([1.,0.,-1.,2.])

    # convert_plane_from_three_to_four
    _plane_four = hexagonal.convert_plane_from_three_to_four(
            three=plane_three)
    np.testing.assert_allclose(plane_four, _plane_four)

    # convert_plane_from_four_to_three
    _plane_three = hexagonal.convert_plane_from_four_to_three(
            four=plane_four)
    np.testing.assert_allclose(plane_three, _plane_three)


def test_hexagonal_plane(ti_cell_wyckoff_c):
    """
    Check HexagonalPlane.
    """
    lattice = ti_cell_wyckoff_c[0]
    basal_four = np.array([0.,0.,0.,1.])
    twin_four = np.array([1.,0.,-1.,2.])
    hex_pln_basal = hexagonal.HexagonalPlane(lattice=lattice,
                                             four=basal_four)
    hex_pln_twin = hexagonal.HexagonalPlane(lattice=lattice,
                                            four=twin_four)
    c_four = np.array([0.,0.,0.,1.])
    frac_coord = np.array([0.,0.,1.])

    # reset_indices
    _hex_pln_basal = deepcopy(hex_pln_twin)
    _hex_pln_basal.reset_indices(four=basal_four)
    _hex_pln_basal_four = _hex_pln_basal.four
    np.testing.assert_allclose(_hex_pln_basal_four, basal_four)

    # inverse
    _inv_hex_pln_twin = deepcopy(hex_pln_twin)
    _inv_hex_pln_twin.inverse()
    _inv_twin_four = _inv_hex_pln_twin.four
    np.testing.assert_allclose(_inv_twin_four, (-1.)*twin_four)

    # get_distance_from_plane
    _c_norm = hex_pln_basal.get_distance_from_plane(
            frac_coord=frac_coord)
    np.testing.assert_allclose(_c_norm, c)
