#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test twinpy/structure.py
"""

import unittest
import numpy as np
from copy import deepcopy
from twinpy.lattice.lattice import create_hexagonal_lattice
from twinpy.lattice.hexagonal_plane import HexagonalPlane

class TestHexagonalPlane(unittest.TestCase):

    def setUp(self):
        self.a = 3
        self.c = 5
        self.lattice = create_hexagonal_lattice(a=self.a, c=self.c)
        self.four = (1, 0, -1, 2)
        self.plane = HexagonalPlane(self.lattice, four=self.four)

    def tearDown(self):
        pass

    def test_attributes(self):
        # volume
        np.testing.assert_allclose(self.a**2*np.sqrt(3)/2*self.c,
                            self.plane.volume)

        # convert plane from four to three
        np.testing.assert_array_equal(self.plane.three, np.array([1, 0, 2]))

        # reset indices
        clone_plane = deepcopy(self.plane)
        clone_plane.reset_indices(three=self.plane.three * (-1))
        np.testing.assert_array_equal(clone_plane.four,
                                      np.array([-1, 0, 1, -2]))

    def test_get_direction_normal_to_plane(self):
        print(self.plane.get_direction_normal_to_plane().three)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHexagonalPlane)
    unittest.TextTestRunner(verbosity=2).run(suite)
