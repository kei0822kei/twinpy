#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test twinpy/structure.py
"""

import unittest
import numpy as np
from copy import deepcopy
from twinpy.lattice.lattice import create_hexagonal_lattice
from twinpy.lattice.hexagonal_direction import HexagonalDirection

class TestHexagonalDirection(unittest.TestCase):

    def setUp(self):
        self.a = 3
        self.c = 5
        self.lattice = create_hexagonal_lattice(a=self.a, c=self.c)
        self.four = (1, 0, -1, 2)
        self.direction = HexagonalDirection(lattice=self.lattice, four=self.four)

    def tearDown(self):
        pass

    def test_attributes(self):
        # volume
        np.testing.assert_allclose(self.a**2*np.sqrt(3)/2*self.c,
                            self.direction.volume)

        # # convert direction from four to three
        # np.testing.assert_array_equal(self.direction.three, np.array([1, 0, 2]))

        # # reset indices
        # clone_direction = deepcopy(self.direction)
        # clone_direction.reset_indices(three=self.direction.three * (-1))
        # np.testing.assert_array_equal(clone_direction.four,
        #                               np.array([-1, 0, 1, -2]))



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHexagonalDirection)
    unittest.TextTestRunner(verbosity=2).run(suite)
