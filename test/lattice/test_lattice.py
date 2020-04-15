#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test twinpy/lattice/lattice.py
"""

import unittest
import numpy as np
from twinpy.lattice.lattice import (check_hexagonal_lattice,
                                    Lattice)

class TestLattice(unittest.TestCase):

    def setUp(self):
        self.a, self.b, self.c = 2., 3., 4.
        self.orthogonal = Lattice(self.get_orthogonal(a=self.a,
                                                      b=self.b,
                                                      c=self.c))
        self.hexagonal = Lattice(self.get_hexagonal(a=self.a,
                                                    c=self.c))

    def tearDown(self):
        pass

    def get_hexagonal(self, a, c):
        lattice = np.array([[  1.,           0., 0.],
                            [-0.5, np.sqrt(3)/2, 0.],
                            [  0.,           0., 1.]]) \
                           * np.array([[a,a,c]]).T
        return lattice

    def get_orthogonal(self, a, b, c):
        lattice = np.eye(3) * np.array([[a,b,c]]).T
        return lattice

    def test_check_hexagonal_lattice(self):
        check_hexagonal_lattice(self.hexagonal.lattice)

    def test_reciprocal_lattice(self):
        """
        check reciprocal lattice
        """
        expected = np.diag([1/self.a, 1/self.b, 1/self.c])
        actual = self.orthogonal.reciprocal_lattice
        np.testing.assert_allclose(expected, actual)

        expected = np.array([[np.sqrt(3)/2, 0.5,      0.     ],
                            [      0.     , 1. ,      0.     ],
                            [      0.     , 0. , np.sqrt(3)/2]]) \
                       * np.array([[self.a*self.c,
                                    self.a*self.c,
                                    self.a**2]]) \
                             / self.hexagonal.volume
        actual = self.hexagonal.reciprocal_lattice
        np.testing.assert_allclose(expected, actual)

    def test_metric(self):
        """
        check metric

        Notes:
            -- metric of orthogonal lattice
            -- metric of hexagonal lattice
        """
        ortho_metric = np.diag([self.a**2, self.b**2, self.c**2])
        np.testing.assert_allclose(ortho_metric, self.orthogonal.metric)

        hex_metric = np.array([[ 1. , -0.5, 0.],
                               [-0.5,  1. , 0.],
                               [ 0. ,  0. , 1.]]) \
                              * np.array([[self.a**2,self.a**2,self.c**2]]).T
        np.testing.assert_allclose(hex_metric, self.hexagonal.metric)

    def test_dot(self):
        frac_1 = np.array([1,0,0])
        frac_2 = np.array([0,1,0])

        ortho_actual = self.orthogonal.dot(frac_1, frac_2)
        ortho_expected = 0.
        np.testing.assert_allclose(ortho_expected, ortho_actual)

        hex_actual = self.hexagonal.dot(frac_1, frac_2)
        hex_expected = -2.
        np.testing.assert_allclose(hex_expected, hex_actual)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLattice)
    unittest.TextTestRunner(verbosity=2).run(suite)
