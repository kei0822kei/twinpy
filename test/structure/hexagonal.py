#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test twinpy/structure.py
"""

import unittest
import numpy as np
from twinpy.structure.hexagonal import HexagonalStructure

class TestHexagonalStructure(unittest.TestCase):

    def setUp(self):
        self.a = 2.93
        self.c = 4.65
        self.specie = 'Ti'
        self.wyckoffs = ['c', 'd']

    def tearDown(self):
        pass

    def test_init(self):
        """
        test make HexagonalStructure object

        Note:
            - failcase_1: input lattice
            - failcase_2: invalid wyckoff
            - failcase_3: a axis is negative value
        """
        for wyckoff in self.wyckoffs:
            structure = HexagonalStructure(
                    a=self.a,
                    c=self.c,
                    specie=self.specie,
                    wyckoff=wyckoff)
        try:
            failcase_1 = HexagonalStructure(
                    a=self.a,
                    c=self.c,
                    specie=self.specie,
                    wyckoff='c',
                    lattice=np.array([1,1,1]))
        except ValueError:
            pass
        try:
            failcase_2 = HexagonalStructure(
                    a=self.a,
                    c=self.c,
                    specie=self.specie,
                    wyckoff='a')
        except AssertionError:
            pass
        try:
            failcase_3 = HexagonalStructure(
                    a=-3.,
                    c=4.,
                    specie=self.specie,
                    wyckoff='c')
        except AssertionError:
            pass

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHexagonalStructure)
    unittest.TextTestRunner(verbosity=2).run(suite)
