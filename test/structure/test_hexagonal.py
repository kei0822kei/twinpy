#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test twinpy/structure.py
"""

import unittest
import numpy as np
from twinpy.structure.hexagonal import (is_hcp,
                                        HexagonalStructure)

class TestHexagonalStructure(unittest.TestCase):

    def setUp(self):
        self.a = 2.93
        self.c = 4.65
        self.symbol = 'Ti'
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
            - failcase_4: not hcp
        """
        for wyckoff in self.wyckoffs:
            structure = HexagonalStructure(
                    a=self.a,
                    c=self.c,
                    symbol=self.symbol,
                    wyckoff=wyckoff)
        try:
            failcase_1 = HexagonalStructure(
                    a=self.a,
                    c=self.c,
                    symbol=self.symbol,
                    wyckoff='c',
                    lattice=np.array([1,1,1]))
            raise RuntimeError("unexpectedly passed fail case!")
        except ValueError:
            pass
        try:
            failcase_2 = HexagonalStructure(
                    a=self.a,
                    c=self.c,
                    symbol=self.symbol,
                    wyckoff='a')
            raise RuntimeError("unexpectedly passed fail case!")
        except AssertionError:
            pass
        try:
            failcase_3 = HexagonalStructure(
                    a=-3.,
                    c=4.,
                    symbol=self.symbol,
                    wyckoff='c')
            raise RuntimeError("unexpectedly passed fail case!")
        except AssertionError:
            pass
        try:
            is_hcp(structure.hexagonal_lattice.lattice,
                   structure.atoms_from_lattice_points,
                   ['Ti', 'Mg'])
            raise RuntimeError("unexpectedly passed fail case!")
        except AssertionError:
            pass

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHexagonalStructure)
    unittest.TextTestRunner(verbosity=2).run(suite)
