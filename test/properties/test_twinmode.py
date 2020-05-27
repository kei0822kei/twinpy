#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test twinpy/lattice/lattice.py
"""

import unittest
import numpy as np
from twinpy.lattice.lattice import Lattice, create_hexagonal_lattice
from twinpy.properties.twinmode import TwinIndices

class TestLattice(unittest.TestCase):

    def setUp(self):
        self.a, self.c = 2., 4.
        self.wyckoff = 'c'
        self.hexagonal = Lattice(create_hexagonal_lattice(a=self.a,
                                                          c=self.c))
        self.twinmodes = ['10-12']

    def tearDown(self):
        pass

    def test_get_twin_indices(self):
        for twinmode in self.twinmodes:
            indices = TwinIndices(lattice=self.hexagonal,
                                  twinmode=twinmode,
                                  wyckoff=self.wyckoff)
            twin_indices = indices.get_indices()
            print(twin_indices['S'].four)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLattice)
    unittest.TextTestRunner(verbosity=2).run(suite)
