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

    hex_lattice = ti_cell_wyckoff_c[0]
    hex_lat = Lattice(lattice=hex_lattice)

    _test_reciprocal_lattice(hex_lat)
