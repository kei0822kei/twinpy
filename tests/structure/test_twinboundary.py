#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is pytest for twinpy.structure.twinboudnary.
"""

from twinpy.structure.twinboundary import TwinBoundaryStructure


def test_TwinBoundaryStructure(ti_cell_wyckoff_c):
    """
    Check TwinBoundaryStructure class object.
    """
    twinmode = '10-12'
    twintype = 1
    wyckoff = 'c'

    layers = 10
    delta = 0.
    xshift = 0.
    yshift = 0.
    shear_strain_ratio = 0.

    lattice, _, symbols = ti_cell_wyckoff_c

    twinboundary = TwinBoundaryStructure(
            lattice=lattice,
            symbol=symbols[0],
            twinmode=twinmode,
            twintype=twintype,
            wyckoff=wyckoff)

    twinboundary.run(layers=layers,
                     delta=delta,
                     xshift=xshift,
                     yshift=yshift,
                     shear_strain_ratio=shear_strain_ratio)
