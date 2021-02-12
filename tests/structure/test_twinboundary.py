#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is pytest for twinpy.structure.twinboundary.
"""

import os
from pymatgen.io.vasp.inputs import Poscar
import twinpy
from twinpy.structure.base import check_same_cells
from twinpy.structure.twinboundary import TwinBoundaryStructure
from twinpy.interfaces.pymatgen import get_cell_from_pymatgen_structure


def test_TwinBoundaryStructure(ti_cell_wyckoff_c):
    """
    Check TwinBoundaryStructure class object.
    """
    def _test_run(twinmode, filename):
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
                           wyckoff=wyckoff,
                           )
        twinboundary.run(
                layers=layers,
                delta=delta,
                xshift=xshift,
                yshift=yshift,
                shear_strain_ratio=shear_strain_ratio,
                )
        cell = twinboundary.get_cell_for_export(get_lattice=False,
                                                move_atoms_into_unitcell=True)

        pos = Poscar.from_file(filename)
        cell_expected = get_cell_from_pymatgen_structure(pos.structure)

        is_same = check_same_cells(cell, cell_expected)
        assert is_same

    _twinpy_dir = os.path.dirname(os.path.dirname(twinpy.__file__))
    for _twinmode in ['10-11', '10-12', '11-21', '11-22']:
        _name = 'tb_{}_s0_typ1.poscar'.format(_twinmode)
        _filename = os.path.join(_twinpy_dir,
                                 'tests',
                                 'data',
                                 'twinboundary_poscar',
                                 _name)
        _test_run(_twinmode, _filename)
