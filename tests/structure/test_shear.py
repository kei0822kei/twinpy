#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is pytest for twinpy.structure.shear.
"""

import os
from pymatgen.io.vasp.inputs import Poscar
import twinpy
from twinpy.structure.base import check_same_cells
from twinpy.structure.shear import ShearStructure
from twinpy.interfaces.pymatgen import get_cell_from_pymatgen_structure


def test_TwinBoundaryStructure(ti_cell_wyckoff_c):
    """
    Check TwinBoundaryStructure class object.
    """
    def _test_run(twinmode, shear_strain_ratio, filename):
        twinmode = twinmode
        twintype = 1
        wyckoff = 'c'
        layers = 10
        delta = 0.
        shear_strain_ratio = shear_strain_ratio
        lattice, _, symbols = ti_cell_wyckoff_c
        shear = ShearStructure(
                lattice=lattice,
                symbol=symbols[0],
                shear_strain_ratio=shear_strain_ratio,
                twinmode=twinmode,
                wyckoff=wyckoff,
                )
        shear.run(is_primitive=False)
        cell = shear.get_cell_for_export(get_lattice=False,
                                         move_atoms_into_unitcell=True)

        pos = Poscar.from_file(filename)
        cell_expected = get_cell_from_pymatgen_structure(pos.structure)

        is_same = check_same_cells(cell, cell_expected)
        if not is_same:
            assert RuntimeError("Failed to create shear structure.")

    _twinpy_dir = os.path.dirname(os.path.dirname(twinpy.__file__)) 
    for _twinmode in ['10-11', '10-12', '11-21', '11-22']:
        for i, _ratio in enumerate([0, 1]):
            _parent_shear = 'parent' if i == 0 else 'shear'
            _name = '{}_dl{}_typ1.poscar'.format(_twinmode, _ratio)
            _filename = os.path.join(_twinpy_dir,
                                     'tests',
                                     'data',
                                     'shear_poscar',
                                     _name)
            _test_run(_twinmode, _ratio, _filename)
