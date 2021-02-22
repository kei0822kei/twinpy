#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is pytest for twinpy.structure.shear.
"""

import os
import numpy as np
from pymatgen.io.vasp.inputs import Poscar
import twinpy
from twinpy.structure.base import check_same_cells
from twinpy.structure.shear import ShearStructure
from twinpy.interfaces.pymatgen import get_cell_from_pymatgen_structure
from twinpy.structure.lattice import CrystalLattice


def test_ShearStructure(ti_cell_wyckoff_c):
    """
    Check ShearBoundaryStructure class object.
    """
    def _test_run(twinmode,
                  shear_strain_ratio,
                  filename):
        wyckoff = 'c'
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
        assert is_same

    def _test_expansion():
        wyckoff = 'c'
        lattice, _, symbols = ti_cell_wyckoff_c
        shear_strain_ratio = 0.3
        twinmode = '11-21'
        expansion_ratios = np.array([1.1, 0.9 ,1.2])
        shear_orig = ShearStructure(
                        lattice=lattice,
                        symbol=symbols[0],
                        shear_strain_ratio=shear_strain_ratio,
                        twinmode=twinmode,
                        wyckoff=wyckoff,
                        )
        shear_expand = ShearStructure(
                          lattice=lattice,
                          symbol=symbols[0],
                          shear_strain_ratio=shear_strain_ratio,
                          twinmode=twinmode,
                          wyckoff=wyckoff,
                          )
        shear_lattices = []
        shear_expand.set_expansion_ratios(expansion_ratios=expansion_ratios)
        for shr in [shear_orig, shear_expand]:
            shr.run(is_primitive=False)
            shear_lattice = shr.get_cell_for_export(
                    get_lattice=False,
                    move_atoms_into_unitcell=True)[0]
            shear_lattices.append(shear_lattice)
        shear_lattice_expand = CrystalLattice(
                lattice=shear_lattices[0]).get_expanded_lattice(
                        expansion_ratios=expansion_ratios)
        np.testing.assert_allclose(shear_lattice_expand,
                                   shear_lattices[1])

    _twinpy_dir = os.path.dirname(os.path.dirname(twinpy.__file__))

    # test run
    for _twinmode in ['10-11', '10-12', '11-21', '11-22']:
        for _ratio in [0, 1]:
            _name = 'shr_{}_s{}.poscar'.format(_twinmode, _ratio)
            _filename = os.path.join(_twinpy_dir,
                                     'tests',
                                     'data',
                                     'shear_poscar',
                                     _name)
            _test_run(_twinmode, _ratio, _filename)

    # test expansion
    _test_expansion()
