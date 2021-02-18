#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is pytest for twinpy.structure.lattice.
"""

import numpy as np
from twinpy.common.kpoints import Kpoints


def test_kpoints(ti_cell_wyckoff_c):
    """
    Check kpoints.
    """
    def _test_get_mesh_offset_auto(kpt,
                                   kpoints_expected,
                                   interval,
                                   mesh,
                                   include_two_pi,
                                   decimal_handling,
                                   use_symmetry,
                                   ):
        kpoints = kpt.get_mesh_offset_auto(
                interval=interval,
                mesh=mesh,
                include_two_pi=include_two_pi,
                decimal_handling=decimal_handling,
                use_symmetry=use_symmetry,
                )
        assert kpoints == kpoints_expected

    def _test_intervals_from_mesh(kpt,
                                  intervals_expected,
                                  mesh,
                                  include_two_pi):
        intervals = kpt.get_intervals_from_mesh(
                mesh=mesh,
                include_two_pi=include_two_pi,
                )
        np.testing.assert_allclose(intervals, intervals_expected)

    def _test_get_dict(kpt,
                       reciprocal_lattice_expected,
                       interval,
                       mesh,
                       include_two_pi,
                       decimal_handling,
                       use_symmetry,
                       ):
        dic = kpt.get_dict(
                interval=interval,
                mesh=mesh,
                include_two_pi=include_two_pi,
                decimal_handling=decimal_handling,
                use_symmetry=use_symmetry,
                )
        np.testing.assert_allclose(dic['reciprocal_lattice'],
                                   reciprocal_lattice_expected)

    unit_cubic_lattice = np.eye(3)
    hex_lattice = ti_cell_wyckoff_c[0]
    _unit_kpt = Kpoints(lattice=unit_cubic_lattice)
    _hex_kpt = Kpoints(lattice=hex_lattice)

    _inputs = {
            'kpt': _unit_kpt,
            'reciprocal_lattice_expected': np.eye(3) * 2 * np.pi,
            'interval': 0.2,
            'mesh': None,
            'include_two_pi': True,
            'decimal_handling': None,
            'use_symmetry': False,
            }
    # test get_dict
    _test_get_dict(**_inputs)

    # test get_mesh_offset_auto
    del _inputs['reciprocal_lattice_expected']
    _inputs.update({
            'kpoints_expected': ([5, 5, 5], [0.5, 0.5, 0.5]),
            'include_two_pi': False,
            })
    _test_get_mesh_offset_auto(**_inputs)
    _inputs.update({
            'kpoints_expected': ([6, 6, 6], [0.5, 0.5, 0.5]),
            'use_symmetry': True,
            })
    _test_get_mesh_offset_auto(**_inputs)
    _inputs.update({
            'kpt': _hex_kpt,
            'interval': 0.15,
            'kpoints_expected': ([17, 17, 10], [0., 0., 0.5]),
            'include_two_pi': True,
            'use_symmetry': True,
            })
    _test_get_mesh_offset_auto(**_inputs)
    _inputs.update({
            'kpoints_expected': ([16, 16, 9], [0., 0., 0.5]),
            'decimal_handling': 'floor',
            'use_symmetry': False,
            })
    _test_get_mesh_offset_auto(**_inputs)
    _inputs.update({
            'kpoints_expected': ([17, 17, 10], [0., 0., 0.5]),
            'decimal_handling': 'ceil',
            'use_symmetry': False,
            })
    _test_get_mesh_offset_auto(**_inputs)

    # test get_mesh_offset_auto
    _test_intervals_from_mesh(_unit_kpt,
                              intervals_expected=[0.1, 0.2, 0.5],
                              mesh=[10, 5, 2],
                              include_two_pi=False)
