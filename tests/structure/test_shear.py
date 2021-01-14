#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" test twinpy/structure.py
"""

import unittest
import numpy as np
from twinpy.lattice.lattice import Lattice, get_hexagonal_lattice_from_a_c
from twinpy.structure.shear import ShearStructure

class TestShearStructure(unittest.TestCase):

    def setUp(self):
        self.a = 2.93
        self.c = 4.65
        self.lattice = get_hexagonal_lattice_from_a_c(self.a, self.c)
        self.symbol = 'Ti'
        self.wyckoffs = ['c', 'd']
        self.twinmodes = ['10-12', '10-11', '11-21', '11-22']
        self.dim = (2,2,2)
        self.shear_strain_ratio = 0.5
        self.xshift = 0.
        self.yshift = 0.

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
        for twinmode in self.twinmodes:
            for wyckoff in self.wyckoffs:
                shear = ShearStructure(
                        lattice=self.lattice,
                        symbol=self.symbol,
                        wyckoff=wyckoff)
                shear.set_parent(twinmode=twinmode)
                shear.run(
                        shear_strain_ratio=self.shear_strain_ratio,
                        dim=self.dim,
                        xshift=self.xshift,
                        yshift=self.yshift,
                        )
                # shear.get_phonopy_structure(structure_type='base')
                # conv = shear.get_phonopy_structure(structure_type='conventional')
                prim = shear.get_phonopy_structure(structure_type='primitive')
        # try:
        #     failcase_1 = get_hexagonal_structure_from_a_c(
        #             a=self.a,
        #             c=self.c,
        #             symbol=self.symbol,
        #             wyckoff='c',
        #             lattice=np.array([1,1,1]))
        #     raise RuntimeError("unexpectedly passed fail case!")
        # except ValueError:
        #     pass
        # try:
        #     failcase_2 = get_hexagonal_structure_from_a_c(
        #             a=self.a,
        #             c=self.c,
        #             symbol=self.symbol,
        #             wyckoff='a')
        #     raise RuntimeError("unexpectedly passed fail case!")
        # except AssertionError:
        #     pass
        # try:
        #     failcase_3 = get_hexagonal_structure_from_a_c(
        #             a=-3.,
        #             c=4.,
        #             symbol=self.symbol,
        #             wyckoff='c')
        #     raise RuntimeError("unexpectedly passed fail case!")
        # except AssertionError:
        #     pass
        # try:
        #     is_hcp(lattice=structure.hexagonal_lattice.lattice,
        #            scaled_positions=structure.atoms_from_lattice_points,
        #            symbols=['Ti', 'Mg'])
        #     raise RuntimeError("unexpectedly passed fail case!")
        # except AssertionError:
        #     pass

    # def test_get_shear_properties(self):
    #     structure = get_hexagonal_structure_from_a_c(
    #             a=self.a,
    #             c=self.c,
    #             symbol=self.symbol,
    #             wyckoff=self.wyckoffs[0])
    #     for twinmode in self.twinmodes:
    #         structure.set_parent(twinmode=twinmode)
    #         structure.set_shear_ratio(0.6)
    #         rotation = structure.get_shear_properties()['rotation']
    #         np.testing.assert_allclose(np.linalg.norm(rotation, axis=0),
    #                                    np.array([1,1,1]))
    #         np.testing.assert_allclose(np.linalg.norm(rotation, axis=1),
    #                                    np.array([1,1,1]))

    # def test_get_parent_structure(self):
    #     structure = get_hexagonal_structure_from_a_c(
    #             a=self.a,
    #             c=self.c,
    #             symbol=self.symbol,
    #             wyckoff=self.wyckoffs[0])
    #     for twinmode in self.twinmodes:
    #         structure.set_parent(twinmode)
    #         structure.set_shear_ratio(0.)
    #         structure.set_dimension(np.array([2,1,2]))
    #         structure.run(is_primitive=False)
    #         output_lattice = Lattice(structure.output_structure['lattice'])
    #         # np.testing.assert_allclose(np.array([90., 90., 90.]),
    #         #                            output_lattice.angles,
    #         #                            err_msg=twinmode)
    #         structure.get_poscar(filename="/home/mizo/parent"+twinmode+".poscar", get_lattice=True)

    # def test_get_twinboundary_structure(self):
    #     structure = get_hexagonal_structure_from_a_c(
    #             a=self.a,
    #             c=self.c,
    #             symbol=self.symbol,
    #             wyckoff=self.wyckoffs[0])
    #     for twinmode in self.twinmodes:
    #         structure.set_parent(twinmode)
    #         # structure.set_dimension(np.array([2,1,2]))
    #         # structure.set_yshift(0.5)
    #         structure.set_twintype(twintype=1)
    #         structure.run()
    #         output_lattice = Lattice(structure.output_structure['lattice'])
    #         # structure.get_poscar(filename="/home/mizo/tb"+twinmode+".poscar", get_lattice=True)
    #         # structure.get_poscar(filename="/home/mizo/tb"+twinmode+"_shift.poscar", get_lattice=False)
    #         structure.get_poscar(filename="/home/mizo/tb"+twinmode+".poscar", get_lattice=False)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestShearStructure)
    unittest.TextTestRunner(verbosity=2).run(suite)
