#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
aiida interface
"""
import numpy as np
from twinpy.structure.base import get_phonopy_structure
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.common import NotExistentAttributeError
from aiida.orm import load_node

def check_process_class(node,
                        expected_process_class:str):
    """
    check process class of node is the same as the expected

    Args:
        node: aiida node
        expected_process_class (str): expected process class

    Raises:
        AssertionError: input node is not the same as expected
    """
    assert node.process_class.get_name() == expected_process_class, \
            "input node: {}, expected: {} (NOT MATCH)". \
            format(node.process_class, expected_process_class)

def get_cell_from_aiida(aiida_structure, get_scaled_positions=True):
    """
    cell = (lattice, scaled_positions, symbols)
    """
    lattice = np.array(aiida_structure.cell)
    positions = np.array([site.position for site in aiida_structure.sites])
    if get_scaled_positions:
        positions = np.dot(np.linalg.inv(lattice.T), positions.T).T
    symbols = [site.kind_name for site in aiida_structure.sites]
    return (lattice, positions, symbols)


@with_dbenv()
class RelaxWorkChain():
    """
    relax work chain class

       .. attribute:: att1

          Optional comment string.


       .. attribute:: att2

          Optional comment string.

    """

    def __init__(
           self,
           relax_pk:int,
       ):
        """
        Args:
            relax_pk (int): relax pk
        """
        node = load_node(relax_pk)
        check_process_class(node, 'RelaxWorkChain')

        self._node = node
        self._relax_pk = relax_pk
        self._initial_structure_pk = node.inputs.structure.pk
        self._initial_cell = get_cell_from_aiida(
                load_node(self._initial_structure_pk))
        self._final_structure_pk = node.outputs.relax__structure.pk
        self._final_cell = get_cell_from_aiida(
                load_node(self._final_structure_pk))
        self._set_forces()

    def _set_forces(self):
        self._forces = None
        try:
            self._forces = self.node.outputs.forces.get_array('final')
        except NotExistentAttributeError:
            print("output forces not found, so skip setting forces")

    @property
    def forces(self):
        """
        forces acting on atoms after relax
        """
        return self._forces

    def _set_stress(self):
        self._stress = None
        try:
            self._stress = self.node.outputs.stress.get_array('final')
        except NotExistentAttributeError:
            print("output stress not found, so skip setting forces")

    @property
    def stress(self):
        """
        stress acting on lattice after relax
        """
        return self._stress

    @property
    def initial_cell(self):
        """
        initial cell
        """
        return self._initial_cell

    @property
    def final_cell(self):
        """
        final cell
        """
        return self._final_cell

    def get_pks(self):
        """
        get related pks
        """
        return {
                'relax_pk': self._relax_pk,
                'initial_structure_pk': self._initial_structure_pk,
                'final_structure_pk': self._final_structure_pk,
               }
