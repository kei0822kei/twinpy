#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
aiida interface
"""
import numpy as np
from twinpy.structure.base import get_phonopy_structure
from twinpy.api_twinpy import get_twinpy_from_cell
from twinpy.analysis.shear_analyzer import ShearAnalizer
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.common import NotExistentAttributeError
from aiida.orm import load_node, Node, QueryBuilder
from aiida.plugins import WorkflowFactory
from phonopy import Phonopy

RELAX_WF = WorkflowFactory('vasp.relax')
PHONOPY_WF = WorkflowFactory('phonopy.phonopy')

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


@with_dbenv()
class PhonopyWorkChain():
    """
    phonon work chain class
    """

    def __init__(
            self,
            phonopy_pk:int,
        ):
        """
        Args:
            relax_pk (int): relax pk
        """
        node = load_node(phonopy_pk)
        check_process_class(node, 'PhonopyWorkChain')

        self._node = node
        self._phonopy_pk = phonopy_pk
        self._structure_pk = node.inputs.structure.pk
        self._unitcell = get_cell_from_aiida(
                load_node(self._structure_pk))
        self._phonon_settings = node.outputs.phonon_setting_info.get_dict()
        self._force_sets = node.outputs.force_sets.get_array('force_sets')

    def get_unitcell(self):
        """
        get cell
        """
        return self._unitcell

    def get_phonon(self):
        """
        return phonopy object
        """
        phonon = Phonopy(
                get_phonopy_structure(self._unitcell),
                supercell_matrix=self._phonon_settings['supercell_matrix'],
                primitive_matrix=self._phonon_settings['primitive_matrix'])
        phonon.set_displacement_dataset(
                self._phonon_settings['displacement_dataset'])
        phonon.set_forces(self._force_sets)
        phonon.produce_force_constants()
        return phonon


@with_dbenv()
class ShearWorkChain():
    """
    shear work chain class
    """

    def __init__(
            self,
            shear_pk:int,
        ):
        """
        Args:
            relax_pk (int): relax pk
        """
        node = load_node(shear_pk)
        check_process_class(node, 'ShearWorkChain')

        self._node = node
        self._shear_pk = shear_pk
        self._shear_conf = node.inputs.shear_conf.get_dict()
        self._shear_ratios = \
            node.called[-1].outputs.shear_settings.get_dict()['shear_ratios']
        self._input_cell = get_cell_from_aiida(node.inputs.structure)

    def _get_pks(self, workflow):
        """
        get relax and phonopy pks
        """
        if workflow == 'relax':
            wf = RELAX_WF
        elif workflow == 'phonopy':
            wf = PHONOPY_WF
        else:
            raise RuntimeError("workflow must be 'relax' or 'phonopy'")
        node_qb = QueryBuilder()
        node_qb.append(Node, filters={'id':{'==': self._shear_pk}}, tag='wf')
        node_qb.append(wf, with_incoming='wf', project=['id'])
        nodes = node_qb.all()
        node_pks = [ node[0]  for node in nodes ]
        node_pks.reverse()
        return node_pks

    def _get_original_cells(self):
        """
        get original cell before standardize
        """
        orig_cells = []
        for ratio in self._shear_ratios:
            twinpy = get_twinpy_from_cell(
                    cell=self._input_cell,
                    twinmode=self._shear_conf['twinmode'])
            twinpy.set_shear(
                xshift=self._shear_conf['xshift'],
                yshift=self._shear_conf['yshift'],
                dim=[1,1,1],
                shear_strain_ratio=ratio,
                )
            # orig_cells.append(twinpy.shear.get_structure_for_export())
            orig_cells.append(twinpy.shear.get_base_primitive_cell(ratio))
        return orig_cells

    def get_analyzer(self):
        """
        get ShearAnalizer class object
        """
        relax_pks = self._get_pks(workflow='relax')
        phonon_pks = self._get_pks(workflow='phonopy')
        input_cells = [
            get_cell_from_aiida(load_node(relax_pk).inputs.structure)
            for relax_pk in relax_pks ]
        relax_cells = [
            get_cell_from_aiida(load_node(relax_pk).outputs.relax__structure)
            for relax_pk in relax_pks ]
        phonons = [
            PhonopyWorkChain(phonon_pk).get_phonon()
            for phonon_pk in phonon_pks ]
        analyzer = ShearAnalizer(
            structure_type=self._shear_conf['structure_type'],
            orig_cells=self._get_original_cells(),
            input_cells=input_cells)
        analyzer.set_relax_output_structures(relax_cells)
        analyzer.set_phonons(phonons)
        return analyzer
