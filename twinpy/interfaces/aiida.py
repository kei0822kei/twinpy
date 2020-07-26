#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aiida interface for twinpy.
"""
import numpy as np
from twinpy.api_twinpy import get_twinpy_from_cell
from twinpy.analysis.shear_analyzer import ShearAnalyzer
from twinpy.interfaces.phonopy import get_phonopy_structure
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.common import NotExistentAttributeError
from aiida.orm import load_node, Node, QueryBuilder, StructureData
from aiida.plugins import WorkflowFactory
from phonopy import Phonopy

RELAX_WF = WorkflowFactory('vasp.relax')
PHONOPY_WF = WorkflowFactory('phonopy.phonopy')


def check_process_class(node,
                        expected_process_class:str):
    """
    Check process class of node is the same as the expected.

    Args:
        node: aiida node
        expected_process_class (str): expected process class

    Raises:
        AssertionError: input node is not the same as expected
    """
    assert node.process_class.get_name() == expected_process_class, \
            "input node: {}, expected: {} (NOT MATCH)". \
            format(node.process_class, expected_process_class)


def get_aiida_structure(cell:tuple):
    """
    Get aiida structure from input cell.

    Args:
        cell (tuple): cell = (lattice, scaled_positions, symbols)

    Returns:
        StructureData: aiida structure data
    """
    structure = StructureData(cell=cell[0])
    for symbol, scaled_position in zip(cell[2], cell[1]):
        position = np.dot(cell[0].T,
                          scaled_position.reshape(3,1)).reshape(3)
        structure.append_atom(position=position, symbols=symbol)
    return structure


def get_cell_from_aiida(structure:StructureData,
                        get_scaled_positions:bool=True):
    """
    Get cell from input aiida structure.

    Args:
        structure (StructureData): aiida structure data
        get_scaled_positions (bool): if True, return scaled positions

    Returns:
        tuple: cell
    """
    lattice = np.array(structure.cell)
    positions = np.array([site.position for site in structure.sites])
    if get_scaled_positions:
        positions = np.dot(np.linalg.inv(lattice.T), positions.T).T
    symbols = [site.kind_name for site in structure.sites]
    return (lattice, positions, symbols)


def get_pks(pk:int,
            workflow_name:str) -> dict:
    """
    Get workflow pk in the specified pk.

    Args:
        pk (int): input pk
        workflow_name (str): workflow name such as 'vasp.relax',
                             'phonopy.phonopy'

    Returns:
        dict: workflow pks in input pk
    """
    wf = WorkflowFactory(workflow_name)
    node_qb = QueryBuilder()
    node_qb.append(Node, filters={'id':{'==': pk}}, tag='wf')
    node_qb.append(wf, with_incoming='wf', project=['id'])
    nodes = node_qb.all()
    node_pks = [ node[0] for node in nodes ]
    node_pks.reverse()
    return node_pks


@with_dbenv()
class RelaxWorkChain():
    """
    Relax work chain class.
    """

    def __init__(
            self,
            pk:int,
            ):
        """
        Args:
            pk (int): relax pk
        """
        node = load_node(pk)
        check_process_class(node, 'RelaxWorkChain')

        self._node = node
        self._pk = pk
        self._initial_structure_pk = node.inputs.structure.pk
        self._initial_cell = get_cell_from_aiida(
                load_node(self._initial_structure_pk))
        self._final_structure_pk = node.outputs.relax__structure.pk
        self._final_cell = get_cell_from_aiida(
                load_node(self._final_structure_pk))
        self._stress = None
        self._set_stress()
        self._forces = None
        self._set_forces()

    @property
    def node(self):
        """
        RelaxWorkChain node.
        """
        return self._node

    @property
    def pk(self):
        """
        RelaxWorkChain pk.
        """
        return self._pk

    def _set_forces(self):
        """
        Set forces.
        """
        try:
            self._forces = self._node.outputs.forces.get_array('final')
        except NotExistentAttributeError:
            print("output forces not found, so skip setting forces")

    @property
    def forces(self):
        """
        Forces acting on atoms after relax.
        """
        return self._forces

    def _set_stress(self):
        """
        Set stress acting on lattice after relax.
        """
        try:
            self._stress = self._node.outputs.stress.get_array('final')
        except NotExistentAttributeError:
            print("output stress not found, so skip setting forces")

    @property
    def stress(self):
        """
        Stress acting on lattice after relax.
        """
        return self._stress

    @property
    def initial_cell(self):
        """
        Initial cell.
        """
        return self._initial_cell

    @property
    def final_cell(self):
        """
        Final cell.
        """
        return self._final_cell

    def get_pks(self) -> dict:
        """
        Get pks.

        Returns:
            dict: containing relax pk and structure pk
        """
        return {
                 'relax_pk': self._pk,
                 'initial_structure_pk': self._initial_structure_pk,
                 'final_structure_pk': self._final_structure_pk,
               }


@with_dbenv()
class PhonopyWorkChain():
    """
    Phononpy work chain class.
    """

    def __init__(
            self,
            pk:int,
            ):
        """
        Args:
            pk (int): phonopy pk
        """
        node = load_node(pk)
        check_process_class(node, 'PhonopyWorkChain')

        self._node = node
        self._pk = pk
        self._unitcell = get_cell_from_aiida(
                load_node(node.inputs.structure.pk))
        self._phonon_settings = node.inputs.phonon_settings.get_dict()
        self._phonon_setting_info = node.outputs.phonon_setting_info.get_dict()
        self._force_sets = node.outputs.force_sets.get_array('force_sets')

    @property
    def node(self):
        """
        Get PhonopyWorkChain node.
        """
        return self._node

    @property
    def pk(self):
        """
        PhonopyWorkChain pk.
        """
        return self._pk

    @property
    def unitcell(self):
        """
        Input unitcell.
        """
        return self._unitcell

    @property
    def phonon_settings(self):
        """
        Input phonon settings.
        """
        return self._phonon_settings

    @property
    def phonon_setting_info(self):
        """
        Output phonon setting_info.
        """
        return self._phonon_setting_info

    @property
    def force_sets(self):
        """
        Output force sets.
        """
        return self._force_sets

    def get_pks(self) -> dict:
        """
        Get pks.

        Returns:
            dict: containing relax pk and structure pk
        """
        return {
                 'phonopy_pk': self._pk,
                 'input_structure_pk': self._node.inputs.structure.pk,
               }

    def get_phonon(self) -> Phonopy:
        """
        Get phonopy object.

        Returns:
            Phonopy: phonopy object
        """
        phonon = Phonopy(
                get_phonopy_structure(self._unitcell),
                supercell_matrix=self._phonon_setting_info['supercell_matrix'],
                primitive_matrix=self._phonon_setting_info['primitive_matrix'])
        phonon.set_displacement_dataset(
                self._phonon_setting_info['displacement_dataset'])
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
            pk:int,
            ):
        """
        Args:
            relax_pk (int): relax pk
        """
        node = load_node(pk)
        check_process_class(node, 'ShearWorkChain')

        self._node = node
        self._pk = pk
        self._shear_conf = node.inputs.shear_conf.get_dict()
        self._shear_ratios = \
            node.called[-1].outputs.shear_settings.get_dict()['shear_ratios']
        self._hexagonal_cell = get_cell_from_aiida(node.inputs.structure)
        self._gamma = node.outputs.gamma.value
        self._shear_original_cells = None
        self._shear_primitive_cells = None
        self._shear_relax_cells = None
        self._relaxes = None
        self._set_relaxes()
        self._phonon_pks = None

    @property
    def node(self):
        """
        ShearWorkChain node.
        """
        return self._node

    @property
    def pk(self):
        """
        ShearWorkChain pk.
        """
        return self._pk

    @property
    def shear_conf(self):
        """
        Input shear conf
        """
        return self._shear_conf

    @property
    def shear_ratios(self):
        """
        Output shear ratios
        """
        return self._shear_ratios

    @property
    def hexagonal_cell(self):
        """
        Input hexagonal cell
        """
        return self._hexagonal_cell

    @property
    def gamma(self):
        """
        Output gamma
        """
        return self._gamma

    def _set_relaxes(self):
        """
        Set relax in ShearWorkChain.
        """
        relax_pks = get_pks(pk=self._pk, workflow_name='vasp.relax')
        self._relaxes = [ RelaxWorkChain(pk=pk) for pk in relax_pks ]

    @property
    def relaxes(self):
        """
        Output relaxes in ShearWorkChain.
        """
        return self._relaxes

    def _set_phonon_pks(self):
        """
        Set phonon_pks in ShearWorkChain.
        """
        self._phonon_pks = get_pks(pk=self._pk, workflow='phonon')

    @property
    def phonon_pks(self):
        """
        Output phonon_pks in ShearWorkChain.
        """
        return self._phonon_pks

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
        get ShearAnalyzer class object
        """
        input_cells = [
            get_cell_from_aiida(load_node(relax_pk).inputs.structure)
            for relax_pk in self._relax_pks ]
        relax_cells = [
            get_cell_from_aiida(load_node(relax_pk).outputs.relax__structure)
            for relax_pk in self._relax_pks ]
        phonons = [
            PhonopyWorkChain(phonon_pk).get_phonon()
            for phonon_pk in self._phonon_pks ]
        analyzer = ShearAnalyzer(
            structure_type=self._shear_conf['structure_type'],
            orig_cells=self._get_original_cells(),
            input_cells=input_cells)
        analyzer.set_relax_cells(relax_cells)
        analyzer.set_phonons(phonons)
        return analyzer

    # def _get_original_cells(self):
    #     """
    #     get original cell before standardize
    #     """
    #     orig_cells = []
    #     for ratio in self._shear_ratios:
    #         twinpy = get_twinpy_from_cell(
    #                 cell=self._input_cell,
    #                 twinmode=self._shear_conf['twinmode'])
    #         twinpy.set_shear(
    #             xshift=self._shear_conf['xshift'],
    #             yshift=self._shear_conf['yshift'],
    #             dim=[1,1,1],
    #             shear_strain_ratio=ratio,
    #             )
    #         # orig_cells.append(twinpy.shear.get_structure_for_export())
    #         orig_cells.append(twinpy.shear.get_base_primitive_cell(ratio))
    #     return orig_cells
