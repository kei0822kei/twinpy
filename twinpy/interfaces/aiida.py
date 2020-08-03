#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aiida interface for twinpy.
"""
import numpy as np
from twinpy.analysis.shear_analyzer import ShearAnalyzer
from twinpy.interfaces.phonopy import get_phonopy_structure
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.common import NotExistentAttributeError
from aiida.orm import (load_node,
                       Node,
                       QueryBuilder,
                       StructureData,
                       CalcFunctionNode)
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


def get_workflow_pks(pk:int,
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
    Shear work chain class.
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
        is_phonon = node.inputs.is_phonon.value

        self._shear_conf = node.inputs.shear_conf.get_dict()
        self._shear_ratios = \
            node.called[-1].outputs.shear_settings.get_dict()['shear_ratios']
        self._hexagonal_cell = get_cell_from_aiida(node.inputs.structure)
        self._gamma = node.outputs.gamma.value

        self._node = node
        self._pk = pk

        self._create_shears_pk = None
        self._original_cell_pks = None
        self._original_cells = None
        self._set_shear_structures()

        self._relax_pks = None
        self._relaxes = None
        self._set_relaxes()

        self._phonon_pks = None
        self._phonons = None
        if is_phonon:
            self._set_phonons()

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
    def original_cells(self):
        """
        Output shear original cells
        """
        return self._original_cells

    @property
    def gamma(self):
        """
        Output gamma
        """
        return self._gamma

    def _set_shear_structures(self):
        """
        Set original cells in ShearWorkChain.
        """
        qb = QueryBuilder()
        qb.append(Node, filters={'id':{'==': self._pk}}, tag='wf')
        qb.append(CalcFunctionNode,
                  filters={'label':{'==': 'get_shear_structures'}},
                  with_incoming='wf',
                  project=['id'])
        create_shears_pk = qb.all()[0][0]

        qb = QueryBuilder()
        qb.append(Node, filters={'id':{'==': create_shears_pk}}, tag='cs')
        qb.append(StructureData,
                  with_incoming='cs',
                  project=['id', 'label'])
        structs = qb.all()
        orig_cell_pks = [ struct[0] for struct in structs
                                    if 'shear_orig' in struct[1] ]
        orig_cell_pks.sort(key=lambda x: x)

        self._create_shears_pk = create_shears_pk
        self._original_cell_pks = orig_cell_pks
        self._original_cells = [ get_cell_from_aiida(load_node(pk))
                                     for pk in self._original_cell_pks ]

    def _set_relaxes(self):
        """
        Set relax in ShearWorkChain.
        """
        self._relax_pks = get_workflow_pks(pk=self._pk,
                                           workflow_name='vasp.relax')
        self._relaxes = [ RelaxWorkChain(pk=pk) for pk in self._relax_pks ]

    @property
    def relax_pks(self):
        """
        Output relax pks.
        """
        return self._relax_pks

    @property
    def relaxes(self):
        """
        Output relaxes in ShearWorkChain.
        """
        return self._relaxes

    def _set_phonons(self):
        """
        Set phonon_pks in ShearWorkChain.
        """
        self._phonon_pks = get_workflow_pks(pk=self._pk,
                                            workflow_name='phonopy.phonopy')
        self._phonons = [ PhonopyWorkChain(pk=pk) for pk in self._phonon_pks ]

    @property
    def phonon_pks(self):
        """
        Output phonon pks.
        """
        return self._phonon_pks

    @property
    def phonons(self):
        """
        Output phonons in ShearWorkChain.
        """
        return self._phonons

    def get_analyzer(self) -> ShearAnalyzer:
        """
        Get ShearAnalyzer class object.
        """
        original_cells = self._original_cells
        input_cells = [ relax.initial_cell for relax in self._relaxes ]
        relax_cells = [ relax.final_cell for relax in self._relaxes ]
        analyzer = ShearAnalyzer(original_cells=original_cells,
                                 input_cells=input_cells,
                                 relax_cells=relax_cells)
        phns = [ phonon_wf.get_phonon() for phonon_wf in self._phonons ]
        analyzer.set_phonons(phonons=phns)
        return analyzer

    def get_pks(self) -> dict:
        """
        Get pks.

        Returns:
            dict: keys and corresponding pks
        """
        pks = {
                'shear_pk': self._pk,
                'get_shear_structures_pk': self._create_shears_pk,
                'original_cell_pks': self._original_cell_pks,
                'relax_pks': self._relax_pks,
                'phonon_pks': self._phonon_pks,
              }
        return pks


@with_dbenv()
class TwinBoudnaryRelaxWorkChain():
    """
    TwinBoundaryRelax work chain class.
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
        check_process_class(node, 'TwinBoundaryRelaxWorkChain')

        self._pk = pk
        self._node = node
        self._relax_times = None
        self._input_structure_pk = None
        self._twinboundary_relax_conf = None
        self._set_input_data()
        self._relax_pks = None
        self._set_relax_pks()
        self._relaxes = None
        self._set_relaxes()

    @property
    def pk(self):
        """
        TwinBoundaryRelaxWorkChain pk.
        """
        return self._pk

    @property
    def node(self):
        """
        TwinBoundaryRelaxWorkChain node.
        """
        return self._node

    @property
    def relax_times(self):
        """
        Relax times.
        """
        return self._relax_times

    @property
    def twinboundary_relax_conf(self):
        """
        TwinBoundary relax conf.
        """
        return self._twinboundary_relax_conf

    def _set_input_data(self):
        """
        Set calculation input data.
        """
        self._relax_times = self._node.inputs.relax_times.value
        self._input_structure_pk = self._node.inputs.structure.pk
        self._twinboundary_relax_conf = \
                self._node.inputs.twinboundary_relax_conf

    def _set_relax_pks(self):
        """
        Set relax pks.
        """
        rlx_qb = QueryBuilder()
        rlx_qb.append(Node, filters={'id':{'==': self._pk}}, tag='wf')
        rlx_qb.append(RELAX_WF, with_incoming='wf', project=['id'])
        relaxes = rlx_qb.all()
        self._relax_pks = [ relax[0] for relax in relaxes ]

    def _set_relaxes(self):
        """
        Set relaxes.
        """
        self._relaxes = [ RelaxWorkChain(load_node(pk))
                              for pk in self._relax_pks ]

    @property
    def relaxes(self):
        """
        TwinBoundary relax conf.
        """
        return self._relaxes

    def get_pks(self) -> dict:
        """
        Get pks.

        Returns:
            dict: various pks
        """
        relax_pks = np.array(self._relax_pks)
        isif2_pks = relax_pks[[ 2*i for i in range(len(self._relax_times)) ]]
        isif7_pks = relax_pks[[ 2*i+1 for i in range(len(self._relax_times)) ]]
        dic = {
                'twinboudnary_relax_pk': self._pk,
                'input_structure_pk': self._input_structure_pk,
                'relax_pks': self._relax_pks,
                'relax_isif2_pks': isif2_pks,
                'relax_isif7_pks': isif7_pks,
                }
        return dic
