#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aiida interface for twinpy.
"""
import numpy as np
import warnings
from twinpy.analysis.shear_analyzer import ShearAnalyzer
from twinpy.interfaces.phonopy import get_phonopy_structure
from twinpy.common.kpoints import get_mesh_offset_from_direct_lattice
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.common import NotExistentAttributeError
from aiida.orm import (load_node,
                       Node,
                       QueryBuilder,
                       StructureData,
                       CalcFunctionNode,
                       WorkChainNode)
from aiida.plugins import WorkflowFactory
from aiida.common.exceptions import NotExistentAttributeError
from phonopy import Phonopy

VASP_WF = WorkflowFactory('vasp.vasp')
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


class _WorkChain():
    """
    Aiida WorkChain class.
    """

    def __init__(
            self,
            pk:int,
            ):
        """
        Args:
            pk (int): relax pk
            process_class (str): process class
        """
        node = load_node(pk)
        self._node = node
        self._process_class = self._node.process_class.get_name()
        self._pk = pk
        self._label = self._node.label
        self._description = self._node.description
        self._exit_status = self._node.exit_status
        if self._exit_status != 0:
            warnings.warn("Warning: exit status was %d" % self._exit_status)

    @property
    def process_class(self):
        """
        Process class.
        """
        return self._process_class

    @property
    def node(self):
        """
        WorkChain node.
        """
        return self._node

    @property
    def pk(self):
        """
        WorkChain pk.
        """
        return self._pk

    @property
    def label(self):
        """
        Label.
        """
        return self._label

    @property
    def description(self):
        """
        Description.
        """
        return self._description

    @property
    def exit_status(self):
        """
        Exit status.
        """
        return self._exit_status


class _AiidaVaspWorkChain(_WorkChain):
    """
    Aiida-Vasp base work chain class.
    """

    def __init__(
            self,
            pk:int,
            ):
        """
        Args:
            pk (int): relax pk
            process_class (str): process class
        """
        super().__init__(pk=pk)
        self._initial_structure_pk = None
        self._initial_cell = None
        self._set_initial_structure()
        self._stress = None
        self._set_stress()
        self._forces = None
        self._set_forces()

    def _set_initial_structure(self):
        """
        Set initial structure.
        """
        self._initial_structure_pk = self._node.inputs.structure.pk
        self._initial_cell = get_cell_from_aiida(
                load_node(self._initial_structure_pk))

    @property
    def initial_cell(self):
        """
        Initial cell.
        """
        return self._initial_cell

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

    def get_kpoints_info(self) -> dict:
        """
        Get sampling kpoints information.

        Returns:
            dict: kpoints information
        """
        mesh, offset = self._node.inputs.kpoints.get_kpoints_mesh()
        sampling_kpoints = self._node.outputs.kpoints.get_array('kpoints')
        weights = self._node.outputs.kpoints.get_array('weights')
        total_mesh = mesh[0] * mesh[1] * mesh[2]
        weights_num = (weights * total_mesh).astype(int)
        twinpy_kpoints = get_mesh_offset_from_direct_lattice(
                lattice=self._initial_cell[0],
                mesh=mesh)
        kpts = {
                'mesh': mesh,
                'offset': offset,
                'sampling_kpoints': sampling_kpoints,
                'weights': weights_num,
                'reciprocal_lattice': twinpy_kpoints['reciprocal_lattice'],
                'reciprocal_abc': twinpy_kpoints['abc'],
                'intervals': twinpy_kpoints['intervals'],
                'include_two_pi': twinpy_kpoints['include_two_pi'],
                }
        return kpts

    def get_vasp_settings(self) -> dict:
        """
        Get input parameters.

        Returns:
            dict: input parameters
        """
        potcar = {
          'potential_family': self._node.inputs.potential_family.value,
          'potential_mapping': self._node.inputs.potential_mapping.get_dict(),
          }

        settings = {
                'incar': self._node.inputs.parameters.get_dict(),
                'potcar': potcar,
                'kpoints': self._node.inputs.kpoints.get_kpoints_mesh(),
                }
        return settings

    def get_misc(self) -> dict:
        """
        Get misc.
        """
        return self._node.outputs.misc.get_dict()


@with_dbenv()
class AiidaVaspWorkChain(_AiidaVaspWorkChain):
    """
    Vasp work chain class.
    """

    def __init__(
            self,
            pk:int,
            ):
        """
        Args:
            pk (int): relax pk
        """
        process_class = 'VaspWorkChain'
        check_process_class(load_node(pk), process_class)
        super().__init__(pk=pk)
        self._final_structure_pk = None
        self._final_cell = None
        self._set_final_structure()

    def _set_final_structure(self):
        """
        Set final structure.
        """
        try:
            self._final_structure_pk = self._node.outputs.structure.pk
            self._final_cell = get_cell_from_aiida(
                    load_node(self._final_structure_pk))
        except NotExistentAttributeError:
            warnings.warn("Final structure could not find.")

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
            dict: containing vasp pk and structure pk
        """
        return {
                 'vasp_pk': self._pk,
                 'initial_structure_pk': self._initial_structure_pk,
                 'final_structure_pk': self._final_structure_pk,
               }


@with_dbenv()
class AiidaRelaxWorkChain(_AiidaVaspWorkChain):
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
        process_class = 'RelaxWorkChain'
        check_process_class(load_node(pk), process_class)
        super().__init__(pk=pk)
        self._final_structure_pk = None
        self._final_cell = None
        self._set_final_structure()

    def _set_final_structure(self):
        """
        Set final structure.
        """
        try:
            self._final_structure_pk = self._node.outputs.relax__structure.pk
            self._final_cell = get_cell_from_aiida(
                    load_node(self._final_structure_pk))
        except NotExistentAttributeError:
            warnings.warn("Final structure could not find.")

    @property
    def final_cell(self):
        """
        Final cell.
        """
        return self._final_cell

    def get_relax_settings(self) -> dict:
        """
        Get relax settings.

        Returns:
            dict: relax settings
        """
        keys = [ key for key in self._node.inputs._get_keys()
                     if 'relax' in key ]
        settings = {}
        for key in keys:
            name = key.replace('relax__', '')
            settings[name] = self._node.inputs.__getattr__(key).value
        return settings

    def get_vasp_calculation_pks(self) -> tuple:
        """
        Get VaspWorkChain pks.

        Returns:
            tuple: (relax_calcs, static_calc)
        """
        qb = QueryBuilder()
        qb.append(Node, filters={'id':{'==': self._pk}})
        qb.append(WorkChainNode)  # extract vasp.verify WorkChainNodes
        qb.append(WorkChainNode,
                  project=['id'])  # extract vasp.vasp WorkChainNodes
        qb.order_by({WorkChainNode: {'id': 'asc'}})
        vasp_pks = qb.all()

        relax_pks = None
        static_pk = None
        if 'relax' not in \
                load_node(vasp_pks[-1][0]).inputs.parameters.get_dict().keys():
            static_pk = vasp_pks[-1][0]
            relax_pks = [ pk[0] for pk in vasp_pks[:-1] ]
        else:
            warnings.warn("Could not find final static_pk calculation.")
            relax_pks = vasp_pks
        return (relax_pks, static_pk)

    def get_pks(self) -> dict:
        """
        Get pks.

        Returns:
            dict: containing relax pk and structure pk
        """
        relax_pks, static_pk = self.get_vasp_calculation_pks()
        return {
                 'relax_pk': self._pk,
                 'initial_structure_pk': self._initial_structure_pk,
                 'final_structure_pk': self._final_structure_pk,
                 'vasp_relax_pks': relax_pks,
                 'static_pk': static_pk,
               }


@with_dbenv()
class AiidaPhonopyWorkChain():
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
        process_class = 'PhonopyWorkChain'
        node = load_node(pk)
        check_process_class(node, process_class)

        self._node = node
        self._process_class = process_class
        self._pk = pk
        self._unitcell = get_cell_from_aiida(
                load_node(node.inputs.structure.pk))
        self._phonon_settings = node.inputs.phonon_settings.get_dict()
        self._phonon_setting_info = node.outputs.phonon_setting_info.get_dict()
        self._force_sets = node.outputs.force_sets.get_array('force_sets')

    @property
    def process_class(self):
        """
        Process class.
        """
        return self._process_class

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
class AiidaShearWorkChain():
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
class AiidaTwinBoudnaryRelaxWorkChain():
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



# def get_workflow_pks(pk:int,
#                      workflow_name:str) -> dict:
#     """
#     Get workflow pk in the specified pk.
# 
#     Args:
#         pk (int): input pk
#         workflow_name (str): workflow name such as 'vasp.relax',
#                              'phonopy.phonopy'
# 
#     Returns:
#         dict: workflow pks in input pk
#     """
#     wf = WorkflowFactory(workflow_name)
#     node_qb = QueryBuilder()
#     node_qb.append(Node, filters={'id':{'==': pk}}, tag='wf')
#     node_qb.append(wf, with_incoming='wf', project=['id'])
#     nodes = node_qb.all()
#     node_pks = [ node[0] for node in nodes ]
#     node_pks.reverse()
#     return node_pks
