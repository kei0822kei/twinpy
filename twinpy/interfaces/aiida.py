#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aiida interface for twinpy.
"""
import numpy as np
from pprint import pprint
import warnings
from matplotlib import pyplot as plt
from twinpy.analysis.shear_analyzer import ShearAnalyzer
from twinpy.interfaces.phonopy import get_phonopy_structure
from twinpy.structure.base import check_same_cells, check_same_cells
from twinpy.structure.diff import get_structure_diff
from twinpy.api_twinpy import get_twinpy_from_cell
from twinpy.common.kpoints import get_mesh_offset_from_direct_lattice
from twinpy.common.utils import print_header
from twinpy.plot.base import line_chart, DEFAULT_COLORS, DEFAULT_MARKERS
# from twinpy.plot.twinboundary import plane_diff
from twinpy.lattice.lattice import Lattice
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

RELAX_WF = WorkflowFactory('vasp.relax')


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
            node:Node,
            ):
        """
        Args:
            node: aiida Node
        """
        self._node = node
        self._process_class = self._node.process_class.get_name()
        self._process_state = self._node.process_state.value
        self._pk = node.pk
        self._label = self._node.label
        self._description = self._node.description
        self._exit_status = self._node.exit_status
        if self._process_state != 'finished':
            warnings.warn("process state: %s" % self._process_state)
        else:
            if self._exit_status != 0:
                warnings.warn(
                    "Warning: exit status was {}".format(self._exit_status))

    @property
    def process_class(self):
        """
        Process class.
        """
        return self._process_class

    @property
    def process_state(self):
        """
        Process state.
        """
        return self._process_state

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

    def _print_common_information(self):
        print_header('About This Node')
        print("process class:%s" % self._process_class)
        print("process state:%s" % self._process_state)
        print("pk:%s" % self._pk)
        print("label:%s" % self._label)
        print("description:%s" % self._description)
        print("exit status:%s" % self._exit_status)
        print("\n\n")


class _AiidaVaspWorkChain(_WorkChain):
    """
    Aiida-Vasp base work chain class.
    """

    def __init__(
            self,
            node:Node,
            ):
        """
        Args:
            node: aiida Node
        """
        super().__init__(node=node)
        self._initial_structure_pk = None
        self._initial_cell = None
        self._set_initial_structure()
        self._stress = None
        self._forces = None
        if self._process_state == 'finished':
            self._set_stress()
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

    def get_max_force(self) -> float:
        """
        Get maximum force acting on atoms.
        """
        max_force = float(np.linalg.norm(self._forces, axis=1).max())
        return max_force

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
        total_mesh = mesh[0] * mesh[1] * mesh[2]
        twinpy_kpoints = get_mesh_offset_from_direct_lattice(
                lattice=self._initial_cell[0],
                mesh=mesh)
        kpts = {
                'mesh': mesh,
                'total_mesh': twinpy_kpoints['total_mesh'],
                'offset': offset,
                'reciprocal_lattice': twinpy_kpoints['reciprocal_lattice'],
                'reciprocal_volume': twinpy_kpoints['reciprocal_volume'],
                'reciprocal_abc': twinpy_kpoints['abc'],
                'intervals': twinpy_kpoints['intervals'],
                'include_two_pi': twinpy_kpoints['include_two_pi'],
                }
        if self._exit_status is not None:
            sampling_kpoints = self._node.outputs.kpoints.get_array('kpoints')
            weights = self._node.outputs.kpoints.get_array('weights')
            weights_num = (weights * total_mesh).astype(int)
            kpts['sampling_kpoints'] = sampling_kpoints
            kpts['weights'] = weights_num
        return kpts

    def get_vasp_settings(self) -> dict:
        """
        Get input parameters.

        Returns:
            dict: input parameters
        """
        potcar = {
            'potential_family': self._node.inputs.potential_family.value,
            'potential_mapping': self._node.inputs.potential_mapping.get_dict()
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

    def _print_vasp_results(self):
        """
        Print VASP run results.
        """
        print_header('VASP settings')
        pprint(self.get_vasp_settings())
        print("\n\n")
        print_header("kpoints information")
        pprint(self.get_kpoints_info())
        if self._process_state == 'finished':
            print("\n\n")
            print_header('VASP outputs')
            print("stress")
            pprint(self._stress)
            print("\n")
            print("max force acting on atoms")
            print(str(self.get_max_force())+"\n")


@with_dbenv()
class AiidaVaspWorkChain(_AiidaVaspWorkChain):
    """
    Vasp work chain class.
    """

    def __init__(
            self,
            node:Node,
            ):
        """
        Args:
            node: aiida Node
        """
        process_class = 'VaspWorkChain'
        check_process_class(node, process_class)
        super().__init__(node=node)
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
            warnings.warn("Final structure could not find.\n"
                          "process state:{} (pk={})".format(
                self.process_state, self._node.pk))

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

    def get_description(self):
        """
        Get description.
        """
        self._print_common_information()
        print_header('PKs')
        pprint(self.get_pks())
        print("\n\n")
        self._print_vasp_results()


@with_dbenv()
class AiidaRelaxWorkChain(_AiidaVaspWorkChain):
    """
    Relax work chain class.
    """

    def __init__(
            self,
            node:Node,
            ):
        """
        Args:
            node: aiida Node
        """
        process_class = 'RelaxWorkChain'
        check_process_class(node, process_class)
        super().__init__(node=node)
        self._final_structure_pk = None
        self._final_cell = None
        self._current_final_structure_pk = None
        self._current_final_cell = None
        self._set_final_structure()
        self._additional_relax_pks = []

    def _set_final_structure(self):
        """
        Set final structure.
        """
        try:
            self._final_structure_pk = self._node.outputs.relax__structure.pk
            self._final_cell = get_cell_from_aiida(
                    load_node(self._final_structure_pk))
            self._current_final_structure_pk = self._final_structure_pk
            self._current_final_cell = self._final_cell
        except NotExistentAttributeError:
            warnings.warn("Final structure could not find.\n"
                          "process state:{} (pk={})".format(
                self.process_state, self._node.pk))

            relax_pks, static_pk = self.get_vasp_calculation_pks()
            if relax_pks is None:
                self._current_final_structure_pk = self._initial_structure_pk
                self._current_final_cell = self._initial_cell
            else:
                if static_pk is not None:
                    self._current_final_structure_pk = \
                            load_node(static_pk).inputs.structure.pk
                    self._current_final_cell = get_cell_from_aiida(
                            load_node(static_pk).inputs.structure)
                else:
                    aiida_vasp = AiidaVaspWorkChain(load_node(relax_pks[-1]))
                    self._current_final_structure_pk = \
                            aiida_vasp._initial_structure_pk
                    self._current_final_cell = aiida_vasp._initial_cell

    @property
    def final_cell(self):
        """
        Final cell.
        """
        return self._final_cell

    def set_additional_relax(self, aiida_relax_pks:list):
        """
        Set additional_relax_pks in the case final structure of
        RelaxWorkChain is further relaxed.

        Args:
            aiida_relaxes (list): list of relax pks

        Raises:
            RuntimeError: Input node is not RelaxWorkChain
            RuntimeError: Output structure and next input structue does not match.
        """
        previous_rlx = self.pk
        structure_pk = self._final_structure_pk
        aiida_relaxes = [ load_node(relax_pk) for relax_pk in aiida_relax_pks ]
        for aiida_relax in aiida_relaxes:
            if aiida_relax.process_class.get_name() != 'RelaxWorkChain':
                raise RuntimeError(
                        "Input node (pk={}) is not RelaxWorkChain".format(
                            aiida_relax.pk))
            if structure_pk == aiida_relax.inputs.structure.pk:
                if aiida_relax.process_state.value == 'finished':
                    structure_pk = aiida_relax.outputs.relax__structure.pk
                else:
                    warnings.warn("RelaxWorkChain (pk={}) state is {}".format(
                        aiida_relax.pk, aiida_relax.process_state.value))
                previous_rlx = aiida_relax.pk
            else:
                print("previous relax: pk={}".format(previous_rlx))
                print("next relax: pk={}".format(aiida_relax.pk))
                raise RuntimeError("Output structure and next input structure "
                                   "does not match.")
        self._additional_relax_pks = aiida_relax_pks

    @property
    def additional_relax_pks(self):
        """
        Additional relax pks.
        """
        return self._additional_relax_pks

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
            relax_pks = [ pk[0] for pk in vasp_pks ]
        return (relax_pks, static_pk)

    def get_vasp_calculation_nodes(self) -> tuple:
        """
        Get VaspWorkChain nodes.

        Returns:
            tuple: (relax_calcs, static_calc)
        """
        relax_pks, static_pk = self.get_vasp_calculation_pks()
        if self._exit_status == 0:
            relax_nodes = [ AiidaVaspWorkChain(load_node(pk))
                                for pk in relax_pks ]
            static_nodes = AiidaVaspWorkChain(load_node(static_pk))
        else:
            relax_nodes = [ AiidaVaspWorkChain(load_node(pk))
                                for pk in relax_pks[:-1] ]
            static_nodes = None
        return (relax_nodes, static_nodes)

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
                 'current_final_structure_pk': self._current_final_structure_pk,
                 'vasp_relax_pks': relax_pks,
                 'static_pk': static_pk,
               }

    def get_description(self):
        """
        Get description.
        """
        self._print_common_information()
        print_header('PKs')
        pprint(self.get_pks())
        print("\n\n")
        print_header("relax settings")
        pprint(self.get_relax_settings())
        print("\n\n")
        self._print_vasp_results()

    def _get_dic_for_plot_convergence(self) -> dict:
        """
        Get dictionary for plot convergence
        """
        relax_nodes, _ = self.get_vasp_calculation_nodes()
        dic = {}
        dic['max_force'] = np.array(
                [ node.get_max_force() for node in relax_nodes ])
        dic['stress'] = np.array([ node.stress.flatten()[[0,4,8,5,6,1]]
                              for node in relax_nodes ])  # xx yy zz yz zx xy
        dic['energy'] = np.array(
                [ node.get_misc()['total_energies']['energy_no_entropy']
                      for node in relax_nodes ])
        dic['abc'] = np.array([ Lattice(node.final_cell[0]).abc
                           for node in relax_nodes ])
        dic['steps'] = np.array([ i+1 for i in range(len(relax_nodes)) ])
        return dic

    def plot_convergence(self):
        """
        Plot convergence.
        """
        plt.rcParams["font.size"] = 14
        aiida_relax_pks = [ self.get_pks()['relax_pk'] ]
        aiida_relax_pks.extend(self._additional_relax_pks)

        fig = plt.figure(figsize=(16,13))
        ax1 = fig.add_axes((0.15, 0.1, 0.35,  0.35))
        ax2 = fig.add_axes((0.63, 0.1, 0.35, 0.35))
        ax3 = fig.add_axes((0.15, 0.55, 0.35, 0.35))
        ax4 = fig.add_axes((0.63, 0.55, 0.35, 0.35))

        x_val_fix = 0
        for j, rlx_pk in enumerate(aiida_relax_pks):
            aiida_relax = AiidaRelaxWorkChain(load_node(rlx_pk))
            dic = aiida_relax._get_dic_for_plot_convergence()
            dic['steps'] = dic['steps'] + x_val_fix
            static_x_val = dic['steps'][-1] + 0.1
            x_val_fix = dic['steps'][-1]

            line_chart(
                    ax1,
                    dic['steps'],
                    dic['max_force'],
                    'relax times',
                    'max force',
                    c=DEFAULT_COLORS[0],
                    marker=DEFAULT_MARKERS[0],
                    facecolor='white')
            ax1.set_ylim((0, None))
            line_chart(
                    ax2,
                    dic['steps'],
                    dic['energy'],
                    'relax times',
                    'energy',
                    c=DEFAULT_COLORS[0],
                    marker=DEFAULT_MARKERS[0],
                    facecolor='white')
            stress_labels = ['xx', 'yy', 'zz', 'yz', 'zx', 'xy']
            for i in range(6):
                if j == 0:
                    label = stress_labels[i]
                else:
                    label = None
                line_chart(
                        ax3,
                        dic['steps'],
                        dic['stress'][:,i],
                        'relax times',
                        'stress',
                        c=DEFAULT_COLORS[i],
                        marker=DEFAULT_MARKERS[i],
                        facecolor='white',
                        label=label)
            ax3.legend(loc='lower right')
            abc_labels = ['a', 'b', 'c']
            for i in range(3):
                if j == 0:
                    label = abc_labels[i]
                else:
                    label = None
                line_chart(
                        ax4,
                        dic['steps'],
                        dic['abc'][:,i],
                        'relax times',
                        'lattice abc',
                        c=DEFAULT_COLORS[i],
                        marker=DEFAULT_MARKERS[i],
                        facecolor='white',
                        label=label)
            ax4.legend(loc='lower right')
            ax4.set_ylim((0, None))

            if aiida_relax._exit_status == 0:
                ax1.scatter(static_x_val, aiida_relax.get_max_force(),
                            c=DEFAULT_COLORS[0], marker='*', s=150)
                ax2.scatter(static_x_val,
                            aiida_relax.get_misc()\
                                    ['total_energies']['energy_no_entropy'],
                            c=DEFAULT_COLORS[0], marker='*', s=150)
                rlx_stress = aiida_relax.stress.flatten()[[0,4,8,5,6,1]]
                for i in range(6):
                    ax3.scatter(static_x_val, rlx_stress[i], c=DEFAULT_COLORS[i],
                                marker='*', s=150)
                for i in range(3):
                    rlx_abc = Lattice(aiida_relax.final_cell[0]).abc
                    ax4.scatter(static_x_val, rlx_abc[i], c=DEFAULT_COLORS[i],
                                marker='*', s=150)
            else:
                strings = "WARNING: RelaxWorkChain pk={} is still working"\
                        .format(aiida_relax._pk)
                print('+'*len(strings))
                print(strings)
                print('+'*len(strings))


@with_dbenv()
class AiidaPhonopyWorkChain(_WorkChain):
    """
    Phononpy work chain class.
    """
    def __init__(
            self,
            node:Node,
            ):
        """
        Args:
            node: aiida Node
        """
        process_class = 'PhonopyWorkChain'
        check_process_class(node, process_class)
        super().__init__(node=node)
        self._unitcell = get_cell_from_aiida(
                load_node(node.inputs.structure.pk))
        self._phonon_settings = node.inputs.phonon_settings.get_dict()
        self._phonon_setting_info = node.outputs.phonon_setting_info.get_dict()
        self._unitcell = None
        self._primitive = None
        self._supercell = None
        self._structure_pks = None
        self._set_structures()
        self._force_sets = None
        self._set_force_sets()

    def _set_structures(self):
        """
        Set structures.
        """
        self._unitcell = get_cell_from_aiida(self._node.inputs.structure)
        self._primitive = get_cell_from_aiida(self._node.outputs.primitive)
        self._supercell = get_cell_from_aiida(self._node.outputs.supercell)
        self._structure_pks = {
                'unitcell_pk': self._node.inputs.structure.pk,
                'primitive_pk': self._node.outputs.primitive.pk,
                'supercell_pk': self._node.outputs.supercell.pk,
                }

    @property
    def unitcell(self):
        """
        Input unitcell.
        """
        return self._unitcell

    @property
    def primitive(self):
        """
        Output primitive cell.
        """
        return self._primitive

    @property
    def supercell(self):
        """
        Output supercell.
        """
        return self._supercell

    @property
    def structure_pks(self):
        """
        Structure pks.
        """
        return self._structure_pks

    def _set_force_sets(self):
        """
        Set force_sets.
        """
        try:
            self._force_sets = \
                    self._node.outputs.force_sets.get_array('force_sets')
        except NotExistentAttributeError:
            warnings.warn("Could not find force sets. Probably, "
                          "this job still running or finished improperly.\n"
                          "process state: {} (pk={})".format(
                              self._process_state, self._pk))

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
        pks = self._structure_pks.copy()
        pks.update({'phonopy_pk': self._pk})
        return pks

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

    def export_phonon(self, filename:str=None):
        """
        Export phonopy object to yaml file.

        Args:
            filename (str): Output filename. If None, filename becomes
                            pk<number>_phonopy.yaml.
        """
        phonon = self.get_phonon()
        if filename is None:
            filename = 'pk%d_phonopy.yaml' % self._pk
        phonon.save(filename)

    def get_description(self):
        """
        Get description.
        """
        self._print_common_information()
        print_header('PKs')
        pprint(self.get_pks())
        print("\n\n")
        print_header('phonopy settings')
        pprint(self._phonon_settings)


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
class AiidaTwinBoudnaryRelaxWorkChain(_WorkChain):
    """
    TwinBoundaryRelax work chain class.
    """

    def __init__(
            self,
            node:Node,
            ):
        """
        Args:
            relax_pk (int): relax pk
        """
        process_class = 'TwinBoundaryRelaxWorkChain'
        check_process_class(node, process_class)
        super().__init__(node=node)
        self._twinboundary_settings = None
        self._structures = None
        self._structure_pks = None
        self._set_twinboundary()
        self._twinpy = None
        self._standardize = None
        self._set_twinpy()
        self._check_structures()
        self._additional_relax_pks = []

    def _set_twinboundary(self):
        """
        Set twinboundary from vasp.
        """
        parameters = self._node.called[-1].outputs.parameters.get_dict()
        aiida_hexagonal = self._node.inputs.structure
        aiida_tb = self._node.called[-1].outputs.twinboundary
        aiida_tb_original = self._node.called[-1].outputs.twinboundary_orig
        aiida_tb_relax = self._node.called[-2].outputs.relax__structure
        tb = get_cell_from_aiida(aiida_tb)
        tb_original = get_cell_from_aiida(aiida_tb_original)
        tb_relax = get_cell_from_aiida(aiida_tb_relax)

        round_cells = []
        for cell in [ tb, tb_original, tb_relax ]:
            round_lattice = np.round(cell[0], decimals=8)
            round_lattice = cell[0]
            round_atoms = np.round(cell[1], decimals=8) % 1
            round_cell = (round_lattice, round_atoms, cell[2])
            round_cells.append(round_cell)

        self._twinboundary_settings = parameters
        self._structures = {
                'hexagonal': get_cell_from_aiida(aiida_hexagonal),
                'twinboundary': round_cells[0],
                'twinboundary_original': round_cells[1],
                'twinboundary_relax': round_cells[2],
                }
        self._structure_pks = {
                'hexagonal_pk': aiida_hexagonal.pk,
                'twinboundary_pk': aiida_tb.pk,
                'twinboundary_original_pk': aiida_tb_original.pk,
                'twinboundary_relax_pk': aiida_tb_relax.pk,
                }

    @property
    def twinboundary_settings(self):
        """
        Twinboundary settings.
        """
        return self._twinboundary_settings

    @property
    def structures(self):
        """
        Twinboundary structures
        """
        return self._structures

    @property
    def structure_pks(self):
        """
        Twinboundary structure pks
        """
        return self._structure_pks

    def __get_relax_twinboudnary_original_frame(self, rlx_cell, std):
        """
        Todo:
            Future replace convert_to_original_frame in standardize.py
        """
        M_bar_p = rlx_cell[0].T
        x_p = rlx_cell[1].T
        P_c = std.conventional_to_primitive_matrix
        R = std.rotation_matrix
        P = std.transformation_matrix
        p = std.origin_shift.reshape(3,1)

        M_p = np.dot(np.linalg.inv(R), M_bar_p)
        M_s = np.dot(M_p, np.linalg.inv(P_c))
        M = np.dot(M_s, P)
        x_s = np.dot(P_c, x_p)
        x = np.round(np.dot(np.linalg.inv(P), x_s)
                       - np.dot(np.linalg.inv(P), p), decimals=8) % 1
        cell = (M.T, x.T, rlx_cell[2])
        return cell

    def _set_twinpy(self):
        """
        Set twinpy structure object and standardize object.
        """
        params = self._twinboundary_settings
        cell = get_cell_from_aiida(
                load_node(self._structure_pks['hexagonal_pk']))
        twinpy = get_twinpy_from_cell(
                cell=cell,
                twinmode=params['twinmode'])
        twinpy.set_twinboundary(
                layers=params['layers'],
                delta=params['delta'],
                twintype=params['twintype'],
                xshift=params['xshift'],
                yshift=params['yshift'],
                shear_strain_ratio=params['shear_strain_ratio'],
                )
        std = twinpy.get_twinboundary_standardize(
                get_lattice=params['get_lattice'],
                move_atoms_into_unitcell=params['move_atoms_into_unitcell'],
                )
        tb_relax_orig = self.__get_relax_twinboudnary_original_frame(
                rlx_cell=self._structures['twinboundary_relax'],
                std=std,
                )
        self._structures['twinboundary_relax_original'] = tb_relax_orig
        self._twinpy = twinpy
        self._standardize = std

    @property
    def twinpy(self):
        """
        Twinpy structure object.
        """
        return self._twinpy

    @property
    def standardize(self):
        """
        Stadardize object of twinpy original cell.
        """
        return self._standardize

    def _check_structures(self):
        """
        Check structures by reconstucting twinboundary.
        """
        params = self._twinboundary_settings
        tb_orig_cell = self._standardize.cell
        tb_std_cell = self._standardize.get_standardized_cell(
                to_primitive=params['to_primitive'],
                no_idealize=params['no_idealize'],
                symprec=params['symprec'],
                no_sort=params['no_sort'],
                get_sort_list=params['get_sort_list'],
                )
        check_same_cells(first_cell=self._structures['twinboundary_original'],
                         second_cell=tb_orig_cell)
        check_same_cells(first_cell=self._structures['twinboundary'],
                         second_cell=tb_std_cell)
        np.testing.assert_allclose(
                self._structures['twinboundary_original'][0],
                self._structures['twinboundary_relax_original'][0],
                atol=1e-6)

    def set_additional_relax(self, aiida_relax_pks:list):
        """
        Set additional_relax_pks in the case final structure of
        TwinBoundaryRelax WorkChain is further relaxed.

        Args:
            aiida_relaxes (list): list of relax pks

        Raises:
            RuntimeError: Input node is not RelaxWorkChain
            RuntimeError: Output structure and next input structue does not match.
        """
        previous_rlx = self.get_pks()['relax_pk']
        structure_pk = self._structure_pks['twinboundary_relax_pk']
        aiida_relaxes = [ load_node(relax_pk) for relax_pk in aiida_relax_pks ]
        for aiida_relax in aiida_relaxes:
            if aiida_relax.process_class.get_name() != 'RelaxWorkChain':
                raise RuntimeError(
                        "Input node (pk={}) is not RelaxWorkChain".format(
                            aiida_relax.pk))
            if structure_pk == aiida_relax.inputs.structure.pk:
                if aiida_relax.process_state.value == 'finished':
                    structure_pk = aiida_relax.outputs.relax__structure.pk
                else:
                    warnings.warn("RelaxWorkChain (pk={}) state is {}".format(
                        aiida_relax.pk, aiida_relax.process_state.value))
                previous_rlx = aiida_relax.pk
            else:
                print("previous relax: pk={}".format(previous_rlx))
                print("next relax: pk={}".format(aiida_relax.pk))
                raise RuntimeError("Output structure and next input structure "
                                   "does not match.")
        self._additional_relax_pks = aiida_relax_pks

    @property
    def additional_relax_pks(self):
        """
        Additional relax pks.
        """
        return self._additional_relax_pks

    def get_pks(self):
        """
        Get pks.
        """
        relax_pk = self._node.called[-2].pk
        pks = self._structure_pks.copy()
        pks['relax_pk'] = relax_pk
        return pks

    def _get_additional_relax_final_cell(self):
        """
        Get additional relax final structure.
        """
        if self._additional_relax_pks == []:
            raise RuntimeError("additional_relax_pks is not set.")
        else:
            aiida_relax = AiidaRelaxWorkChain(
                    load_node(self._additional_relax_pks[-1]))
            final_cell = get_cell_from_aiida(load_node(
                aiida_relax.get_pks()['current_final_structure_pk']))
        return final_cell

    def get_diff(self, get_additional_relax:bool=False) -> dict:
        """
        Get diff between vasp input and output twinboundary structure.

        Args:
            get_additional_relax (bool): if True, output twinboundary structure
                                         becomes the final structure of
                                         additional_relax

        Returns:
            dict: diff between vasp input and output twinboundary structure

        Raises:
            AssertionError: lattice matrix is not identical
        """
        if get_additional_relax:
            final_cell = self._get_additional_relax_final_cell()
        else:
            final_cell = self._structures['twinboundary_relax']
        cells = (self._structures['twinboundary'],
                 final_cell)
        diff = get_structure_diff(cells=cells,
                                  base_index=0,
                                  include_base=False)
        if not get_additional_relax:
            np.testing.assert_allclose(
                    diff['lattice_diffs'][0],
                    np.zeros((3,3)),
                    atol=1e-8,
                    err_msg="lattice matrix is not identical")
        return diff

    def get_planes_angles(self,
                          is_fractional:bool=False,
                          get_additional_relax:bool=False) -> dict:
        """
        Get plane coords from lower plane to upper plane.
        Return list of z coordinates of original cell frame.

        Args:
            is_fractional (bool): if True, return with fractional coordinate
            get_additional_relax (bool): if True, output twinboundary structure
                                         becomes the final structure of
                                         additional_relax

        Returns:
            dict: plane coords of input and output twinboundary structures.
        """
        if not is_fractional and get_additional_relax:
            raise RuntimeError(
                    "is_fractional=False and get_additional_relax=True "
                    "is not allowed because in the case "
                    "get_additional_relax=True, c norm changes.")


        if get_additional_relax:
            final_cell = self._get_additional_relax_final_cell()
            orig_final_cell = \
                    self.__get_relax_twinboudnary_original_frame(
                            rlx_cell=final_cell,
                            std=self._standardize)
        else:
            orig_final_cell = self._structures['twinboundary_relax_original']
        cells = [self._structures['twinboundary_original'],
                 orig_final_cell]

        coords_list = []
        angles_list = []
        for cell in cells:
            lattice = Lattice(cell[0])
            atoms = cell[1]
            c_norm = lattice.abc[2]
            k_1 = np.array([0,0,1])
            sort_atoms = atoms[np.argsort(atoms[:,2])]

            # planes
            coords = [ lattice.dot(np.array(atom), k_1) / c_norm
                           for atom in sort_atoms ]
            num = len(coords)
            ave_coords = np.sum(
                    np.array(coords).reshape(int(num/2), 2), axis=1) / 2

            epsilon = 1e-9    # to remove '-0.0'
            if is_fractional:
                d = np.round(ave_coords+epsilon, decimals=8) / c_norm
            else:
                d = np.round(ave_coords+epsilon, decimals=8)
            coords_list.append(list(d))

            # angles
            sub_coords_orig = sort_atoms[[i for i in range(1,num,2)]] \
                                  - sort_atoms[[i for i in range(0,num,2)]]
            sub_coords_plus = sort_atoms[[i for i in range(1,num,2)]]+np.array([0,1,0]) \
                                  - sort_atoms[[i for i in range(0,num,2)]]
            sub_coords_minus = sort_atoms[[i for i in range(1,num,2)]]-np.array([0,1,0]) \
                                  - sort_atoms[[i for i in range(0,num,2)]]
            coords = []
            for i in range(len(sub_coords_orig)):
                norm_orig = lattice.get_norm(sub_coords_orig[i])
                norm_plus = lattice.get_norm(sub_coords_plus[i])
                norm_minus = lattice.get_norm(sub_coords_minus[i])
                norms = [norm_orig, norm_plus, norm_minus]
                if min(norms) == norm_orig:
                    coords.append(sub_coords_orig[i])
                elif min(norms) == norm_plus:
                    coords.append(sub_coords_plus[i])
                else:
                    coords.append(sub_coords_minus[i])

            angles = [ lattice.get_angle(frac_coord_first=coord,
                                         frac_coord_second=np.array([0,1,0]),
                                         get_acute=True)
                       for coord in coords ]
            angles_list.append(np.array(angles))

        return {
                'planes': {'before': coords_list[0],
                           'relax': coords_list[1]},
                'angles': {'before': angles_list[0],
                           'relax': angles_list[1]},
                }

    def get_distances(self,
                      is_fractional:bool=False,
                      get_additional_relax:bool=False) -> dict:
        """
        Get distances from the result of 'get_planes'.

        Args:
            is_fractional (bool): if True, return with fractional coordinate
            get_additional_relax (bool): if True, output twinboundary structure
                                         becomes the final structure of
                                         additional_relax

        Returns:
            dict: distances between planes of input and output
                  twinboundary structures.
        """
        planes = self.get_planes_angles(is_fractional=is_fractional,
                                        get_additional_relax=get_additional_relax)['planes']
        lattice = Lattice(self._structures['twinboundary_original'][0])
        c_norm = lattice.abc[2]
        keys = ['before', 'relax']
        dic = {}
        for key in keys:
            coords = planes[key]
            if is_fractional:
                coords.append(1.)
            else:
                coords.append(c_norm)
            distances = [ coords[i+1] - coords[i]
                              for i in range(len(coords)-1) ]
            dic[key] = distances
        return dic

    def get_formation_energy(self, bulk_relax_pk:int):
        """
        Get formation energy. Unit [mJ/m^(-2)]

        Args:
            bulk_relax_pk: relax pk of bulk relax calculation
        """
        def __get_excess_energy():
            rlx_pks = [ self.get_pks()['relax_pk'], bulk_relax_pk ]
            energies = []
            natoms = []
            for i, rlx_pk in enumerate(rlx_pks):
                rlx = load_node(rlx_pk)
                cell = get_cell_from_aiida(rlx.inputs.structure)
                if i == 1:
                    check_same_cells(cell, self._structures['hexagonal'])
                energy = rlx.outputs.energies.get_array('energy_no_entropy')[0]
                energies.append(energy)
                natoms.append(len(cell[2]))
            excess_energy = energies[0] - energies[1] * (natoms[0] / natoms[1])
            return excess_energy

        eV = 1.6022 * 10 ** (-16)
        ang = 10 ** (-10)
        unit = eV / ang ** 2
        orig_lattice = self._structures['twinboundary_original'][0]
        A = orig_lattice[0,0] * orig_lattice[1,1]
        excess_energy = __get_excess_energy()
        formation_energy = np.array(excess_energy) / (2*A) * unit
        return formation_energy

    def plot_convergence(self):
        """
        Plot convergence.
        """
        relax_pk = self.get_pks()['relax_pk']
        aiida_relax = AiidaRelaxWorkChain(load_node(relax_pk))
        aiida_relax.set_additional_relax(self._additional_relax_pks)
        aiida_relax.plot_convergence()

    def plot_plane_diff(self,
                        is_fractional:bool=False,
                        is_decorate:bool=True):
        """
        Plot plane diff.

        Args:
            is_fractional (bool): if True, z coords with fractional coordinate
            is_decorate (bool): if True, decorate figure
        """
        def _get_data(get_additional_relax):
            c_norm = self.structures['twinboundary_original'][0][2,2]
            distances = self.get_distances(
                    is_fractional=is_fractional,
                    get_additional_relax=get_additional_relax)
            planes = self.get_planes_angles(
                    is_fractional=is_fractional,
                    get_additional_relax=get_additional_relax)['planes']
            before_distances = distances['before'].copy()
            bulk_interval = before_distances[1]
            rlx_distances = distances['relax'].copy()
            z_coords = planes['before'].copy()
            z_coords.insert(0, z_coords[0] - before_distances[-1])
            z_coords.append(z_coords[-1] + before_distances[0])
            rlx_distances.insert(0, rlx_distances[-1])
            rlx_distances.append(rlx_distances[0])
            ydata = np.array(z_coords) + bulk_interval / 2
            if is_fractional:
                rlx_distances = np.array(rlx_distances) * c_norm
            return (rlx_distances, ydata, z_coords, bulk_interval)

        if self.additional_relax_pks == []:
            datas = [ _get_data(False) ]
            labels = ['isif7']
        else:
            datas = [ _get_data(bl) for bl in [False, True] ]
            labels = ['isif7', 'isif3']

        fig = plt.figure(figsize=(8,13))
        ax = fig.add_subplot(111)
        for i in range(len(datas)):
            xdata, ydata, z_coords, bulk_interval = datas[i]
            line_chart(ax=ax,
                       xdata=xdata,
                       ydata=ydata,
                       xlabel='distance',
                       ylabel='z coords',
                       sort_by='y',
                       label=labels[i])

        if is_decorate:
            num = len(z_coords)
            tb_idx = [1, int(num/2), num-1]
            xmax = max([ max(data[0]) for data in datas ])
            xmin = min([ min(data[0]) for data in datas ])
            ymax = max([ max(data[1]) for data in datas ])
            ymin = min([ min(data[1]) for data in datas ])
            for idx in tb_idx:
                ax.hlines(z_coords[idx],
                          xmin=xmin-0.005,
                          xmax=xmax+0.005,
                          linestyle='--',
                          linewidth=1.5)
            yrange = ymax - ymin
            if is_fractional:
                c_norm = self.structures['twinboundary_original'][0][2,2]
                vline_x = bulk_interval * c_norm
            else:
                vline_x = bulk_interval
            ax.vlines(vline_x,
                      ymin=ymin-yrange*0.01,
                      ymax=ymax+yrange*0.01,
                      linestyle='--',
                      linewidth=0.5)
            ax.legend()

    def plot_angle_diff(self,
                        is_fractional:bool=False,
                        is_decorate:bool=True):
        """
        Plot angle diff.

        Args:
            is_fractional (bool): if True, z coords with fractional coordinate
            is_decorate (bool): if True, decorate figure
        """
        def _get_data(get_additional_relax):
            c_norm = self.structures['twinboundary_original'][0][2,2]
            distances = self.get_distances(
                    is_fractional=is_fractional,
                    get_additional_relax=get_additional_relax)
            planes_angles = self.get_planes_angles(
                    is_fractional=is_fractional,
                    get_additional_relax=get_additional_relax)
            planes = planes_angles['planes']
            before_distances = distances['before'].copy()
            z_coords = planes['before'].copy()
            z_coords.insert(0, z_coords[0] - before_distances[-1])
            z_coords.append(z_coords[-1] + before_distances[0])

            angles = planes_angles['angles']
            rlx_angles = list(angles['relax'])
            bulk_angle = angles['before'][1]
            rlx_angles.insert(0, rlx_angles[-1])
            rlx_angles.append(rlx_angles[1])
            ydata = np.array(z_coords)
            rlx_angles = np.array(rlx_angles)
            return (rlx_angles, ydata, z_coords, bulk_angle)

        if self.additional_relax_pks == []:
            datas = [ _get_data(False) ]
            labels = ['isif7']
        else:
            datas = [ _get_data(bl) for bl in [False, True] ]
            labels = ['isif7', 'isif3']

        fig = plt.figure(figsize=(8,13))
        ax = fig.add_subplot(111)
        for i in range(len(datas)):
            xdata, ydata, z_coords, bulk_angle = datas[i]
            line_chart(ax=ax,
                       xdata=xdata,
                       ydata=ydata,
                       xlabel='angle',
                       ylabel='z coords',
                       sort_by='y',
                       label=labels[i])

        if is_decorate:
            num = len(z_coords)
            tb_idx = [1, int(num/2), num-1]
            xmax = max([ max(data[0]) for data in datas ])
            xmin = min([ min(data[0]) for data in datas ])
            ymax = max([ max(data[1]) for data in datas ])
            ymin = min([ min(data[1]) for data in datas ])
            for idx in tb_idx:
                ax.hlines(z_coords[idx],
                          xmin=xmin-0.005,
                          xmax=xmax+0.005,
                          linestyle='--',
                          linewidth=1.5)
            yrange = ymax - ymin
            vline_x = bulk_angle
            ax.vlines(vline_x,
                      ymin=ymin-yrange*0.01,
                      ymax=ymax+yrange*0.01,
                      linestyle='--',
                      linewidth=0.5)
            ax.legend()

    def get_description(self):
        """
        Get description.
        """
        self._print_common_information()
        print_header('PKs')
        pprint(self.get_pks())
        print("\n\n")
        print_header('twinboudnary settings')
        pprint(self.twinboundary_settings)
