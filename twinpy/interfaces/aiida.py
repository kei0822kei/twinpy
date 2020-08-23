#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aiida interface for twinpy.
"""
import numpy as np
from pprint import pprint
import warnings
from twinpy.analysis.shear_analyzer import ShearAnalyzer
from twinpy.interfaces.phonopy import get_phonopy_structure
from twinpy.structure.base import check_same_cells, check_same_cells
from twinpy.structure.diff import get_structure_diff
from twinpy.api_twinpy import get_twinpy_from_cell
from twinpy.common.kpoints import get_mesh_offset_from_direct_lattice
from twinpy.common.utils import print_header
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
        sampling_kpoints = self._node.outputs.kpoints.get_array('kpoints')
        weights = self._node.outputs.kpoints.get_array('weights')
        total_mesh = mesh[0] * mesh[1] * mesh[2]
        weights_num = (weights * total_mesh).astype(int)
        twinpy_kpoints = get_mesh_offset_from_direct_lattice(
                lattice=self._initial_cell[0],
                mesh=mesh)
        kpts = {
                'mesh': mesh,
                'total_mesh': twinpy_kpoints['total_mesh'],
                'offset': offset,
                'sampling_kpoints': sampling_kpoints,
                'weights': weights_num,
                'reciprocal_lattice': twinpy_kpoints['reciprocal_lattice'],
                'reciprocal_volume': twinpy_kpoints['reciprocal_volume'],
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
        if self._process_state == 'finished':
            print_header("kpoints information")
            pprint(self.get_kpoints_info())
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
        if self._process_state == 'finished':
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
            relax_pks = [ pk[0] for pk in vasp_pks ]
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

    def _set_twinpy(self):
        """
        Set twinpy structure object and standardize object.
        """
        def __set_relax_twinboudnary_original_frame(rlx_cell, std):
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
        tb_relax_orig = __set_relax_twinboudnary_original_frame(
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

    def get_pks(self):
        """
        Get pks.
        """
        relax_pk = self._node.called[-2].pk
        pks = self._structure_pks.copy()
        pks['relax_pk'] = relax_pk
        return pks

    def get_diff(self) -> dict:
        """
        Get diff between vasp input and output twinboundary structure.

        Returns:
            dict: diff between vasp input and output twinboundary structure.

        Raises:
            AssertionError: lattice matrix is not identical
        """
        cells = (self._structures['twinboundary'],
                 self._structures['twinboundary_relax'])
        diff = get_structure_diff(cells=cells,
                                  base_index=0,
                                  include_base=False)
        np.testing.assert_allclose(diff['lattice_diffs'][0],
                                   np.zeros((3,3)),
                                   atol=1e-8,
                                   err_msg="lattice matrix is not identical")
        return diff

    def get_planes(self, is_fractional:bool=False) -> dict:
        """
        Get plane coords from lower plane to upper plane.
        Return list of z coordinates of original cell frame.

        Args:
            is_fractional (bool): if True, return with fractional coordinate

        Returns:
            dict: plane coords of input and output twinboundary structures.
        """
        lattice = Lattice(self._structures['twinboundary_original'][0])
        input_atoms = self._structures['twinboundary_original'][1]
        output_atoms = self._structures['twinboundary_relax_original'][1]
        c_norm = lattice.abc[2]
        coords_list = []
        k_1 = np.array([0,0,1])
        for atoms in [ input_atoms, output_atoms ]:
            sort_atoms = atoms[np.argsort(atoms[:,2])]
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
        return {
                'before': coords_list[0],
                'relax': coords_list[1],
                }

    def get_distances(self, is_fractional:bool=False) -> dict:
        """
        Get distances from the result of 'get_planes'.

        Args:
            is_fractional (bool): if True, return with fractional coordinate

        Returns:
            dict: distances between planes of input and output
                  twinboundary structures.
        """
        lattice = Lattice(self._structures['twinboundary_original'][0])
        c_norm = lattice.abc[2]
        planes = self.get_planes(is_fractional=is_fractional)
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
