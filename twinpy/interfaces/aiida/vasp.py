#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interface for Aiida-Vasp.
"""
import numpy as np
from pprint import pprint
import warnings
from matplotlib import pyplot as plt
from twinpy.interfaces.aiida import (check_process_class,
                                     get_cell_from_aiida,
                                     _WorkChain)
from twinpy.common.kpoints import get_mesh_offset_from_direct_lattice
from twinpy.common.utils import print_header
from twinpy.plot.base import line_chart, DEFAULT_COLORS, DEFAULT_MARKERS
from twinpy.lattice.lattice import Lattice
from twinpy.analysis.relax_analyzer import RelaxAnalyzer
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.common import NotExistentAttributeError
from aiida.orm import (load_node,
                       Node,
                       WorkChainNode,
                       QueryBuilder)


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
        self._energy = None
        if self._process_state == 'finished':
            self._set_properties()

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

    def _set_properties(self):
        """
        Set properties.
        """
        self._forces = self._node.outputs.forces.get_array('final')
        self._stress = self._node.outputs.stress.get_array('final')
        self._energy = self._node.outputs.energies.get_array('energy_no_entropy')[0]

    @property
    def forces(self):
        """
        Forces acting on atoms after relax.
        """
        return self._forces

    @property
    def stress(self):
        """
        Stress acting on lattice after relax.
        """
        return self._stress

    @property
    def energy(self):
        """
        Total energy.
        """
        return self._energy

    def get_max_force(self) -> float:
        """
        Get maximum force acting on atoms.
        """
        max_force = float(np.linalg.norm(self._forces, axis=1).max())
        return max_force

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
            'parser_settings':
                self._node.inputs.settings.get_dict()['parser_settings'],
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
            print("# stress")
            pprint(self._stress)
            print("\n")
            print("# max force acting on atoms")
            print(str(self.get_max_force())+"\n")


@with_dbenv()
class AiidaVaspWorkChain(_AiidaVaspWorkChain):
    """
    Vasp work chain class.
    """

    def __init__(
            self,
            node:Node,
            ignore_warning:bool=False,
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
        self._set_final_structure(ignore_warning=ignore_warning)

    def _set_final_structure(self, ignore_warning):
        """
        Set final structure.
        """
        try:
            self._final_structure_pk = self._node.outputs.structure.pk
            self._final_cell = get_cell_from_aiida(
                    load_node(self._final_structure_pk))
        except NotExistentAttributeError:
            if not ignore_warning:
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

    def get_relax_analyzer(self, original_cell:tuple=None):
        """
        Get RelaxAnalyzer class object.

        Args:
            original_cell (tuple): Original cell whose standardized cell
                                   is initail_cell.

        Returns:
            RelaxAnalyzer: RelaxAnalyzer class object.
        """
        analyzer = RelaxAnalyzer(initial_cell=self._initial_cell,
                                 final_cell=self._final_cell,
                                 original_cell=original_cell,
                                 forces=self._forces,
                                 stress=self._stress,
                                 energy=self._energy)
        return analyzer

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
            ignore_warning:bool=False,
            ):
        """
        Args:
            node: aiida Node

        Todo:
            Fix names. Name 'final_cell' 'final_structure_pk' 'cuurent-'
            are strange because final_cell and final_structure_pk
            the result from first relax pk. Therefore, 'current-'
            is truely final structure pk and final cell.
        """
        process_class = 'RelaxWorkChain'
        check_process_class(node, process_class)
        super().__init__(node=node)
        self._final_structure_pk = None
        self._final_cell = None
        self._current_final_structure_pk = None
        self._current_final_cell = None
        self._set_final_structure(ignore_warning)

    def _set_final_structure(self, ignore_warning):
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
            if not ignore_warning:
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
    def current_final_cell(self):
        """
        Final cell.
        """
        return self._current_final_cell

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
            static_nodes = AiidaVaspWorkChain(load_node(static_pk),
                                              ignore_warning=True)
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
        pks = {
                 'relax_pk': self._pk,
                 'initial_structure_pk': self._initial_structure_pk,
                 'final_structure_pk': self._final_structure_pk,
                 'current_final_structure_pk': self._current_final_structure_pk,
                 'vasp_relax_pks': relax_pks,
                 'static_pk': static_pk,
              }
        return pks

    def get_relax_analyzer(self, original_cell:tuple=None):
        """
        Get RelaxAnalyzer class object.

        Args:
            original_cell (tuple): Original cell whose standardized cell
                                   is initail_cell.

        Returns:
            RelaxAnalyzer: RelaxAnalyzer class object.
        """
        analyzer = RelaxAnalyzer(initial_cell=self._initial_cell,
                                 final_cell=self._final_cell,
                                 original_cell=original_cell,
                                 forces=self._forces,
                                 stress=self._stress,
                                 energy=self._energy)
        return analyzer

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

    # def _get_dic_for_plot_convergence(self) -> dict:
    #     """
    #     Get dictionary for plot convergence
    #     """
    #     relax_nodes, _ = self.get_vasp_calculation_nodes()
    #     dic = {}
    #     dic['max_force'] = np.array(
    #             [ node.get_max_force() for node in relax_nodes ])
    #     # stress xx yy zz yz zx xy
    #     dic['stress'] = np.array([ node.stress.flatten()[[0,4,8,5,6,1]]
    #                                for node in relax_nodes ])
    #     dic['energy'] = np.array(
    #             [ node.get_misc()['total_energies']['energy_no_entropy']
    #                   for node in relax_nodes ])
    #     dic['abc'] = np.array([ Lattice(node.final_cell[0]).abc
    #                             for node in relax_nodes ])
    #     dic['steps'] = np.array([ i+1 for i in range(len(relax_nodes)) ])
    #     return dic

    # def plot_convergence(self):
    #     """
    #     Plot convergence.

    #     Todo:
    #         This function must be moved in plot directory.
    #     """
    #     plt.rcParams["font.size"] = 14
    #     aiida_relax_pks = [ self.get_pks()['relax_pk'] ]
    #     aiida_relax_pks.extend(self._additional_relax_pks)

    #     fig = plt.figure(figsize=(16,13))
    #     ax1 = fig.add_axes((0.15, 0.1, 0.35,  0.35))
    #     ax2 = fig.add_axes((0.63, 0.1, 0.35, 0.35))
    #     ax3 = fig.add_axes((0.15, 0.55, 0.35, 0.35))
    #     ax4 = fig.add_axes((0.63, 0.55, 0.35, 0.35))

    #     x_val_fix = 0
    #     for j, rlx_pk in enumerate(aiida_relax_pks):
    #         aiida_relax = AiidaRelaxWorkChain(load_node(rlx_pk))
    #         dic = aiida_relax._get_dic_for_plot_convergence()
    #         dic['steps'] = dic['steps'] + x_val_fix
    #         static_x_val = dic['steps'][-1] + 0.1
    #         x_val_fix = dic['steps'][-1]

    #         line_chart(
    #                 ax1,
    #                 dic['steps'],
    #                 dic['max_force'],
    #                 'relax times',
    #                 'max force',
    #                 c=DEFAULT_COLORS[0],
    #                 marker=DEFAULT_MARKERS[0],
    #                 facecolor='white')
    #         # ax1.set_ylim((0, None))
    #         ax1.set_yscale('log')
    #         line_chart(
    #                 ax2,
    #                 dic['steps'],
    #                 dic['energy'],
    #                 'relax times',
    #                 'energy',
    #                 c=DEFAULT_COLORS[0],
    #                 marker=DEFAULT_MARKERS[0],
    #                 facecolor='white')
    #         stress_labels = ['xx', 'yy', 'zz', 'yz', 'zx', 'xy']
    #         for i in range(6):
    #             if j == 0:
    #                 label = stress_labels[i]
    #             else:
    #                 label = None
    #             line_chart(
    #                     ax3,
    #                     dic['steps'],
    #                     dic['stress'][:,i],
    #                     'relax times',
    #                     'stress',
    #                     c=DEFAULT_COLORS[i],
    #                     marker=DEFAULT_MARKERS[i],
    #                     facecolor='white',
    #                     label=label)
    #         ax3.legend(loc='lower right')
    #         abc_labels = ['a', 'b', 'c']
    #         for i in range(3):
    #             if j == 0:
    #                 label = abc_labels[i]
    #             else:
    #                 label = None
    #             line_chart(
    #                     ax4,
    #                     dic['steps'],
    #                     dic['abc'][:,i],
    #                     'relax times',
    #                     'lattice abc',
    #                     c=DEFAULT_COLORS[i],
    #                     marker=DEFAULT_MARKERS[i],
    #                     facecolor='white',
    #                     label=label)
    #         ax4.legend(loc='lower right')
    #         ax4.set_ylim((0, None))

    #         if aiida_relax._exit_status == 0:
    #             ax1.scatter(static_x_val, aiida_relax.get_max_force(),
    #                         c=DEFAULT_COLORS[0], marker='*', s=150)
    #             ax2.scatter(static_x_val,
    #                         aiida_relax.get_misc()
    #                                 ['total_energies']['energy_no_entropy'],
    #                         c=DEFAULT_COLORS[0], marker='*', s=150)
    #             rlx_stress = aiida_relax.stress.flatten()[[0,4,8,5,6,1]]
    #             for i in range(6):
    #                 ax3.scatter(static_x_val, rlx_stress[i],
    #                             c=DEFAULT_COLORS[i],
    #                             marker='*', s=150)
    #             for i in range(3):
    #                 rlx_abc = Lattice(aiida_relax.final_cell[0]).abc
    #                 ax4.scatter(static_x_val, rlx_abc[i], c=DEFAULT_COLORS[i],
    #                             marker='*', s=150)
    #         else:
    #             strings = "WARNING: Exit state RelaxWorkChain pk={} is {}."\
    #                     .format(aiida_relax._pk, aiida_relax._exit_status)
    #             print('+'*len(strings))
    #             print(strings)
    #             print("         Skip plotting.")
    #             print('+'*len(strings))


class AiidaRelaxCollection():
    """
    Collection of AiidaRelaxWorkChain.
    """

    def __init__(
           self,
           aiida_relaxes:list,
       ):
        """
        Args:
            aiida_relaxes (list): List of AiidaRelaxWorkChain.
        """
        self._aiida_relaxes = aiida_relaxes
        self._aiida_relax_pks = [ relax.pk for relax in aiida_relaxes ]
        self._initial_structure_pk = None
        self._initial_cell = None
        self._current_final_structure_pk = None
        self._current_final_cell = None
        self._final_structure_pk = None
        self._final_cell = None
        self._set_structures()

    def _set_structures(self):
        """
        Check previous output structure and next input structure
        is same.
        """
        relax_pk = None
        structure_pk = None
        for i, aiida_relax in enumerate(self._aiida_relaxes):
            if i > 0:
                if aiida_relax.get_pks()['initial_structure_pk'] \
                        != structure_pk:
                    raise RuntimeError(
                            "Relax pk {} output structure pk {} "
                            "and relax pk {} input structure pk {} "
                            "does not match.".format(
                                relax_pk,
                                structure_pk,
                                aiida_relax.pk,
                                aiida_relax.get_pks()['initial_structure_pk'],
                                ))
            relax_pk = aiida_relax.pk
            structure_pk = aiida_relax.get_pks()['current_final_structure_pk']

        self._current_final_structure_pk = structure_pk
        self._current_final_cell = \
                get_cell_from_aiida(load_node(structure_pk))

        if self._aiida_relaxes[-1].process_state == 'finished':
            self._final_structure_pk = \
                    self._current_final_structure_pk
            self._final_cell = self._current_final_cell

        self._initial_structure_pk = \
                self._aiida_relaxes[0].get_pks()['initial_structure_pk']
        self._initial_cell = \
                get_cell_from_aiida(load_node(self._initial_structure_pk))

    @property
    def aiida_relaxes(self):
        """
        List of AiidaRelaxWorkChain class object.
        """
        return self._aiida_relaxes

    @property
    def initial_cell(self):
        """
        Initial cell.
        """
        return self._initial_cell

    @property
    def current_final_cell(self):
        """
        Current final cell.
        """
        return self._current_final_cell

    @property
    def final_cell(self):
        """
        Final cell.
        """
        return self._final_cell

    def get_pks(self):
        """
        Get pks.
        """
        pks = {
            'aiida_relax_pks': self._aiida_relax_pks,
            'initial_structure_pk': self._initial_structure_pk,
            'current_final_structure_pk': self._current_final_structure_pk,
            'final_structure_pk': self._final_structure_pk,
            }
        return pks

    def get_relax_analyzer(self, original_cell:tuple=None):
        """
        Get RelaxAnalyzer class object.

        Args:
            original_cell (tuple): Original cell whose standardized cell
                                   is initail_cell.
        """
        pks = self.get_pks()
        if pks['final_structure_pk'] is None:
            warnings.warn("Final structure in Latest RelaxWorkChain (pk={}) "
                           "does not find. So build RelaxAnalyzer with "
                           "previous RelaxWorkChain (pk={}) "
                           " as a final structure.".format(
                               self._additional_relax_pks[-1],
                               self._additional_relax_pks[-2],
                               ))
            final_relax = self._aiida_relaxes[-2]
        else:
            final_relax = self._aiida_relaxes[-1]

        initail_cell = self._initial_cell
        final_cell = final_relax._final_cell
        forces = final_relax.forces
        stress = final_relax.stress
        energy = final_relax.energy
        relax_analyzer = RelaxAnalyzer(
                initial_cell=initail_cell,
                final_cell=final_cell,
                original_cell=original_cell,
                forces=forces,
                stress=stress,
                energy=energy,
                )
        return relax_analyzer
