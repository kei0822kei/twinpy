#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interface for Aiida-Vasp.
"""
import numpy as np
from pprint import pprint
import warnings
from matplotlib import pyplot as plt
from twinpy.interfaces.aiida.base import (check_process_class,
                                          get_cell_from_aiida,
                                          _WorkChain)
from twinpy.common.kpoints import get_mesh_offset_from_direct_lattice
from twinpy.common.utils import print_header
from twinpy.plot.relax import RelaxPlot
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
            node: Aiida Node.
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
        self._energy = \
                self._node.outputs.energies.get_array('energy_no_entropy')[0]

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
            dict: Contains kpoints information.
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
            dict: Contains input parameters.
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
            node: Aiida Node.
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
            dict: Contains vasp pk and structure pk.
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

    def get_outputs(self):
        """
        Get outputs.
        """
        dic = self._get_outputs()
        lattice = Lattice(lattice=self._final_cell[0])
        dic['abc'] = lattice.abc

        return dic

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
            are strange because final_cell and final_structure_pk are
            the results from first relax pk. Therefore, 'current-'
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
            dict: Contains relax settings.
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
            tuple: (relax_calcs, static_calc).
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

    def get_vasp_calculations(self) -> tuple:
        """
        Get AiidaVaspWorkChain class objects.

        Returns:
            tuple: (relax_calcs, static_calc).
        """
        relax_pks, static_pk = self.get_vasp_calculation_pks()
        if self._exit_status == 0:
            relax = [ AiidaVaspWorkChain(load_node(pk))
                                for pk in relax_pks ]
            static = AiidaVaspWorkChain(load_node(static_pk),
                                              ignore_warning=True)
        else:
            relax = [ AiidaVaspWorkChain(load_node(pk))
                                for pk in relax_pks[:-1] ]
            static = None
        return (relax, static)

    def get_pks(self) -> dict:
        """
        Get pks.

        Returns:
            dict: Contains relax pk and structure pk.
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

    def get_relaxplot(self) -> dict:
        """
        Get RelaxPlot class object.
        """
        relax_vasps, static_vasp = self.get_vasp_calculations()
        relax_data = {}
        relax_data['max_force'] = np.array(
                [ relax_vasp.get_max_force() for relax_vasp in relax_vasps ])
        # stress xx yy zz yz zx xy
        relax_data['stress'] = \
                np.array([ relax_vasp.stress.flatten()[[0,4,8,5,6,1]]
                               for relax_vasp in relax_vasps ])
        relax_data['energy'] = np.array([ relax_vasp.energy
                                              for relax_vasp in relax_vasps ])
        relax_data['abc'] = np.array([ Lattice(relax_vasp.final_cell[0]).abc
                                           for relax_vasp in relax_vasps ])
        relax_data['steps'] = \
                np.array([ i+1 for i in range(len(relax_vasps)) ])

        if static_vasp is None:
            static_data = None
        else:
            static_data = {
                    'max_force': static_vasp.get_max_force(),
                    'stress': static_vasp.stress.flatten()[[0,4,8,5,6,1]] ,
                    'energy': static_vasp.energy,
                    'abc': Lattice(static_vasp.initial_cell[0]).abc,
                    }

        relax_plot = RelaxPlot(relax_data=relax_data,
                               static_data=static_data)

        return relax_plot

    def plot_convergence(self):
        """
        Plot convergence.

        Todo:
            This function must be moved in plot directory.
        """
        plt.rcParams["font.size"] = 14

        fig = plt.figure(figsize=(16,13))
        ax1 = fig.add_axes((0.15, 0.1, 0.35,  0.35))
        ax2 = fig.add_axes((0.63, 0.1, 0.35, 0.35))
        ax3 = fig.add_axes((0.15, 0.55, 0.35, 0.35))
        ax4 = fig.add_axes((0.63, 0.55, 0.35, 0.35))

        relax_plot = self.get_relaxplot()
        relax_plot.plot_max_force(ax1)
        relax_plot.plot_energy(ax2)
        relax_plot.plot_stress(ax3)
        relax_plot.plot_abc(ax4)

        return fig


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
        are the same.
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
            warnings.warn("Final structure in latest RelaxWorkChain (pk={}) "
                           "does not find. So build RelaxAnalyzer with "
                           "previous RelaxWorkChain (pk={}) "
                           " as a final structure.".format(
                               self._aiida_relaxes[-1].pk,
                               self._aiida_relaxes[-2].pk,
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

    def get_relaxplots(self) -> list:
        """
        Get RelaxPlot class objects.
        """
        relax_plots = []
        for relax in self._aiida_relaxes:
            relax_plots.append(relax.get_relaxplot())

        start_step = 1
        for relax_plot in relax_plots:
            relax_plot.set_steps(start_step=start_step)
            start_step = relax_plot._relax_data['steps'][-1]

        return relax_plots

    def plot_convergence(self) -> list:
        """
        Get RelaxPlot class objects.
        """
        fig = plt.figure(figsize=(16,13))
        ax1 = fig.add_axes((0.15, 0.1, 0.35,  0.35))
        ax2 = fig.add_axes((0.63, 0.1, 0.35, 0.35))
        ax3 = fig.add_axes((0.15, 0.55, 0.35, 0.35))
        ax4 = fig.add_axes((0.63, 0.55, 0.35, 0.35))

        relax_plots = self.get_relaxplots()

        for i, relax_plot in enumerate(relax_plots):
            if i == 0:
                decorate = True
            else:
                decorate = False
            relax_plot.plot_max_force(ax1, decorate=decorate)
            relax_plot.plot_energy(ax2, decorate=decorate)
            relax_plot.plot_stress(ax3, decorate=decorate)
            relax_plot.plot_abc(ax4, decorate=decorate)

        return fig
