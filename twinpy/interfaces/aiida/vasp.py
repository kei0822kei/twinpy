#!/usr/bin/env python

"""
Interface for Aiida-Vasp.
"""
from pprint import pprint
import warnings
import numpy as np
from matplotlib import pyplot as plt
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.orm import (load_node,
                       Node,
                       WorkChainNode,
                       QueryBuilder)
from aiida.common.exceptions import NotExistentAttributeError
from twinpy.common.kpoints import Kpoints
from twinpy.common.utils import print_header
from twinpy.structure.lattice import CrystalLattice
from twinpy.interfaces.aiida.base import (check_process_class,
                                          get_cell_from_aiida,
                                          _WorkChain)
from twinpy.analysis.relax_analyzer import RelaxAnalyzer
from twinpy.plot.base import line_chart
from twinpy.plot.relax import RelaxPlot


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
        if self._exit_status == 0:
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
        try:
            misc = self._node.outputs.misc.get_dict()
            self._forces = self._node.outputs.forces.get_array('final')
            self._stress = self._node.outputs.stress.get_array('final')
            self._energy = misc['total_energies']['energy_extrapolated']
        except NotExistentAttributeError:
            warnings.warn("Could not extract outputs. Please check report.")

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

    def get_kpoints_info(self, include_two_pi:bool=True) -> dict:
        """
        Get sampling kpoints information.

        Args:
            include_two_pi: If True, 2*pi is included for reciprocal lattice.

        Returns:
            dict: Contains kpoints information.
        """
        mesh, offset = self._node.inputs.kpoints.get_kpoints_mesh()
        total_mesh = mesh[0] * mesh[1] * mesh[2]
        kpt = Kpoints(lattice=self._initial_cell[0])
        dic = kpt.get_dict(mesh=mesh, include_two_pi=include_two_pi)
        dic['offset'] = offset
        del dic['input_interval']
        del dic['decimal_handling']
        del dic['use_symmetry']
        if self._exit_status == 0:
            sampling_kpoints = self._node.outputs.kpoints.get_array('kpoints')
            weights = self._node.outputs.kpoints.get_array('weights')
            weights_num = (weights * total_mesh).astype(int)
            dic['sampling_kpoints'] = sampling_kpoints
            dic['weights'] = weights_num
            dic['total_irreducible_kpoints'] =len(weights_num)
        return dic

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
            'incar': self._node.inputs.parameters.get_dict()['incar'],
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
        kpoints_info_for_print = self.get_kpoints_info()
        if self._exit_status == 0:
            del kpoints_info_for_print['sampling_kpoints']
            del kpoints_info_for_print['weights']

        print_header('VASP settings')
        pprint(self.get_vasp_settings())
        print("\n")
        print_header("kpoints information")
        pprint(kpoints_info_for_print)
        if self._exit_status == 0:
            print("\n")
            print_header('VASP outputs')
            print("# stress")
            pprint(self._stress)
            print("")
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
        self._step_energies = None
        self._set_step_energies(ignore_warning=ignore_warning)

    def _set_step_energies(self, ignore_warning):
        """
        Set step energies.
        """
        try:
            eg = self._node.outputs.energies
            self._step_energies = {
                'energy_extrapolated': eg.get_array('energy_extrapolated'),
                'energy_extrapolated_final':
                    eg.get_array('energy_extrapolated_final'),
                    }
        except NotExistentAttributeError:
            if not ignore_warning:
                warnings.warn("Output energy could not find.\n"
                              "process state:{} (pk={})".format(
                                  self.process_state, self._node.pk))

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

    @property
    def step_energies(self):
        """
        Energy for each steps.
        """
        return self._step_energies

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

    def plot_energy_convergence(self):
        """
        Plot energy convergence.
        """
        fig = plt.figure()
        ax =fig.add_subplot(111)
        energies = self._step_energies['energy_extrapolated']
        steps = [ i+1 for i in range(len(energies)) ]
        line_chart(ax,
                   xdata=steps,
                   ydata=energies,
                   xlabel='Relax steps',
                   ylabel='Energy [eV]',
                   )
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    def get_description(self):
        """
        Get description.
        """
        self._print_common_information()
        print_header('PKs')
        pprint(self.get_pks())
        print("\n")
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
            node: Aiida Node.
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
            relax_structure = self._node.outputs.relax__structure
            self._final_structure_pk = relax_structure.pk
            self._final_cell = get_cell_from_aiida(relax_structure)
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
                            aiida_vasp.get_pks()['initial_structure_pk']
                    self._current_final_cell = aiida_vasp.initial_cell

    @property
    def final_cell(self):
        """
        Final cell.
        """
        return self._final_cell

    @property
    def current_final_cell(self):
        """
        Current final cell.
        """
        return self._current_final_cell

    def get_relax_settings(self) -> dict:
        """
        Get relax settings.

        Returns:
            dict: Contains relax settings.
        """
        keys = [ key for key in dir(self._node.inputs) if 'relax' in key ]
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

    def get_relaxplot(self, start_step:int=1) -> RelaxPlot:
        """
        Get RelaxPlot class object.

        Args:
            start_step: The step number of the first relax in this WorkChain.
                        If you relax 20 steps in the privious RelaxWorkChain,
                        for example, start_step becomes 21.

        Returns:
            RelaxPlot: RelaxPlot class object.
        """
        relax_vasps, static_vasp = self.get_vasp_calculations()
        relax_data = {}
        relax_data['max_force'] = \
                np.array([ relax_vasp.get_max_force()
                               for relax_vasp in relax_vasps ])
        # stress xx yy zz yz zx xy
        relax_data['stress'] = \
                np.array([ relax_vasp.stress.flatten()[[0,4,8,5,6,1]]
                               for relax_vasp in relax_vasps ])
        relax_data['energy'] = \
                np.array([ relax_vasp.energy
                               for relax_vasp in relax_vasps ])
        relax_data['abc'] = \
                np.array([ CrystalLattice(relax_vasp.final_cell[0]).abc
                               for relax_vasp in relax_vasps ])
        relax_data['step_energies_collection'] = \
                [ relax_vasp.step_energies for relax_vasp in relax_vasps ]

        if static_vasp is None:
            static_data = None
        else:
            static_data = {
                    'max_force': static_vasp.get_max_force(),
                    'stress': static_vasp.stress.flatten()[[0,4,8,5,6,1]] ,
                    'energy': static_vasp.energy,
                    'abc': CrystalLattice(static_vasp.initial_cell[0]).abc,
                    }

        relax_plot = RelaxPlot(relax_data=relax_data,
                               static_data=static_data,
                               start_step=start_step)

        return relax_plot

    def plot_convergence(self):
        """
        Plot convergence.
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
            start_step = relax_plot.relax_data['steps'][-1]

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
            decorate = bool(i == 0)

            relax_plot.plot_max_force(ax1, decorate=decorate)
            relax_plot.plot_energy(ax2, decorate=decorate)
            relax_plot.plot_stress(ax3, decorate=decorate)
            relax_plot.plot_abc(ax4, decorate=decorate)

        return fig
