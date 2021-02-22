#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module is used for provide information from vasp relax.
"""

import numpy as np
from twinpy.plot.base import DEFAULT_COLORS, DEFAULT_MARKERS, line_chart
from twinpy.structure.diff import get_structure_diff


class RelaxPlot():
    """
    Relax transition plot class.
    """

    def __init__(
           self,
           relax_data:dict,
           static_data:dict=None,
           start_step:int=1,
           ):
        """
        Args:
            relax_data: Relax data.
            static_data: Static data.
            start_step: The step number of the first relax in this WorkChain.
                        If you relax 20 steps in the privious RelaxWorkChain,
                        for example, start_step becomes 21.
        """
        self._relax_data = relax_data
        self._static_data = static_data
        self._start_step = start_step
        if self._static_data is None:
            self._exist_static = False
        else:
            self._exist_static = True
        self._vasp_final_steps = None
        self._set_vasp_final_steps()

    @property
    def relax_data(self):
        """
        Relax data.
        """
        return self._relax_data

    @property
    def static_data(self):
        """
        Static data.
        """
        return self._static_data

    @property
    def start_step(self):
        """
        Start step.
        """
        return self._start_step

    def _set_vasp_final_steps(self):
        """
        Set vasp final steps.
        """
        eg_cols = self._relax_data['step_energies_collection']
        vasp_final_steps = []
        count = self._start_step - 1
        for cols in eg_cols:
            count += len(cols['energy_extrapolated'])
            vasp_final_steps.append(count)
        self._vasp_final_steps = vasp_final_steps

    def plot_max_force(self,
                       ax,
                       decorate:bool=True):
        """
        Plot max force.

        Args:
            ax: Matplotlib subplot.
            decorate (bool): If True, decorate figure.
        """
        if decorate:
            xlabel = 'Relax Steps'
            ylabel = 'Max Force'
        else:
            xlabel = ylabel = None

        steps = self._vasp_final_steps
        max_forces = self._relax_data['max_force']
        line_chart(
                ax,
                steps,
                max_forces,
                xlabel=xlabel,
                ylabel=ylabel,
                c=DEFAULT_COLORS[0],
                marker=DEFAULT_MARKERS[0],
                facecolor='None')

        if self._exist_static:
            static_step = steps[-1] + 0.1
            static_max_force = self._static_data['max_force']
            ax.scatter(static_step, static_max_force,
                       c=DEFAULT_COLORS[0], marker='*', s=150)

    def plot_energy(self,
                    ax,
                    decorate:bool=True):
        """
        Plot energy.

        Args:
            ax: Matplotlib subplot.
            decorate (bool): If True, decorate figure.
        """
        if decorate:
            xlabel = 'Relax Steps'
            ylabel = 'Energy [eV]'
        else:
            xlabel = ylabel = None

        steps = self._vasp_final_steps
        energies = self._relax_data['energy']
        eg_cols = self._relax_data['step_energies_collection']
        vasp_energies = []
        for cols in eg_cols:
            vasp_energies.extend(cols['energy_extrapolated'])
        vasp_steps = [ i+1 for i in range(len(vasp_energies)) ]

        line_chart(
                ax,
                vasp_steps,
                vasp_energies,
                xlabel,
                ylabel,
                c=DEFAULT_COLORS[0],
                marker=DEFAULT_MARKERS[0],
                s=5,
                facecolor='None')
        ax.scatter(self._vasp_final_steps, energies, c=DEFAULT_COLORS[0],
                   marker=DEFAULT_MARKERS[1], facecolor=DEFAULT_MARKERS[0])

        if self._exist_static:
            static_step = steps[-1] + 0.1
            static_energy = self._static_data['energy']
            ax.scatter(static_step, static_energy,
                       edgecolor=DEFAULT_COLORS[0], marker='*', s=150,
                       facecolor='None')

    def plot_stress(self,
                    ax,
                    decorate:bool=True):
        """
        Plot stress.

        Args:
            ax: Matplotlib subplot.
            decorate (bool): If True, decorate figure.
        """
        if decorate:
            xlabel = 'Relax Steps'
            ylabel = 'Stress'
            stress_labels = ['xx', 'yy', 'zz', 'yz', 'zx', 'xy']
        else:
            xlabel = ylabel = None
            stress_labels = [None] * 6

        steps = self._vasp_final_steps
        stresses = self._relax_data['stress']

        for i in range(6):
            line_chart(
                    ax,
                    steps,
                    stresses[:,i],
                    xlabel,
                    ylabel,
                    c=DEFAULT_COLORS[i],
                    marker=DEFAULT_MARKERS[i],
                    facecolor='None',
                    label=stress_labels[i])

        if self._exist_static:
            static_step = steps[-1] + 0.1
            static_stress = self._static_data['stress']
            for i in range(6):
                ax.scatter(static_step, static_stress[i],
                           c=DEFAULT_COLORS[i], marker='*', s=150)

        ax.legend(loc='upper left')

    def plot_abc(self,
                 ax,
                 decorate:bool=True):
        """
        Plot abc.

        Args:
            ax: Matplotlib subplot.
            decorate (bool): If True, decorate figure.
        """
        if decorate:
            xlabel = 'Relax Steps'
            ylabel = 'Length [angstrom]'
            abc_labels = ['a', 'b', 'c']
        else:
            xlabel = ylabel = None
            abc_labels = [None] * 3

        steps = self._vasp_final_steps
        abcs = self._relax_data['abc']

        for i in range(3):
            line_chart(
                    ax,
                    steps,
                    abcs[:,i],
                    xlabel=xlabel,
                    ylabel=ylabel,
                    c=DEFAULT_COLORS[i],
                    marker=DEFAULT_MARKERS[i],
                    facecolor='None',
                    label=abc_labels[i])

        if self._exist_static:
            static_step = steps[-1] + 0.1
            static_abc = self._static_data['abc']
            for i in range(3):
                ax.scatter(static_step, static_abc[i],
                           c=DEFAULT_COLORS[i], marker='*', s=150)

        ax.legend(loc='upper left')


def plot_atom_diff(ax,
                   initial_cell:tuple,
                   final_cell:tuple,
                   decorate:bool=True,
                   direction:str='x',
                   shuffle:bool=True,
                   label:str='default',
                   **kwargs):
    """
    Plot atom diff.

    Args:
        initial_cell (tuple): Initial cell.
        final_cell (tuple): Final cell.
        decorate (bool): If True, decorate figure.
        direction (str): Diff direction.
        shuffle (bool): If True, diffrence of scaled positions,
                        which ignore lattice shear, are ploted.

    Notes:
        For input 'kwargs', see twinpy.plot.base.line_chart.
    """
    diff = get_structure_diff(cells=[initial_cell, final_cell],
                              include_base=False)
    if shuffle:
        scaled_posi_diffs = diff['scaled_posi_diffs'][0]
        cart_posi_diffs = np.dot(final_cell[0].T, scaled_posi_diffs.T).T
    else:
        cart_posi_diffs = diff['cart_posi_diffs'][0]

    z_coords = np.dot(initial_cell[0].T, initial_cell[1].T).T[:,2]

    if label == 'default':
        label = direction

    if decorate:
        xlabel = 'Distance [angstrom]'
        ylabel = 'Initial z coordinate'
    else:
        xlabel = None
        ylabel = None

    dic = {'x': 0, 'y': 1, 'z': 2}
    idx = dic[direction]

    line_chart(ax=ax,
               xdata=cart_posi_diffs[:,idx],
               ydata=z_coords,
               xlabel=xlabel,
               ylabel=ylabel,
               label=label,
               sort_by='y',
               **kwargs)

    ax.legend()
