#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Relax transition plot.
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
           ):
        """
        Args:
            relax_data (dict): Relax data.
            static_data (dict): Static data.
        """
        self._relax_data = relax_data
        self._static_data = static_data
        if self._static_data is None:
            self._exist_static = False
        else:
            self._exist_static = True

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

    def set_steps(self, start_step:int):
        """
        Set steps.

        Args:
            start_step (int): Start step.
        """
        steps = np.array(self._relax_data['steps'])
        self._relax_data['steps'] = list(steps + start_step - 1)

    def plot_max_force(self,
                       ax,
                       is_logscale:bool=True,
                       decorate:bool=True):
        """
        Plot max force.

        Args:
            ax: Matplotlib subplot.
            is_logscale (bool): If True, plot with log scale.
            decorate (bool): If True, decorate figure.
        """
        if decorate:
            xlabel = 'Relax Steps'
            ylabel = 'Max Force'
        else:
            xlabel = ylabel = None

        steps = self._relax_data['steps']
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

        if is_logscale:
            ax.set_yscale('log')

    def plot_energy(self,
                    ax,
                    decorate:bool=True):
        """
        Plot energy.

        Args:
            ax: Matplotlib subplot.
            is_logscale: If True, plot with log scale.
            decorate (bool): If True, decorate figure.
        """
        if decorate:
            xlabel = 'Relax Steps'
            ylabel = 'Energy [eV]'
        else:
            xlabel = ylabel = None

        steps = self._relax_data['steps']
        energies = self._relax_data['energy']
        line_chart(
                ax,
                steps,
                energies,
                xlabel,
                ylabel,
                c=DEFAULT_COLORS[0],
                marker=DEFAULT_MARKERS[0],
                facecolor='None')

        if self._exist_static:
            static_step = steps[-1] + 0.1
            static_energy = self._static_data['energy']
            ax.scatter(static_step, static_energy,
                       c=DEFAULT_COLORS[0], marker='*', s=150)

    def plot_stress(self,
                    ax,
                    is_logscale:bool=False,
                    decorate:bool=True):
        """
        Plot stress.

        Args:
            ax: Matplotlib subplot.
            is_logscale: If True, plot with log scale.
            decorate (bool): If True, decorate figure.
        """
        if decorate:
            xlabel = 'Relax Steps'
            ylabel = 'Stress'
            stress_labels = ['xx', 'yy', 'zz', 'yz', 'zx', 'xy']
        else:
            xlabel = ylabel = None
            stress_labels = [None] * 6

        steps = self._relax_data['steps']
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
        if is_logscale:
            ax.set_yscale('log')

    def plot_abc(self,
                 ax,
                 is_logscale:bool=True,
                 decorate:bool=True):
        """
        Plot abc.

        Args:
            ax: Matplotlib subplot.
            is_logscale: If True, plot with log scale.
            decorate (bool): If True, decorate figure.
        """
        if decorate:
            xlabel = 'Relax Steps'
            ylabel = 'Length [angstrom]'
            abc_labels = ['a', 'b', 'c']
        else:
            xlabel = ylabel = None
            abc_labels = [None] * 3

        steps = self._relax_data['steps']
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
        ax.set_ylim((0, None))


def plot_atom_diff(ax,
                   initial_cell:tuple,
                   final_cell:tuple,
                   decorate:bool=True,
                   direction:str='x',
                   shuffle:bool=False,
                   **kwargs):
    """
    Plot atom diff.

    Args:
        initial_cell (tuple): Initial cell.
        final_cell (tuple): Final cell.
    """
    diff = get_structure_diff(cells=[initial_cell, final_cell],
                              include_base=False)
    if shuffle:
        scaled_posi_diffs = diff['scaled_posi_diffs'][0]
        cart_posi_diffs = np.dot(final_cell[0].T, scaled_posi_diffs.T).T
    else:
        cart_posi_diffs = diff['cart_posi_diffs'][0]

    z_coords = np.dot(initial_cell[0].T, initial_cell[1].T).T[:,2]

    if decorate:
        xlabel = 'Distance [angstrom]'
        ylabel = 'Initial z coordinate'
        label = direction
    else:
        xlabel = None
        ylabel = None
        label = None

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
