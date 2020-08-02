#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plot
----
provide various kinds of plot
"""

import numpy as np
from phonopy.phonon.dos import TotalDos as PhonopyTotalDos
from twinpy.plot.base import DEFAULT_COLORS


class TotalDosPlot(PhonopyTotalDos):
    """
    Total dos plot.
    """

    def __init__(self,
                 ax,
                 mesh_object,
                 sigma=None,
                 use_tetrahedron_method=False,
                 ):
        """
        Note:
            For explanation for input variables, see PhonopyTotalDos.
        """
        self.ax = ax
        super().__init__(mesh_object=mesh_object,
                         sigma=sigma,
                         use_tetrahedron_method=use_tetrahedron_method)

    def plot(self,
             ax,
             c,
             alpha,
             linestyle,
             linewidth,
             xlabel=None,
             ylabel=None,
             draw_grid=True,
             flip_xy=False,
             label=None,
             ):
        if flip_xy:
            _xlabel = 'Density of states'
            _ylabel = 'Frequency'
        else:
            _xlabel = 'Frequency'
            _ylabel = 'Density of states'

        if xlabel is not None:
            _xlabel = xlabel
        if ylabel is not None:
            _ylabel = ylabel

        _plot_total_dos(ax,
                        self._frequency_points,
                        self._dos,
                        c=c,
                        alpha=alpha,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        freq_Debye=self._freq_Debye,
                        Debye_fit_coef=self._Debye_fit_coef,
                        xlabel=_xlabel,
                        ylabel=_ylabel,
                        draw_grid=draw_grid,
                        flip_xy=flip_xy,
                        label=label,
                        )


def _plot_total_dos(ax,
                    frequency_points,
                    total_dos,
                    c,
                    alpha,
                    linewidth,
                    linestyle,
                    freq_Debye=None,
                    Debye_fit_coef=None,
                    xlabel=None,
                    ylabel=None,
                    draw_grid=True,
                    flip_xy=False,
                    label=None,
                    ):
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_tick_params(which='both', direction='in')
    ax.yaxis.set_tick_params(which='both', direction='in')

    if freq_Debye is not None:
        freq_pitch = frequency_points[1] - frequency_points[0]
        num_points = int(freq_Debye / freq_pitch)
        freqs = np.linspace(0, freq_Debye, num_points + 1)

    if flip_xy:
        ax.plot(total_dos, frequency_points, c=c, alpha=alpha,
                linewidth=linewidth, linestyle=linestyle, label=label)
        if freq_Debye:
            ax.plot(np.append(Debye_fit_coef * freqs**2, 0),
                    np.append(freqs, freq_Debye), 'b-', linewidth=1)
    else:
        ax.plot(frequency_points, total_dos, c=c, alpha=alpha,
                linewidth=linewidth, linestyle=linestyle, label=label)
        if freq_Debye:
            ax.plot(np.append(freqs, freq_Debye),
                    np.append(Debye_fit_coef * freqs**2, 0), 'b-', linewidth=1)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(draw_grid)


def total_doses_plot(ax,
                     phonons,
                     mesh,
                     sigma=None,
                     freq_min=None,
                     freq_max=None,
                     freq_pitch=None,
                     use_tetrahedron_method=False,
                     cs=None,
                     alphas=None,
                     linewidths=None,
                     linestyles=None,
                     draw_grid=True,
                     flip_xy=False,
                     labels=None,
                     **kwargs):
    """
    Plot multiple dos.

    Note:
        For explanation for input variables, see PhonopyTotalDos.
    """
    if cs is None:
        cs = [ DEFAULT_COLORS[i%len(DEFAULT_COLORS)]
                   for i in range(len(phonons)) ]
    if alphas is None:
        alphas = [ 1. ] * len(phonons)
    if linestyles is None:
        linestyles = [ 'solid' ] * len(phonons)
    if linewidths is None:
        linewidths = [ 1. ] * len(phonons)
    if labels is None:
        labels = [None] * len(phonons)

    total_doses = []
    for phonon in phonons:
        phonon.set_mesh(mesh)
        total_dos = TotalDosPlot(ax,
                                 mesh_object=phonon.mesh,
                                 sigma=sigma,
                                 use_tetrahedron_method=use_tetrahedron_method)
        total_dos.set_draw_area(freq_min, freq_max, freq_pitch)
        total_dos.run()
        total_doses.append(total_dos)

    for i, total_dos in enumerate(total_doses):
        total_dos.plot(ax=ax,
                       c=cs[i],
                       alpha=alphas[i],
                       linewidth=linewidths[i],
                       linestyle=linestyles[i],
                       draw_grid=draw_grid,
                       flip_xy=flip_xy,
                       label=labels[i],
                       )
