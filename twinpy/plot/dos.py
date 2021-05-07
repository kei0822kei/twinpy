#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module is used for plotting phonon density of states.
"""

import numpy as np
from phonopy.phonon.dos import TotalDos, PartialDos
from twinpy.plot.base import (DEFAULT_COLORS,
                              get_plot_properties_for_trajectory)

class _DosPlot():
    """
    Base for TotalDosPlot and TotalDosesPlot.
    """

    def __init__(self,
                 min_freq:float,
                 max_freq:float,
                 max_dos:float,
                 flip_xy:bool=False,
                 ):
        """
        Args:
            flip_xy (bool): Whether to flip x and y.
                            This cannot change later.
        """
        self._min_frequency = min_freq
        self._max_frequency = max_freq
        self._max_dos = max_dos
        self._flip_xy = flip_xy

    @property
    def min_frequency(self):
        """
        Min frequency.
        """
        return self._min_frequency

    @property
    def max_frequency(self):
        """
        Max frequency.
        """
        return self._max_frequency

    def set_xlim(self, ax, xmin:float=None, xmax:float=None):
        """
        Set xlim.

        Args:
            xmin (float): If None, min(frequences) -
                          (max(frequences) - min(frequences)) * 1.05 is set.
            xmax (float): If None, max(frequences) +
                          (max(frequences) - min(frequences)) * 1.05 is set.
        """
        freq_min = self._min_frequency
        freq_max = self._max_frequency
        span = freq_max - freq_min
        if xmin is None:
            _xmin = freq_min - span * 0.05
        else:
            _xmin = xmin
        if xmax is None:
            _xmax = freq_max + span * 0.05
        else:
            _xmax = xmax
        ax.set_xlim(_xmin, _xmax)

    def set_ylim(self, ax, ymin:float=0.01, ymax:float=None):
        """
        Set ylim.

        Args:
            ymax (float): If None, max(dos) * 1.05 is set.
        """
        dos_max = self._max_dos
        if ymax is None:
            _ymax = dos_max * 0.05
        else:
            _ymax = ymax
        ax.set_ylim(ymin, _ymax)

    def plot_xlabel(self, ax):
        """
        Show x label.
        """
        label = "Frequency [THz]"
        if self._flip_xy:
            ax.set_ylabel(label,
                          fontsize=20)
            ax.tick_params(axis='y', labelsize=16)
        else:
            ax.set_xlabel(label,
                          fontsize=20)
            ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(labelbottom=True,
                       labelleft=True,
                       labelright=False,
                       labeltop=False,
                       bottom=True,
                       left=True,
                       right=False,
                       top=False)

    def plot_vline(self, ax, hval:float=0.):
        """
        Plot virtical line.
        """
        if self._flip_xy:
            ax.axhline(hval, c='grey', linestyle='--', linewidth=0.5)
        else:
            ax.axvline(hval, c='grey', linestyle='--', linewidth=0.5)

    def plot_legend(self, ax):
        """
        Plot legend.
        """
        ax.legend()


class TotalDosPlot(_DosPlot):
    """
    Total dos plot.
    """

    def __init__(self,
                 total_dos:TotalDos,
                 flip_xy:bool=False,
                 ):
        """
        Args:
            total_dos: Phonopy TotalDos class object.
            flip_xy (bool): Whether to flip x and y.
                            This cannot change later.
        """
        self._total_dos = total_dos
        min_freq = min(self._total_dos.frequency_points)
        max_freq = max(self._total_dos.frequency_points)
        max_dos = max(self._total_dos.dos)
        super().__init__(min_freq=min_freq,
                         max_freq=max_freq,
                         max_dos=max_dos,
                         flip_xy=flip_xy)

    @property
    def total_dos(self):
        """
        Total dos.
        """
        return self._total_dos

    def get_imaginary_states(self, get_ratio:bool=False):
        dos = self._total_dos.dos
        freq = self._total_dos.frequency_points
        interval = freq[1] - freq[0]
        zero_idx = np.where(freq<0)[0][-1] + 1
        if get_ratio:
            img = np.sum(dos[:zero_idx]) / np.sum(dos) * interval
        else:
            img = np.sum(dos[:zero_idx]) * interval
        return img

    def plot_total_dos(self, ax, label:str=None, is_cumulative=False, c='r', linestyle='-',
                       alpha=1., linewidth=1.5, multi=1.):
        """
        Plot total dos

        Args:
            ax: Matplitlib ax.
        """
        freq = self._total_dos.frequency_points
        dos = self._total_dos.dos

        X = freq
        if is_cumulative:
            _width = X[1] - X[0]
            # Warning: probably not correct when useing tetrahedron => probably OK
            Y = np.cumsum(dos) * _width * multi  # unit []
        else:
            Y = dos * multi  # unit [/THz]

        if self._flip_xy:
            X, Y = Y, X

        ax.plot(X, Y, c=c, linestyle=linestyle, alpha=alpha,
                linewidth=linewidth, label=label)

    def plot_ylabel(self, ax):
        """
        Show y label.
        """
        label = "Total Dos [/THz]"
        if self._flip_xy:
            ax.set_xlabel(label,
                          fontsize=20)
            ax.tick_params(axis='x', labelsize=16)
        else:
            ax.set_ylabel(label,
                          fontsize=20)
            ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(labelbottom=True,
                       labelleft=True,
                       labelright=False,
                       labeltop=False,
                       bottom=True,
                       left=True,
                       right=False,
                       top=False)


class PartialDosPlot(_DosPlot):
    """
    Partial dos plot.
    """

    def __init__(self,
                 partial_dos:PartialDos,
                 flip_xy:bool=False,
                 ):
        """
        Args:
            partial_dos: Phonopy PartialDos class object.
            flip_xy (bool): Whether to flip x and y.
                            This cannot change later.
        """
        self._partial_dos = partial_dos
        min_freq = min(self._partial_dos.frequency_points)
        max_freq = max(self._partial_dos.frequency_points)
        max_dos = self._partial_dos.partial_dos.max()
        super().__init__(min_freq=min_freq,
                         max_freq=max_freq,
                         max_dos=max_dos,
                         flip_xy=flip_xy)

    @property
    def partial_dos(self):
        """
        Partial dos.
        """
        return self._partial_dos

    def plot_partial_dos(self,
                         ax,
                         indices=None,
                         labels=None,
                         is_cumulative=False,
                         cs=None,
                         linestyle='-',
                         alpha=1.,
                         linewidth=1.5):
        """
        Plot partial dos.
        """
        pdos = self._partial_dos
        num_pdos = len(pdos.partial_dos)  # equal atom num

        if indices is None:
            indices = []
            for i in range(num_pdos):
                indices.append([i])
                labels.append([str(i)])

        if labels is None:
            labels = []
            for i in range(num_pdos):
                labels.append([str(i)])

        if cs is None:
            color_nums = [ -(i%len(DEFAULT_COLORS))-1
                               for i in range(len(indices)) ]
            cs = [ DEFAULT_COLORS[num] for num in color_nums ]

        for j, set_for_sum in enumerate(indices):
            pdos_sum = np.zeros_like(pdos.frequency_points)
            for i in set_for_sum:
                if i > num_pdos - 1:
                    print("Index number \'%d\' is specified," % (i + 1))
                    print("but it is not allowed to be larger than the number "
                           "of atoms.")
                    raise ValueError
                if i < 0:
                    print("Index number \'%d\' is specified, but it must be "
                          "positive." % (i + 1))
                    raise ValueError
                pdos_sum += pdos.partial_dos[i]

            X = pdos.frequency_points
            if is_cumulative:
                # Warning: probably not correct when useing tetrahedron => probably OK
                _width = X[1] - X[0]
                Y = np.cumsum(pdos_sum) * _width
            else:
                Y = pdos_sum

            if self._flip_xy:
                X, Y = Y, X

            ax.plot(X, Y,
                    label=labels[j],
                    c=cs[j],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha)

    def plot_ylabel(self, ax):
        """
        Show y label.
        """
        label = "Partial Dos [/THz]"
        if self._flip_xy:
            ax.set_xlabel(label,
                          fontsize=20)
            ax.tick_params(axis='x', labelsize=16)
        else:
            ax.set_ylabel(label,
                          fontsize=20)
            ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(labelbottom=True,
                       labelleft=True,
                       labelright=False,
                       labeltop=False,
                       bottom=True,
                       left=True,
                       right=False,
                       top=False)


class TotalDosesPlot():
    """
    Total Doses Plot.
    """

    def __init__(
           self,
           total_doses:list,
           flip_xy:bool=False,
           ):
        """
        Args:
            total_doses (list): List of TotalDos class object.
            flip_xy (bool): Whether to flip x and y.
                            This cannot change later.
        """
        self._total_doses = total_doses
        self._flip_xy = flip_xy
        self._dosplots = [ TotalDosPlot(total_dos=tdos, flip_xy=self._flip_xy)
                               for tdos in self._total_doses ]
        self._cs = None
        self._alphas = None
        self._linewidths = None
        self._linestyles = None
        self.set_line_properties(base_color='r')

    @property
    def total_doses(self):
        """
        Total doses.
        """
        return self._total_doses

    @property
    def total_dos_plots(self):
        """
        Total dos plots.
        """
        return self._dosplots

    def set_line_properties(self, base_color:str='r'):
        """
        Set line properties.
        """
        self._cs, self._alphas, self._linewidths, self._linestyles = \
                get_plot_properties_for_trajectory(
                        plot_nums=len(self._dosplots),
                        base_color=base_color)

    def set_cs(self, cs):
        """
        Set cs.
        """
        self._cs = cs

    def set_alphas(self, alphas):
        """
        Set alpha.
        """
        self._alphas = alphas

    def set_linewidths(self, linewidths):
        """
        Set linewidths.
        """
        self._linewidths = linewidths

    def set_linestyles(self, linestyles):
        """
        Set linestyles.
        """
        self._linestyles = linestyles

    def plot_total_doses(self, ax, is_cumulative:bool=False):
        for i, dosplot in enumerate(self._dosplots):
            dosplot.plot_total_dos(ax,
                                   c=self._cs[i],
                                   linestyle=self._linestyles[i],
                                   alpha=self._alphas[i],
                                   linewidth=self._linewidths[i],
                                   is_cumulative=is_cumulative,
                                   )
            if i == len(self._dosplots)-1:
                dosplot.plot_vline(ax)
                dosplot.plot_xlabel(ax)
                dosplot.plot_ylabel(ax)

    def set_xlim(self, ax, xmin:float=None, xmax:float=None):
        """
        Set xlim.

        Args:
            xmin (float): If None, min(frequences) -
                          (max(frequences) - min(frequences)) * 1.05 is set.
            xmax (float): If None, max(frequences) +
                          (max(frequences) - min(frequences)) * 1.05 is set.
        """
        freq_min = min([ tdos.min_frequency for tdos in self._dosplots ])
        freq_max = max([ tdos.max_frequency for tdos in self._dosplots ])
        span = freq_max - freq_min
        if xmin is None:
            _xmin = freq_min - span * 0.05
        else:
            _xmin = xmin
        if xmax is None:
            _xmax = freq_max + span * 0.05
        else:
            _xmax = xmax
        self._min_frequency = freq_min
        self._max_frequency = freq_max
        self._xlim = (_xmin, _xmax)
        ax.set_xlim(self._xlim)


class PartialDosesPlot():
    """
    Partial Doses Plot.
    """

    def __init__(
           self,
           partial_doses:list,
           flip_xy:bool=False,
           ):
        """
        Args:
            partial_doses (list): List of PartialDos class object.
            flip_xy (bool): Whether to flip x and y.
                            This cannot change later.
        """
        self._partial_doses = partial_doses
        self._flip_xy = flip_xy
        self._pdosplots = [ PartialDosPlot(partial_dos=pdos,
                                           flip_xy=self._flip_xy)
                                for pdos in self._partial_doses ]

    @property
    def partial_doses(self):
        """
        Total doses.
        """
        return self._partial_doses

    @property
    def partial_dos_plots(self):
        """
        Total dos plots.
        """
        return self._pdosplots

    def plot_partial_doses(self,
                           ax,
                           indices):
        """
        Plot partial doses.
        """
        color_nums = [ -(i%len(DEFAULT_COLORS))-1
                           for i in range(len(indices)) ]
        plot_nums = len(self._partial_doses)
        base_cs = [ DEFAULT_COLORS[num] for num in color_nums ]
        _, alphas, linewidths, linestyles = \
                get_plot_properties_for_trajectory(
                        plot_nums=plot_nums,
                        base_color='r')
        for i, indice in enumerate(indices):
            for j, pdosplot in enumerate(self._pdosplots):
                print(i,j)
                pdosplot.plot_partial_dos(ax,
                                          indices=[indice],
                                          cs=[base_cs[i]],
                                          alpha=alphas[j],
                                          linewidth=linewidths[j],
                                          linestyle=linestyles[j])

                if i == len(indices)-1 and j == len(self._pdosplots)-1:
                    pdosplot.plot_vline(ax)
                    pdosplot.plot_xlabel(ax)
                    pdosplot.plot_ylabel(ax)

    def set_xlim(self, ax, xmin:float=None, xmax:float=None):
        """
        Set xlim.

        Args:
            xmin (float): If None, min(frequences) -
                          (max(frequences) - min(frequences)) * 1.05 is set.
            xmax (float): If None, max(frequences) +
                          (max(frequences) - min(frequences)) * 1.05 is set.
        """
        freq_min = min([ pdos.min_frequency for pdos in self._pdosplots ])
        freq_max = max([ pdos.max_frequency for pdos in self._pdosplots ])
        span = freq_max - freq_min
        if xmin is None:
            _xmin = freq_min - span * 0.05
        else:
            _xmin = xmin
        if xmax is None:
            _xmax = freq_max + span * 0.05
        else:
            _xmax = xmax
        self._min_frequency = freq_min
        self._max_frequency = freq_max
        self._xlim = (_xmin, _xmax)
        ax.set_xlim(self._xlim)
