#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Band plot.
"""

from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from phonopy.phonon.band_structure import BandStructure
from twinpy.plot.base import (create_figure_axes,
                              get_plot_properties_for_trajectory)

plt.rcParams["font.family"] = "times new roman"


def decorate_string_for_latex(string):
    """
    decorate strings for latex
    """
    if string is None:
        decorated_string = ''
    elif string == 'GAMMA':
        decorated_string = "$" + string.replace("GAMMA", r"\Gamma") + "$"
    elif string == 'SIGMA':
        decorated_string = "$" + string.replace("SIGMA", r"\Sigma") + "$"
    elif string == 'DELTA':
        decorated_string = "$" + string.replace("DELTA", r"\Delta") + "$"
    elif string == 'LAMBDA':
        decorated_string = "$" + string.replace("LAMBDA", r"\Lambda") + "$"
    else:
        decorated_string = r"$\mathrm{%s}$" % string
    return decorated_string


def get_axes_distances(band_structure:BandStructure) -> list:
    """
    Get axes distances in angstrome.

    Args:
        band_structure: BandStructure class object.

    Returns:
        list: Axes distances in angstrome.

    Note:
        This function returns list of distances of connected band paths.
        To plot band structure, axes ratios are necessary.
        You can use resultant list as ratios for plotting band structure.
    """
    min_distance = 0.
    widths = []
    for distance, path_connection in zip(band_structure.get_distances(),
                                         band_structure.path_connections):
        if path_connection:
            continue

        width = distance[-1] - min_distance
        widths.append(width)
        min_distance = distance[-1]

    return widths


class BandPlot():
    """
    Band structure plot class.
    """
    def __init__(self,
                 band_structure:BandStructure,
                 ):
        """
        Init.

        Args:
            band_structure: BandStructure class object.
        """
        self._band_structure = band_structure
        self._axes_distances = get_axes_distances(band_structure)
        self._labels = self._band_structure.labels
        self._segment_frequences = None
        self._segment_distances = None
        self._segment_labels_distances = None
        self._set_segments()
        self._ylim = None
        self._min_frequency = None
        self._max_frequency = None
        self.set_ylim(ymin=None, ymax=None)

    def set_ylim(self, ymin:float=None, ymax:float=None):
        """
        Set ylim.

        Args:
            ymin (float): If None, min(frequences) -
                          (max(frequences) - min(frequences)) * 1.05 is set.
            ymax (float): If None, max(frequences) +
                          (max(frequences) - min(frequences)) * 1.05 is set.
        """
        freq_min = min([ freq.min() for freq in self._segment_frequences ])
        freq_max = max([ freq.max() for freq in self._segment_frequences ])
        span = freq_max - freq_min
        if ymin is None:
            _ymin = freq_min - span * 0.05
        else:
            _ymin = ymin
        if ymax is None:
            _ymax = freq_max + span * 0.05
        else:
            _ymax = ymax
        self._min_frequency = freq_min
        self._max_frequency = freq_max
        self._ylim = (_ymin, _ymax)

    def _set_segments(self):
        """
        Set segment frequences and distances.
        """
        frequences = self._band_structure.get_frequencies()
        distances = self._band_structure.get_distances()
        path_connections = self._band_structure.path_connections
        if self._labels is None:
            labels = [''] * (len(path_connections) * 2
                                 - path_connections.count(True))
        else:
            labels = self._labels

        seg_freqs = []
        seg_dis = []
        seg_labels_qpoints = []
        freqs = []
        dis = []
        iter_labels = iter(labels)
        labels_qpoints = []
        # for i in range(len(path_connections)):
        for freq, ds, conn in zip(frequences, distances, path_connections):
            if labels_qpoints == []:
                labels_qpoints = [(iter_labels.__next__(), 0.)]
            freqs.append(freq)
            dis.extend(ds)
            labels_qpoints.append((iter_labels.__next__(), dis[-1]-dis[0]))

            if conn:
                continue

            if len(freqs) == 1:
                seg_freqs.append(freqs[0])
            else:
                seg_freqs.append(np.vstack(freqs))
            dis = np.array(dis) - dis[0]
            seg_dis.append(dis)
            seg_labels_qpoints.append(labels_qpoints)
            freqs = []
            dis = []
            labels_qpoints = []

        self._segment_frequences = seg_freqs
        self._segment_distances = seg_dis
        self._segment_labels_distances = seg_labels_qpoints

    @property
    def band_structure(self):
        """
        Band structure.
        """
        return self._band_structure

    @property
    def axes_distances(self):
        """
        Axes distances.
        """
        return self._axes_distances

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

    @property
    def segment_frequences(self):
        """
        Segment freqences.
        """
        return self._segment_frequences

    @property
    def segment_distances(self):
        """
        Segment distances.
        """
        return self._segment_distances

    def plot_segment_band_structure(self, ax, frequences, distances,
                                    label:str=None, c='r', linestyle='-',
                                    alpha=1., linewidth=1.5,
                                    show_yscale:bool=True):
        """
        Plot segment band structure.
        """
        for i in range(frequences.shape[1]):
            if i == 0:
                _label = label
            else:
                _label = None
            ax.plot(distances, frequences[:,i], c=c, linestyle=linestyle,
                    alpha=alpha, linewidth=linewidth, label=_label)
        ax.set_xlim(min(distances), max(distances))
        labelleft = bool(show_yscale)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(labelbottom=True,
                       labelleft=labelleft,
                       labelright=False,
                       labeltop=False,
                       bottom=True,
                       left=True,
                       right=False,
                       top=False)

    def plot_ylabel(self, ax):
        """
        Plot ylabel.
        """
        ax.set_ylabel(decorate_string_for_latex("Frequency [THz]"),
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

    def plot_vline(self, ax, idx:int):
        """
        Plot segment band.

        Args:
            ax: Axes.
            idx (int): Segmet index.
        """
        labels = [ ld[0] for ld in self._segment_labels_distances[idx] ]
        distances = [ ld[1] for ld in self._segment_labels_distances[idx] ]
        for distance in distances:
            ax.axvline(distance, c='grey', linestyle='--', linewidth=0.5)
        ax.set_xticks(distances)
        decorated_labels = [ decorate_string_for_latex(label)
                                 for label in labels ]
        ax.set_xticklabels(decorated_labels, fontsize=20)

    def plot_hline(self, ax, hval:float=0.):
        """
        Plot horizontal line.
        """
        ax.axhline(hval, c='grey', linestyle='--', linewidth=0.5)

    def plot_band_structure(self,
                            figsize=(8,6),
                            dosplot=None,
                            dos_distance=0.3,
                            c='r'):
        """
        Plot band structure.
        """
        distances = deepcopy(self._axes_distances)
        if dosplot is not None:
            distances.append(dos_distance)
        fig, axes = create_figure_axes(figsize=figsize,
                                       ratios=distances,
                                       axes_pad=0.03)

        for i in range(len((self._segment_frequences))):
            if i == 0:
                show_yscale = True
                self.plot_ylabel(axes[i])
            else:
                show_yscale = False

            self.plot_segment_band_structure(
                    c=c,
                    ax=axes[i],
                    frequences=self._segment_frequences[i],
                    distances=self._segment_distances[i],
                    show_yscale=show_yscale,
                    )
            self.plot_vline(axes[i], i)
            self.plot_hline(axes[i])
            axes[i].set_ylim(self._ylim)

        if dosplot is not None:
            dosplot.plot_total_dos(ax=axes[-1], c=c)
            axes[-1].tick_params(labelleft=False)
            axes[-1].set_ylim(self._ylim)
            self.plot_hline(axes[-1])

        return fig, axes


class BandsPlot():
    """
    Band structure plot class.
    """
    def __init__(self,
                 band_structures:list,
                 ):
        """
        Init.

        Args:
            band_structures (list): List of BandStructure class object.
        """
        self._band_structures = band_structures
        self._bandplots = [ BandPlot(band_structure)
                                for band_structure in self._band_structures ]
        self._axes_distances = deepcopy(self._bandplots[0].axes_distances)
        self._ylim = None
        self._min_frequency = None
        self._max_frequency = None
        self.set_ylim(ymin=None, ymax=None)
        self._cs = None
        self._alphas = None
        self._linewidths = None
        self._linestyles = None
        self.set_line_properties(base_color='r')

    def set_line_properties(self, base_color:str='r'):
        """
        Set line properties
        """
        self._cs, self._alphas, self._linewidths, self._linestyles = \
                get_plot_properties_for_trajectory(
                        plot_nums=len(self._bandplots),
                        base_color=base_color)

    def set_cs(self, cs):
        """
        cs
        """
        self._cs = cs

    def set_alphas(self, alphas):
        """
        cs
        """
        self._alphas = alphas

    def set_linewidths(self, linewidths):
        """
        cs
        """
        self._linewidths = linewidths

    def set_linestyles(self, linestyles):
        """
        cs
        """
        self._linestyles = linestyles

    # def set_ylim(self, ax, ymin:float=None, ymax:float=None):
    def set_ylim(self, ymin:float=None, ymax:float=None):
        """
        Set ylim.

        Args:
            ymin (float): If None, min(frequences) -
                          (max(frequences) - min(frequences)) * 1.05 is set.
            ymax (float): If None, max(frequences) +
                          (max(frequences) - min(frequences)) * 1.05 is set.
        """
        freq_min = min([ bandplot.min_frequency
                             for bandplot in self._bandplots ])
        freq_max = max([ bandplot.max_frequency
                             for bandplot in self._bandplots ])
        span = freq_max - freq_min
        if ymin is None:
            _ymin = freq_min - span * 0.05
        else:
            _ymin = ymin
        if ymax is None:
            _ymax = freq_max + span * 0.05
        else:
            _ymax = ymax
        self._min_frequency = freq_min
        self._max_frequency = freq_max
        self._ylim = (_ymin, _ymax)
        # ax.set_ylim(self._ylim)

    @property
    def band_structures(self):
        """
        Band structures.
        """
        return self._band_structures

    @property
    def bandplots(self):
        """
        List of BandsPlot class object.
        """
        return self._bandplots

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

    def plot_band_structures(self,
                             figsize=(8,6),
                             dosesplot=None,
                             dos_distance=0.3):
        """
        Plot band structures.

        Returns:
            fig: Figure.
            axes (list): Axes.
        """
        distances = deepcopy(self._axes_distances)
        if dosesplot is not None:
            distances.append(dos_distance)
        fig, axes = create_figure_axes(figsize=figsize,
                                       ratios=distances,
                                       axes_pad=0.03)

        for j, bandplot in enumerate(self._bandplots):
            for i in range(len((self._bandplots[0].segment_frequences))):
                if i == 0 and j == 0:
                    label = 'Initial'
                elif i == 0 and j == len(self._bandplots)-1:
                    label = 'Final'
                else:
                    label = None

                show_yscale = bool(i==0)
                bandplot.plot_segment_band_structure(
                        ax=axes[i],
                        frequences=bandplot.segment_frequences[i],
                        distances=bandplot.segment_distances[i],
                        show_yscale=show_yscale,
                        c=self._cs[j],
                        alpha=self._alphas[j],
                        linestyle=self._linestyles[j],
                        linewidth=self._linewidths[j],
                        label=label,
                        )
                if j == 0:
                    bandplot.plot_vline(axes[i], i)
                    bandplot.plot_hline(axes[i])
                    axes[i].set_ylim(self._ylim)

        if dosesplot is not None:
            dosesplot.set_cs(self._cs)
            dosesplot.set_alphas(self._alphas)
            dosesplot.set_linewidths(self._linewidths)
            dosesplot.set_linestyles(self._linestyles)
            dosesplot.plot_total_doses(ax=axes[-1])
            axes[-1].tick_params(labelleft=False)
            axes[-1].set_ylim(self._ylim)
            axes[-1].set_ylabel('')

        return fig, axes
