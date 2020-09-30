#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Band plot.
"""

import numpy as np
from copy import deepcopy
import spglib
from matplotlib import pyplot as plt
import mpl_toolkits.axes_grid1
from phonopy.phonon.band_structure import BandPlot as PhonopyBandPlot
from phonopy.phonon.band_structure import BandStructure
from phonopy.phonon.dos import TotalDos as PhonopyTotalDos
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.cm as cm
import seekpath
from twinpy.lattice.lattice import Lattice
from twinpy.interfaces.phonopy import get_cell_from_phonopy_structure
from twinpy.plot.base import create_figure_axes, get_plot_properties_for_trajectory
from twinpy.plot.dos import total_doses_plot


def get_labels_for_twin() -> dict:
    """
    Get labels for hexagonal twin.

    Returns:
        dict: Contain keys and qpoints.

    Examples:
        Labels and qpoints for twin is as bellow.

        >>> label_qpoints = {
                'GAMMA': [0, 0, 0],
                'M_1'  : [1/2, 0, 0],
                'M_2'  : [-1/2, 1/2, 0],
                'K_1'  : [1/3, 1/3, 0],
                'K_2'  : [-1/3, 2/3, 0],
                'A'    : [0, 0, 1/2],
                'L_1'  : [1/2, 0, 1/2],
                'L_2'  : [-1/2, 1/2, 1/2],
                'H_1'  : [1/3, 1/3, 1/2],
                'H_2'  : [-1/3, 2/3, 1/2],
                }
    """
    label_qpoints = {
        'GAMMA': [0, 0, 0],
        'M_1'  : [1/2, 0, 0],
        'M_2'  : [-1/2, 1/2, 0],
        'K_1'  : [1/3, 1/3, 0],
        'K_2'  : [-1/3, 2/3, 0],
        'A'    : [0, 0, 1/2],
        'L_1'  : [1/2, 0, 1/2],
        'L_2'  : [-1/2, 1/2, 1/2],
        'H_1'  : [1/3, 1/3, 1/2],
        'H_2'  : [-1/3, 2/3, 1/2],
    }
    return label_qpoints


def get_seekpath(cell:tuple) -> dict:
    """
    Get seekpath results.

    Args:
        cell (tuple): Cell.

    Returns:
        dict: Labels and corresponding qpoints.
    """
    if type(cell[2][0]) is str:
        from twinpy.structure.base import get_numbers_from_symbols
        numbers = get_numbers_from_symbols(cell[2])
        _cell = (cell[0], cell[1], numbers)
    else:
        _cell = cell
    skp = seekpath.get_path(_cell)
    return skp


def get_labels_band_paths_from_seekpath(cell:tuple):
    """
    Get labels and band paths from seekpath result.

    Args:
        cell (tuple): Cell.

    Returns:
        list: labels
        np.array: band paths
    """
    skp = get_seekpath(cell)
    paths = skp['path']
    labels = []
    for path in paths:
        try:
            if labels[-1] != path[0]:
                labels.extend(['', path[0]])
        except IndexError:
            labels.append(path[0])
        labels.append(path[1])
    labels_qpoints = skp['point_coords']
    lb, band_paths = get_band_paths_from_labels(
            labels=labels,
            labels_qpoints=labels_qpoints)
    return lb, band_paths


def get_band_paths_from_labels(labels:list=None,
                               labels_qpoints:dict=None):
    """
    Get segment qpoints which is input for phonopy.

    Args:
        labels (list): List of labels. If you want to separate band structure,
                       add '' between labels. See Examples.
        labels_qpoints (dict): Dictionary for labels and corresponding qpoints.

    Returns:
        list: labels
        np.array: band paths

    Examples:
        >>> labels = ['GAMMA', 'M_1', '', 'K_1', 'GAMMA']
        >>> get_segment_qpoints_from_labels(labels)
            [[[  0,   0, 0],
              [1/2,   0, 0]],
             [[1/3, 1/3, 0],
              [  0,   0, 0]]]
    """
    seg_qpt = []
    qpt = []
    lb = []
    for label in labels:
        if label == '':
            seg_qpt.append(qpt)
            qpt = []
        else:
            qpt.append(labels_qpoints[label])
            lb.append(label)
    seg_qpt.append(qpt)
    return lb, seg_qpt


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
        else:
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
            labels = [''] * (len(path_connections) * 2 - path_connections.count(True))
        else:
            labels = self._labels

        seg_freqs = []
        seg_dis = []
        seg_labels_qpoints = []
        freqs = []
        dis = []
        iter_labels = iter(labels)
        labels_qpoints = []
        for i in range(len(path_connections)):
            if labels_qpoints == []:
                labels_qpoints = [(iter_labels.__next__(), 0.)]
            freqs.append(frequences[i])
            dis.extend(distances[i])
            labels_qpoints.append((iter_labels.__next__(), dis[-1]-dis[0]))
            if path_connections[i]:
                continue
            else:
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
                                    labels:list=None, c='r', linestyle='-',
                                    alpha=1., linewidth=1.5,
                                    show_yscale:bool=True):
        """
        Plot segment band structure.
        """
        for i in range(frequences.shape[1]):
            ax.plot(distances, frequences[:,i], c=c, linestyle=linestyle,
                    alpha=alpha, linewidth=linewidth)
        ax.set_xlim(min(distances), max(distances))
        if show_yscale:
            labelleft = True
            ax.set_ylabel(decorate_string_for_latex("Frequency [THz]"),
                          fontsize=20)
        else:
            labelleft = False
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(labelbottom=True,
                       labelleft=labelleft,
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
        for label, distance in zip(labels, distances):
            ax.axvline(distance, c='grey', linestyle='--', linewidth=0.5)
        ax.set_xticks(distances)
        decorated_labels = [ decorate_string_for_latex(label)
                                 for label in labels ]
        ax.set_xticklabels(decorated_labels, fontsize=16)

    def plot_hline(self, ax, hval:float=0.):
        """
        Plot horizontal line.
        """
        ax.axhline(hval, c='b', linestyle='--', linewidth=0.5)

    def plot_band_structure(self):
        """
        Plot band structure.
        """
        fig, axes = create_figure_axes(ratios=self._axes_distances,
                                       axes_pad=0.03)
        for i, ax in enumerate(axes):
            if i == 0:
                show_yscale = True
            else:
                show_yscale = False
            self.plot_segment_band_structure(
                    ax=ax,
                    frequences=self._segment_frequences[i],
                    distances=self._segment_distances[i],
                    show_yscale=show_yscale,
                    )
            self.plot_vline(ax, i)
            self.plot_hline(ax)
            ax.set_ylim(self._ylim)
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

    def set_ylim(self, ymin:float=None, ymax:float=None):
        """
        Set ylim.

        Args:
            ymin (float): If None, min(frequences) -
                          (max(frequences) - min(frequences)) * 1.05 is set.
            ymax (float): If None, max(frequences) +
                          (max(frequences) - min(frequences)) * 1.05 is set.
        """
        freq_min = min([ bandplot.min_frequency for bandplot in self._bandplots ])
        freq_max = max([ bandplot.max_frequency for bandplot in self._bandplots ])
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

    def plot_band_structures(self):
        """
        Plot band structures.

        Returns:
            fig: Figure.
            axes (list): Axes.
        """
        fig, axes = create_figure_axes(ratios=self._axes_distances,
                                       axes_pad=0.03)
        for j, bandplot in enumerate(self._bandplots):
            for i, ax in enumerate(axes):
                if i == 0:
                    show_yscale = True
                else:
                    show_yscale = False
                bandplot.plot_segment_band_structure(
                        ax=ax,
                        frequences=bandplot.segment_frequences[i],
                        distances=bandplot.segment_distances[i],
                        show_yscale=show_yscale,
                        c=self._cs[j],
                        alpha=self._alphas[j],
                        linestyle=self._linestyles[j],
                        linewidth=self._linewidths[j],
                        )
                if j == 0:
                    bandplot.plot_vline(ax, i)
                    bandplot.plot_hline(ax)
                    ax.set_ylim(self._ylim)
        return fig, axes
