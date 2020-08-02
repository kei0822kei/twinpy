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
from phonopy.phonon.dos import TotalDos as PhonopyTotalDos
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.cm as cm
from twinpy.lattice.lattice import Lattice
from twinpy.interfaces.phonopy import get_cell_from_phonopy_structure
from twinpy.plot.dos import total_doses_plot


def get_labels_for_twin() -> dict:
    """
    Get labels for hexagonal twin.
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


def get_segment_qpoints_from_labels(labels):
    """
    Get segment qpoints which is input for phonopy.


    Returns:
        list: label lists for phonopy
        np.array: segment qpoints

    Examples:
        >>> labels = ['GAMMA', 'M_1', '', 'K_1', 'GAMMA']
        >>> get_segment_qpoints_from_labels(labels)
            [[[  0,   0, 0],
              [1/2,   0, 0]],
             [[1/3, 1/3, 0],
              [  0,   0, 0]]]
    """
    all_labels = get_labels_for_twin()
    seg_qpt = []
    qpt = []
    lb = []
    for label in labels:
        if label == '':
            seg_qpt.append(qpt)
            qpt = []
        else:
            qpt.append(all_labels[label])
            lb.append(label)
    seg_qpt.append(qpt)
    return lb, np.array(seg_qpt)


def decorate_string_for_latex(string):
    """
    decorate strings for latex
    """
    if string == 'GAMMA':
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


def run_band_calc(phonon,
                  band_labels=None,
                  segment_qpoints=None,
                  npoints=51):
    qpoints, path_connections = get_band_qpoints_and_path_connections(
            segment_qpoints, npoints=npoints,
            rec_lattice=np.linalg.inv(phonon.get_primitive().cell))
    phonon.run_band_structure(paths=qpoints,
                              with_eigenvectors=True,
                              with_group_velocities=False,
                              is_band_connection=False,
                              path_connections=path_connections,
                              labels=band_labels,
                              is_legacy_plot=False)


class BandsPlot(PhonopyBandPlot):
    """
    Band structure plot class.
    """
    def __init__(self,
                 fig,
                 phonons,
                 transformation_matrices,
                 band_labels,
                 segment_qpoints,
                 xscale=20,
                 npoints=51,
                 with_dos=False,
                 mesh=None,
                 ):
        """
        Multiple band plot.

        Args:
            transformation_matrices (list): list of crystal body rotation matrix.
        """
        self.fig = fig
        self.phonons = deepcopy(phonons)
        self.with_dos = with_dos
        self.mesh = mesh
        self.xscale = xscale

        self.transformation_matrices = transformation_matrices
        self.segment_qpoints = segment_qpoints

        self.band_labels = None
        self.connections = None
        self.npoints = npoints
        self._run_band(band_labels,
                       segment_qpoints,
                       self.npoints)

        self.axs = None
        self._set_axs()
        super().__init__(axs=self.axs)
        self._set_frame()

    def _set_axs(self):
        """
        Set axs.
        """
        n = len([x for x in self.phonons[0].band_structure.path_connections
                    if not x])
        if self.with_dos:
            n += 1
        self.axs = ImageGrid(self.fig, 111,  # similar to subplot(111)
                             nrows_ncols=(1, n),  # n is the number of figures
                             axes_pad=0.11,   # pad between axes in inch.
                             add_all=True,
                             label_mode="L")

    def _set_frame(self):
        """
        Set frame.
        """
        self.decorate(self.band_labels,
                      self.connections,
                      self.phonons[0].band_structure.get_frequencies(),
                      self.phonons[0].band_structure.get_distances())

    def decorate(self, labels, path_connections, frequencies, distances):
        """

        labels : List of str, optional
            Labels of special points.
            See the detail in docstring of Phonopy.run_band_structure.

        """

        if self._decorated:
            raise RuntimeError("Already BandPlot instance is decorated.")
        else:
            self._decorated = True

        if self.xscale is None:
            self.set_xscale_from_data(frequencies, distances)

        distances_scaled = [d * self.xscale for d in distances]

        # T T T F F -> [[0, 3], [4, 4]]
        lefts = [0]
        rights = []
        for i, c in enumerate(path_connections):
            if not c:
                lefts.append(i + 1)
                rights.append(i)
        seg_indices = [list(range(l, r + 1)) for l, r in zip(lefts, rights)]
        special_points = []
        for indices in seg_indices:
            pts = [distances_scaled[i][0] for i in indices]
            pts.append(distances_scaled[indices[-1]][-1])
            special_points.append(pts)

        self._axs[0].set_ylabel('Frequency (THz)')
        l_count = 0
        for ax, spts in zip(self._axs, special_points):
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_tick_params(which='both', direction='in')
            ax.yaxis.set_tick_params(which='both', direction='in')
            ax.set_xlim(spts[0], spts[-1])
            ax.set_xticks(spts)
            if labels is None:
                ax.set_xticklabels(['', ] * len(spts))
            else:
                ax.set_xticklabels(labels[l_count:(l_count + len(spts))])
                l_count += len(spts)
            ax.plot([spts[0], spts[-1]], [0, 0],
                    linestyle=':', linewidth=0.5, color='b')

    def _fix_segment_qpoints(
            self,
            base_transformation_matrix,
            transformation_matrix,
            ):
        """
        Fix segment qpoints.
        """
        seg_frac = []
        print("fuga")
        for qpt in self.segment_qpoints:
            qpt_frac = np.dot(np.linalg.inv(transformation_matrix.T),
                              np.dot(base_transformation_matrix.T,
                                     qpt.T)).T
            # qpt_frac = np.dot(transformation_matrix,
            #                   np.dot(np.linalg.inv(base_transformation_matrix),
            #                          qpt.T)).T
            seg_frac.append(qpt_frac)
        return seg_frac

    def _run_band(self,
                  band_labels,
                  segment_qpoints,
                  npoints):
        for i, phonon in enumerate(self.phonons):
            fixed_segment_qpoints = self._fix_segment_qpoints(
                    base_transformation_matrix=self.transformation_matrices[0],
                    transformation_matrix=self.transformation_matrices[i],
                    )
            run_band_calc(phonon=phonon,
                          band_labels=band_labels,
                          segment_qpoints=fixed_segment_qpoints,
                          npoints=npoints)

        self.band_labels = [ decorate_string_for_latex(label) for label in band_labels ]
        self.connections = self.phonons[0].band_structure.path_connections

    def _revise_distances(self, distances, base_distances):
        segment_lengths = []
        for ds in [distances, base_distances]:
            lengths = []
            init = 0
            for d in ds:
                lengths.append(d[-1]-init)
                init = d[-1]
            segment_lengths.append(lengths)
        ratios = np.array(segment_lengths)[0] /  np.array(segment_lengths)[1]
        revised = []
        seg_start = 0
        for i, distance in enumerate(distances):
            if i == 0:
                revised.append(distance / ratios[i])
            else:
                revised.append(seg_start+(distance-distances[i-1][-1]) / ratios[i])
            seg_start = revised[-1][-1]
        return revised

    def plot_bands(self, cs=None, alphas=None, linestyles=None, linewidths=None, labels=None):
        """
        plot band, kwargs is passed for plotting with matplotlib

        Note:
            currently suppored kwargs
              - 'cs'
              - 'alphas'
              - 'linestyles'
              - 'linewidths'
              - 'labels'
        """
        def _plot(distances, frequencies, connections, is_decorate,
                  c, alpha, linestyle, linewidth, label):
            count = 0
            distances_scaled = [d * self.xscale for d in distances]
            for d, f, cn in zip(distances_scaled,
                                frequencies,
                                connections):
                ax = self.axs[count]
                ax.plot(d, f, c=c, alpha=alpha, linestyle=linestyle,
                             linewidth=linewidth, label=label)
                if is_decorate:
                    ax.axvline(d[-1], c='k', linestyle='dotted', linewidth=0.5)
                if not cn:
                    count += 1

        if cs is None:
            cs = [ DEFAULT_COLORS[i%len(DEFAULT_COLORS)] for i in range(len(self.phonons)) ]
        if alphas is None:
            alphas = [ 1. ] * len(self.phonons)
        if linestyles is None:
            linestyles = [ 'solid' ] * len(self.phonons)
        if linewidths is None:
            linewidths = [ 1. ] * len(self.phonons)
        if labels is None:
            labels = [ None ] * len(self.phonons)

        for i, phonon in enumerate(self.phonons):
            distances = phonon.band_structure.get_distances()
            frequencies = phonon.band_structure.get_frequencies()
            if i == 0:
                _plot(distances=distances,
                      frequencies=frequencies,
                      connections=self.connections,
                      is_decorate=True,
                      c=cs[i],
                      alpha=alphas[i],
                      linestyle=linestyles[i],
                      linewidth=linewidths[i],
                      label=labels[i])
                base_distances = deepcopy(distances)
            else:
                distances = self._revise_distances(distances, base_distances)
                _plot(distances=distances,
                      frequencies=frequencies,
                      connections=self.connections,
                      is_decorate=False,
                      c=cs[i],
                      alpha=alphas[i],
                      linestyle=linestyles[i],
                      linewidth=linewidths[i],
                      label=labels[i])

            if self.with_dos:
                if i == 0:
                    total_doses_plot(ax=self.axs[-1],
                                     phonons=self.phonons,
                                     mesh=self.mesh,
                                     cs=cs,
                                     alphas=alphas,
                                     linewidths=linewidths,
                                     linestyles=linestyles,
                                     flip_xy=True,
                                     draw_grid=False,
                                     labels=labels,
                                     )
                    xlim = self.axs[-1].get_xlim()
                    ylim = self.axs[-1].get_ylim()
                    aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) * 3
                    self.axs[-1].set_aspect(aspect)
                    self.axs[-1].axhline(y=0, linestyle=':', linewidth=0.5, color='b')
                    self.axs[-1].set_xlim((0, None))
                else:
                    total_doses_plot(ax=self.axs[-1],
                                     phonons=self.phonons,
                                     mesh=self.mesh,
                                     cs=cs,
                                     alphas=alphas,
                                     linewidths=linewidths,
                                     linestyles=linestyles,
                                     flip_xy=True,
                                     draw_grid=False,
                                     labels=None,
                                     )
        self.axs[-1].legend()


def bands_plot(fig,
               phonons:list,
               transformation_matrices:list,
               band_labels,
               segment_qpoints,
               xscale=20,
               npoints=51,
               with_dos=False,
               mesh=None,
               cs=None,
               alphas=None,
               linewidths=None,
               linestyles=None,
               labels=None,
               ):
    """
    Bands plot one runner.
    """
    bp = BandsPlot(
            fig=fig,
            phonons=phonons,
            transformation_matrices=transformation_matrices,
            band_labels=band_labels,
            segment_qpoints=segment_qpoints,
            xscale=xscale,
            npoints=npoints,
            with_dos=with_dos,
            mesh=mesh,
            )
    bp.plot_bands(cs=cs,
                  alphas=alphas,
                  linestyles=linestyles,
                  linewidths=linewidths,
                  labels=labels)
