#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Twinboundary plot

This module provide various kinds of plot related to twin boudnary.
"""

import numpy as np
from phonopy.phonon.dos import TotalDos as PhonopyTotalDos
from twinpy.plot.base import line_chart
from twinpy.interfaces.aiida import AiidaTwinBoudnaryRelaxWorkChain


def plane_diff(ax,
               twinboundary_relax:AiidaTwinBoudnaryRelaxWorkChain,
               is_fractional:bool=False,
               is_decorate:bool=True):
    """
    Plot plane diff.

    Args:
        ax: matplotlib ax
        twinboundary_relax: AiidaTwinBoudnaryRelaxWorkChain object
        is_fractional (bool): if True, z coords with fractional coordinate
        is_decorate (bool): if True, decorate figure
    """
    c_norm = twinboundary_relax.structures['twinboundary_original'][0][2,2]
    distances = twinboundary_relax.get_distances(is_fractional=is_fractional)
    planes = twinboundary_relax.get_planes(is_fractional=is_fractional)
    before_distances = distances['before'].copy()
    bulk_interval = before_distances[1]
    rlx_distances = distances['relax'].copy()
    z_coords = planes['before'].copy()
    z_coords.insert(0, z_coords[0] - before_distances[-1])
    z_coords.append(z_coords[-1] + before_distances[0])
    rlx_distances.insert(0, rlx_distances[-1])
    rlx_distances.append(rlx_distances[0])
    ydata = np.array(z_coords) + bulk_interval / 2
    if is_fractional:
        rlx_distances = np.array(rlx_distances) * c_norm
    line_chart(ax=ax,
               xdata=rlx_distances,
               ydata=ydata,
               xlabel='distance',
               ylabel='z coords',
               sort_by='y')

    if is_decorate:
        num = len(z_coords)
        tb_idx = [1, int(num/2), num-1]
        for idx in tb_idx:
            ax.hlines(z_coords[idx],
                      xmin=min(rlx_distances)-0.005,
                      xmax=max(rlx_distances)+0.005,
                      linestyle='--',
                      linewidth=1.5)
        ymax = max(ydata)
        ymin = min(ydata)
        yrange = ymax - ymin
        if is_fractional:
            vline_x = bulk_interval * c_norm
        else:
            vline_x = bulk_interval
        ax.vlines(vline_x,
                  ymin=ymin-yrange*0.01,
                  ymax=ymax+yrange*0.01,
                  linestyle='--',
                  linewidth=0.5)
