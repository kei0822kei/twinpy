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
    def _get_data(get_additional_relax):
        c_norm = twinboundary_relax.structures['twinboundary_original'][0][2,2]
        distances = twinboundary_relax.get_distances(
                is_fractional=is_fractional,
                get_additional_relax=get_additional_relax)
        planes = twinboundary_relax.get_planes(
                is_fractional=is_fractional,
                get_additional_relax=get_additional_relax)
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
        return (rlx_distances, ydata, z_coords, bulk_interval)

    if twinboundary_relax.additional_relax_pks is not None:
        datas = [ _get_data(bl) for bl in [False, True] ]
        labels = ['isif7', 'isif3']
    else:
        datas = [ _get_data(False) ]
        labels = ['isif7']
        xmax = max(datas[0][0])
        xmin = min(datas[0][0])
    for i in range(len(datas)):
        xdata, ydata, z_coords, bulk_interval = datas[i]
        line_chart(ax=ax,
                   xdata=xdata,
                   ydata=ydata,
                   xlabel='distance',
                   ylabel='z coords',
                   sort_by='y',
                   label=labels[i])

    if is_decorate:
        num = len(z_coords)
        tb_idx = [1, int(num/2), num-1]
        xmax = max([ max(data[0]) for data in datas ])
        xmin = min([ min(data[0]) for data in datas ])
        ymax = max([ max(data[1]) for data in datas ])
        ymin = min([ min(data[1]) for data in datas ])
        for idx in tb_idx:
            ax.hlines(z_coords[idx],
                      xmin=xmin-0.005,
                      xmax=xmax+0.005,
                      linestyle='--',
                      linewidth=1.5)
        yrange = ymax - ymin
        if is_fractional:
            c_norm = twinboundary_relax.structures['twinboundary_original'][0][2,2]
            vline_x = bulk_interval * c_norm
        else:
            vline_x = bulk_interval
        ax.vlines(vline_x,
                  ymin=ymin-yrange*0.01,
                  ymax=ymax+yrange*0.01,
                  linestyle='--',
                  linewidth=0.5)
        ax.legend()
