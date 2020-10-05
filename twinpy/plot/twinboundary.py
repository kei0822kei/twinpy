#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Twinboundary plot

This module provide various kinds of plot related to twin boudnary.
"""

import numpy as np
from copy import deepcopy
from twinpy.plot.base import line_chart


def plot_plane(ax,
               distances:list,
               z_coords:list,
               label:str=None,
               decorate:bool=True,
               **kwargs):
    """
    Plot plane.
    """
    if decorate:
        xlabel = 'Distance'
        ylabel = 'Hight'
    else:
        xlabel = ylabel = None

    _distances = deepcopy(distances)
    _distances.insert(0, distances[-1])
    _distances.append(distances[0])
    _z_coords = deepcopy(z_coords)
    _z_coords.insert(0, -distances[-1])
    _z_coords.append(z_coords[-1]+distances[0])

    fixed_z_coords = _z_coords + np.array(_distances) / 2

    line_chart(ax=ax,
               xdata=_distances,
               ydata=fixed_z_coords,
               xlabel=xlabel,
               ylabel=ylabel,
               label=label,
               sort_by='y',
               **kwargs)



    if decorate:
        num = len(_z_coords)
        tb_idx = [1, int(num/2), num-1]
        bulk_distance = _distances[int(num/4)]
        xmin = bulk_distance - 0.025
        xmax = bulk_distance + 0.025
        for idx in tb_idx:
            ax.hlines(_z_coords[idx],
                      xmin=xmin-0.005,
                      xmax=xmax+0.005,
                      linestyle='--',
                      linewidth=1.5)


def plot_angle(ax,
               angles:list,
               z_coords:list,
               label:str=None,
               decorate:bool=True):
    """
    Plot angle.
    """
    if decorate:
        xlabel = 'Angle'
        ylabel = 'Hight'
    else:
        xlabel = ylabel = None

    _angles = deepcopy(angles)
    _z_coords = deepcopy(z_coords)
    _angles.append(angles[0])
    _z_coords.append(z_coords[-1]+z_coords[1])

    line_chart(ax=ax,
               xdata=_angles,
               ydata=_z_coords,
               xlabel=xlabel,
               ylabel=ylabel,
               label=label,
               sort_by='y')

    if decorate:
        num = len(_z_coords)
        tb_idx = [0, int(num/2), num-1]
        bulk_angle = angles[int(num/4)]
        for idx in tb_idx:
            ax.hlines(_z_coords[idx],
                      xmin=-1,
                      xmax=bulk_angle+2,
                      linestyle='--',
                      linewidth=1.5)
