#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
provide various kinds of plot
"""

import numpy as np
from matplotlib import pyplot as plt

DEFAULT_COLORS = ['r', 'b', 'm', 'y', 'g', 'c']
DEFAULT_COLORS.extend(plt.rcParams['axes.prop_cycle'].by_key()['color'])
DEFAULT_MARKERS = ['o', 'v', ',', '^', 'h', 'D', '<', '*', '>', 'd']


def line_chart(ax,
               xdata:np.array,
               ydata:np.array,
               xlabel:str,
               ylabel:str,
               label:str=None,
               alpha=1.,
               **kwargs):
    """
    Plot line chart in ax.

    Args:
        kwargs: c, marker, facecolor
    """
    if 'c' in kwargs.keys():
        c = kwargs['c']
    else:
        c_num = len(ax.get_lines()) % len(DEFAULT_COLORS)
        c = DEFAULT_COLORS[c_num]

    if 'marker' in kwargs.keys():
        marker = kwargs['marker']
    else:
        marker_num = len(ax.get_lines()) % len(DEFAULT_MARKERS)
        marker = DEFAULT_MARKERS[marker_num]

    if 'facecolor' in kwargs.keys():
        facecolor = kwargs['facecolor']
    else:
        facecolor_num = len(ax.get_lines()) % 2
        if facecolor_num == 0:
            facecolor = 'white'
        else:
            facecolor = c

    raw = np.array([xdata, ydata])
    idx = np.array(xdata).argsort()
    sort = raw[:,idx]
    ax.plot(sort[0,:], sort[1,:], linestyle='--', linewidth=0.5, c=c,
            alpha=alpha, label=label)
    ax.scatter(sort[0,:], sort[1,:], facecolor=facecolor, marker=marker,
               edgecolor=c, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def line_chart_group(ax,
                     xdata:np.array,
                     ydata:np.array,
                     gdata:np.array,
                     xlabel:str,
                     ylabel:str,
                     glabel:str,
                     **kwargs):
    """
    Plot group line chart in ax.

    Args:
        kwargs: see line_chart

    Note:
        This function finds the unique value sets in gdata and make groups.
    """
    uniques = np.unique(gdata)
    for unique in uniques:
        idxes = [ idx for idx in range(len(gdata))
                      if np.isclose(gdata[idx], unique) ]
        label = '{}: {}'.format(glabel, unique)
        line_chart(ax=ax,
                   xdata=np.array(xdata)[idxes],
                   ydata=np.array(ydata)[idxes],
                   xlabel=xlabel,
                   ylabel=ylabel,
                   label=label,
                   **kwargs)
    ax.legend()


def line_chart_group_trajectory(ax,
                                xdata:np.array,
                                ydata:np.array,
                                gdata:np.array,
                                xlabel:str,
                                ylabel:str,
                                glabel:str,
                                tdata=None,
                                **kwargs):
    """
    Plot group line chart in ax with trajectory.

    Args:
        kwargs: see line_chart

    Note:
        This function finds the unique value sets in gdata and make groups.
    """
    uniques_ = np.unique(gdata)
    for j, unique_ in enumerate(uniques_):
        c = DEFAULT_COLORS[j%len(DEFAULT_COLORS)]
        uniques = np.unique(tdata)
        minimum = 0.3
        alphas = [ minimum+(1.-minimum)/(len(uniques)-1)*i
                   for i in range(len(uniques)) ]
        for i, unique in enumerate(uniques):
            marker = DEFAULT_MARKERS[i]
            idxes = [ idx for idx in range(len(gdata))
                          if np.isclose(gdata[idx], unique_)
                              and np.isclose(tdata[idx], unique) ]
            if i == len(uniques)-1:
                label = '{}: {}'.format(glabel, unique_)
                line_chart(ax=ax,
                           xdata=np.array(xdata)[idxes],
                           ydata=np.array(ydata)[idxes],
                           xlabel=xlabel,
                           ylabel=ylabel,
                           label=label,
                           alpha=alphas[i],
                           c=c,
                           marker=marker,
                           **kwargs)
            else:
                line_chart(ax=ax,
                           xdata=np.array(xdata)[idxes],
                           ydata=np.array(ydata)[idxes],
                           xlabel=xlabel,
                           ylabel=ylabel,
                           alpha=alphas[i],
                           c=c,
                           marker=marker,
                           **kwargs)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0, fontsize=12)


def get_plot_properties_for_trajectory(plot_nums:int,
                                       base_color:str='r') -> tuple:
    """
    Get plot properties for trajectory.

    Args:
        plot_nums (int): the number of plots
        base_color (str): base color

    Returns:
        tuple: (cs, alphas, linewidths, linestyles)
    """
    alphas = [ 1. ]
    linewidths = [ 1.5 ]
    linestyles = [ 'dashed' ]
    alphas.extend([ 0.6 for _ in range(plot_nums-2) ])
    linewidths.extend([ 1. for _ in range(plot_nums-2) ])
    linestyles.extend([ 'dotted' for _ in range(plot_nums-2) ])
    alphas.append(1.)
    linewidths.append(1.5)
    linestyles.append('solid')
    cs = [ base_color for _ in range(plot_nums) ]
    return (cs, alphas, linewidths, linestyles)
