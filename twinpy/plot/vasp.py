#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plot
----
provide various kinds of plot
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
from twinpy.interfaces.phonopy import get_cell_from_phonopy_structure


def shift_energy_plot(ax, shifts, energies, natoms, a, b, ev_range=4):
    from scipy import stats
    from scipy import interpolate
    from mpl_toolkits.mplot3d import Axes3D
    ax.set_aspect('equal')
    # ene_atom = energies / natoms
    energies = energies
    shifts = np.array(shifts)
    shifts = np.where(shifts>0.5, shifts-1, shifts)
    xyz = np.hstack((shifts, energies.reshape(energies.shape[0],1)))
    x1 = xyz[np.isclose(xyz[:,0], 0.5)] + np.array([-1,0,0])
    y1 = xyz[np.isclose(xyz[:,1], 0.5)] + np.array([0,-1,0])
    xy1 = x1[np.isclose(x1[:,1], 0.5)] + np.array([0,-1,0])
    full_xyz = np.vstack((xyz, x1, y1, xy1))
    sort_ix = np.argsort((len(np.unique(shifts[:,0]))+1)*full_xyz[:,0] + full_xyz[:,1])
    sort_xyz = full_xyz[sort_ix]

    xy = sort_xyz[:,:2]
    z = sort_xyz[:,2]

    x = y = np.linspace(-0.5, 0.5, 500)
    X, Y = np.meshgrid(x, y)

    # i_Z = interpolate.griddata(xy*np.array([a,b]), z, (X*a, Y*b), method='cubic')
    i_Z = interpolate.griddata(xy*np.array([a,b]), z, (X*a, Y*b), method='linear')

    # plot interpolation Z
    # im = ax.pcolormesh(X*a, Y*b, i_Z, cmap="jet_r", vmax=min(min(energies)*0.85, max(energies)))
    # im = ax.pcolormesh(Y*b, X*a, i_Z, cmap="jet_r", vmin=min(energies), vmax=min(energies)+1)
    # im = ax.pcolormesh(Y*b, X*a, i_Z, cmap="jet_r", vmin=min(energies), vmax=min(energies)*0.95)
    im = ax.pcolormesh(Y*b, X*a, i_Z, cmap="jet_r", vmin=min(energies), vmax=min(energies)+ev_range)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    # im = ax.imshow(Z, interpolation='none')
    # plt.colorbar(im, ax=ax, fraction=0.20, label='energy per atom [eV/atom]',)
    plt.colorbar(im, ax=ax, cax=cax, label='total energy [eV]')
    # ax.scatter(xy[:,0]*a, xy[:,1]*b, c='k')
    ax.scatter(xy[:,1]*b, xy[:,0]*a, c='k')
    for i in np.unique(shifts[:,1]):
        ax.axvline(i*b, c='k', linestyle='--')
    for i in np.unique(shifts[:,0]):
        ax.axhline(i*a, c='k', linestyle='--')
    ax.set_title("shift energy")
    ax.set_xlabel("y shift [angstrom]")
    ax.set_ylabel("x shift [angstrom]")
