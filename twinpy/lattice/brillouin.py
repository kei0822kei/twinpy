#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides brillouin zone plottling.
"""
import numpy as np
from seekpath.brillouinzone.brillouinzone import get_BZ


def show_brillouin_zone(reciprocal_lattice:np.array):
    """
    Extract from seekpath's script.

    Args:
        reciprocal_lattice (np.array): Reciprocal lattice.

    Notes:
        This is not supported at 2020/12/06.
        Therefore, some errors may occur.
        Future edit.
    """
    from pylab import figure, show
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from collections import defaultdict

    # draw a vector
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    class Arrow3D(FancyArrowPatch):

        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    faces_data = get_BZ(b1=reciprocal_lattice[0],
                        b2=reciprocal_lattice[1],
                        b3=reciprocal_lattice[2])

    faces_coords = faces_data['faces']

    faces_count = defaultdict(int)
    for face in faces_coords:
        faces_count[len(face)] += 1

    for num_sides in sorted(faces_count.keys()):
        print("{} faces: {}".format(num_sides, faces_count[num_sides]))

    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(
        Poly3DCollection(faces_coords,
                         linewidth=1,
                         alpha=0.9,
                         edgecolor="k",
                         facecolor="#ccccff"))

    # draw origin
    ax.scatter([0], [0], [0], color="g", s=100)

    axes_length = 2
    # Add axes
    ax.add_artist(
        Arrow3D((0, axes_length), (0, 0), (0, 0),
                mutation_scale=20,
                lw=1,
                arrowstyle="-|>",
                color="k"))
    ax.add_artist(
        Arrow3D((0, 0), (0, axes_length), (0, 0),
                mutation_scale=20,
                lw=1,
                arrowstyle="-|>",
                color="k"))
    ax.add_artist(
        Arrow3D((0, 0), (0, 0), (0, axes_length),
                mutation_scale=20,
                lw=1,
                arrowstyle="-|>",
                color="k"))

    # Reset limits
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.axis('off')
    ax.view_init(elev=0, azim=60)

    show()
