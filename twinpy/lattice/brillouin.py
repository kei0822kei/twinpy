#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brillouin zone
"""
import numpy as np
from seekpath.brillouinzone.brillouinzone import get_BZ

def get_labels_for_twin():
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

def show_brillouin_zone(reciprocal_lattice):
    """
    extract from seekpath's script
    """
    from pylab import figure, show
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from collections import defaultdict
    import json

    #draw a vector
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

    #draw origin
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

    ## Reset limits
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.axis('off')
    ax.view_init(elev=0, azim=60)

    show()

# def get_labels_for_twin(structure):
#     """
#     get label and qpoints, which refers to the Inoue's work
#     structure is pathed to seekpath
#     """
#     paths = seekpath.get_path(structure)
#     spgnum = paths['spacegroup_number']
#     results = {'spacegroup_number': spgnum}
#     if spgnum == 194:  # P6_3/mmc
#         label_qpoints = {
#           'GAMMA': [0, 0, 0],
#           'M_1'  : [1/2, 0, 0],
#           'M_2'  : [-1/2, 1/2, 0],
#           'K_1'  : [1/3, 1/3, 0],
#           'K_2'  : [-1/3, 2/3, 0],
#           'A'    : [0, 0, 1/2],
#           'L_1'  : [1/2, 0, 1/2],
#           'L_2'  : [-1/2, 1/2, 1/2],
#           'H_1'  : [1/3, 1/3, 1/2],
#           'H_2'  : [-1/3, 2/3, 1/2],
#         }
# 
#     else:
#         coords = paths['point_coords']
#         if spgnum == 63:  # Cmcm
#             labels_corr = {
#               'GAMMA': 'GAMMA',
#               'M_1'  : 'S',
#               'M_2'  : 'Y',
#               'K_1'  : 'SIGMA_0',
#               'K_2'  : 'C_0',
#               'A'    : 'Z',
#               'L_1'  : 'R',
#               'L_2'  : 'T',
#               'H_1'  : 'A_0',
#               'H_2'  : 'E_0',
#             }
#         elif spgnum == 12:  # C2/m
#             labels_corr = {
#               'GAMMA': 'GAMMA',
#               'M_1'  : 'V',
#               'M_2'  : 'V_2',
#               'K_1'  : 'C',
#               'K_2'  : 'C_2',
#               'A'    : 'A',
#               'L_1'  : 'L_2',
#               'L_2'  : 'M_2',  # nearly equal E_2
#               'H_1'  : 'D_2',
#               'H_2'  : 'D',
#             }
#         results['label_tags'] = labels_corr
#         label_qpoints = {}
#         for label in labels_corr:
#             label_qpoints[label] = paths['point_coords'][labels_corr[label]]
#     results['label_qpoints'] = label_qpoints
#     results['default_paths'] = ['A', 'H_2', 'L_2', 'A', 'GAMMA',
#                                 'K_2', 'M_2', 'GAMMA',
#                                 'M_1', 'K_1', 'GAMMA',
#                                 'A', 'L_1', 'H_1', 'A']
#     return results
