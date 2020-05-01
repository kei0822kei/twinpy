#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brillouin zone
"""
import numpy as np
import seekpath

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
