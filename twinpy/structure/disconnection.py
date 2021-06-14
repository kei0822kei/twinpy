#!/usr/bin/env python

"""
This module deals with disconnection.
"""

from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from twinpy.structure.base import get_atom_positions_from_lattice_points
from twinpy.structure.twinboundary import TwinBoundaryStructure


class Disconnection():
    """
    Disconnection generation class.
    """

    def __init__(
           self,
           twinboundary:TwinBoundaryStructure,
           b_replicate:int,
           ):
        """
        Init.

        Args:
            twinboundary: TwinBoundaryStructure class object.
            b_replicate: Replicate number for b axis.
        """
        self._twinboundary = twinboundary
        self._check_support_twinmode()
        self._b_replicate = b_replicate
        self._lattice = None
        self._set_lattice()
        self._w_tl_vec, self._b_tl_vec \
                = self._twinboundary.get_translation_vectors()
        self._burg_vec = None
        self._set_burgers_vector()
        self._atoms_from_lattice_points = None
        self._set_atoms_from_lattice_points()
        self._output_structure = None

    def _check_support_twinmode(self):
        """
        Check support twinmode.

        Raises:
            RuntimeError: Input twinmode is not supported.
        """
        tm = self._twinboundary.twinmode
        if tm not in ['10-12']:
            raise RuntimeError("twinmode: %s is not supported" % tm)

    def _set_lattice(self):
        """
        Set lattice.
        """
        lat_orig = self._twinboundary.output_structure['lattice']
        lat = lat_orig * np.array([1, self._b_replicate, 1])
        self._lattice = lat

    def _set_atoms_from_lattice_points(self):
        """
        Set atoms from lattice points.
        """
        atoms = self._twinboundary.output_structure['atoms_from_lattice_points']
        b_rep = self._b_replicate
        for key in atoms:
            atoms[key] /= np.array([1., b_rep, 1.])
        atoms['white_left'] = atoms['white'][1].reshape(1,3)
        atoms['black_right'] = atoms['black'][1].reshape(1,3)
        self._atoms_from_lattice_points = atoms

    def _set_burgers_vector(self):
        """
        Set burgers_vector.
        """
        w_vec = self._w_tl_vec
        b_vec = self._b_tl_vec
        burg_vec = np.round(b_vec * 2 - (-w_vec * 2 + np.array([0.,1.,0.])),
                            decimals=6) % 1
        self._burg_vec = burg_vec

    def run(self,
            step_start:int,
            step_range:int,
            ):
        """
        Create disconnection induced structure.

        Args:
            step_start: Step start index.
            step_range: Step range.
        """
        def _sort_arr(arr):
            sort_arr = arr[np.argsort(arr[:,1])]
            return sort_arr

        def _get_white_tb_points():
            w_vec = self._w_tl_vec
            b_rep = self._b_replicate
            layers = self._twinboundary.layers
            start_point = np.array([0., 0., 0.5]) + w_vec * (layers+1)
            points = [ start_point+np.array([0., float(i), 0.])
                           for i in range(b_rep) ]
            points_ = np.array(points) / np.array([1., b_rep, 1])
            points_ = np.round(points_, decimals=6) % 1
            return _sort_arr(np.array(points_))

        def _get_black_tb_points():
            b_vec = self._b_tl_vec
            b_rep = self._b_replicate
            points = []
            for i in range(b_rep):
                if (step_start+1) < i < (step_start+step_range+1):
                    points.append(points[-1]+np.array([0., 1., 0.]))
                elif i == (step_start+1):
                    points.append(
                        points[-1] + np.array([0., 1., 0.]) - 2 * b_vec)
                elif i == (step_start+step_range+1):
                    points.append(points[-1]+np.array([0., 1., 0.]))
                    points.append(np.array([0., float(i), 0.5]))
                else:
                    points.append(np.array([0., float(i), 0.5]))
            return np.array(points) / np.array([1., b_rep, 1]) % 1

        def _get_white_points(b_tb):
            w_vec = self._w_tl_vec
            b_rep = self._b_replicate
            burg_vec = self._burg_vec
            _black_tb = b_tb * np.array([1., b_rep, 1.])
            layers = self._twinboundary.layers
            points = []
            points_left = []
            for i, pt in enumerate(_black_tb):
                if i < step_start:
                    for j in range(layers):
                        points.append(pt+w_vec*(j+1))
                elif i == step_start:
                    points_left.append(pt+w_vec)
                elif (step_start+1) <= i < (step_start+step_range+2):
                    for j in range(layers-2):
                        points.append(pt+burg_vec+w_vec*(j+1))
                else:
                    for j in range(layers):
                        points.append(pt+w_vec*(j+1))
            points_ = np.array(points) / np.array([1., b_rep, 1]) % 1
            points_left_ = np.array(points_left) / np.array([1., b_rep, 1]) % 1
            return (_sort_arr(np.array(points_)),
                    _sort_arr(np.array(points_left_)))

        def _get_black_points(b_tb):
            b_vec = self._b_tl_vec
            b_rep = self._b_replicate
            burg_vec = self._burg_vec
            _black_tb = b_tb * np.array([1., b_rep, 1.])
            layers = self._twinboundary.layers
            points = []
            points_right = []
            for i, pt in enumerate(_black_tb):
                if i <= step_start:
                    for j in range(layers):
                        points.append(pt+b_vec*(j+1))
                elif (step_start+1) <= i < (step_start+step_range+1):
                    for j in range(layers+2):
                        points.append(pt+b_vec*(j+1))
                elif i == (step_start+step_range+1):
                    points_right.append(pt+b_vec)
                else:
                    for j in range(layers):
                        points.append(pt+b_vec*(j+1))
            points_ = np.array(points) / np.array([1., b_rep, 1]) % 1
            points_right_ = np.array(points_right) / np.array([1., b_rep, 1]) % 1
            return (_sort_arr(np.array(points_)),
                    _sort_arr(np.array(points_right_)))

        black_tb = _get_black_tb_points()
        white_tb = _get_white_tb_points()
        white, white_left = _get_white_points(black_tb)
        black, black_right = _get_black_points(black_tb)
        lattice_points = {
                'white': white,
                'white_left': white_left,
                'white_tb': white_tb,
                'black': black,
                'black_right': black_right,
                'black_tb': black_tb,
                }
        output_structure = {
                'lattice': self._lattice,
                'lattice_points': lattice_points,
                'atoms_from_lattice_points': self._atoms_from_lattice_points,
                'symbols': self._twinboundary.output_structure['symbols'],
                }
        self._output_structure = output_structure

    def get_cell_for_export(self):
        scaled_positions = []
        for key in self._output_structure['lattice_points'].keys():
            lps = self._output_structure['lattice_points'][key]
            atoms = self._output_structure['atoms_from_lattice_points'][key]
            positions = get_atom_positions_from_lattice_points(lps, atoms)
            scaled_positions.extend(positions.tolist())
        scaled_positions = np.round(np.array(scaled_positions), decimals=8)
        scaled_positions %= 1.

        twinpy_structure = self._twinboundary.output_structure
        lattice = self._output_structure['lattice']
        symbols = self._output_structure['symbols'] * self._b_replicate

        return (lattice, scaled_positions, symbols)

    def show_dichromatic_lattice(self, scale=0.3):
        colors = ['r', 'green', 'brown', 'b', 'cyan', 'grey']
        b = self._twinboundary.output_structure['lattice'][1,1]
        c = self._twinboundary.output_structure['lattice'][2,2]
        fig = plt.figure(figsize=(b*scale*self._b_replicate, c*scale))
        ax = fig.add_subplot(111)
        atoms = self._output_structure['lattice_points']
        for i, key in enumerate(['white', 'white_tb', 'white_left',
                                 'black', 'black_tb', 'black_right']):
            ax.scatter(atoms[key][:,1], atoms[key][:,2], c=colors[i])
        ax.set_xlim((0,1))
        ax.set_ylim((0,1))
