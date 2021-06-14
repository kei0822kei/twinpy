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
           ):
        """
        Init.
        """
        self._twinboundary = twinboundary
        self._check_support_twinmode()
        self._w_lp = None
        self._wt_lp = None
        self._b_lp = None
        self._bt_lp = None
        self._w_left_lp = None
        self._b_right_lp = None
        self._b_replicate = None
        self._step_start = None
        self._step_range = None
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

    def run(self,
            b_replicate:int,
            step_start:int,
            step_range:int,
            ):

        def _get_scaled_positions(arr, ix, rep):
            if arr != []:
                return ( arr + np.array([1,ix,1]) ) / np.array([1,rep,1] ) % 1
            return []

        def _get_atom_positions(tb_atom_positions, rep):
            disco_atoms = {}
            for key in tb_atom_positions:
                disco_atoms[key] = tb_atom_positions[key] / np.array([1,rep,1])
            disco_atoms['white_left'] = disco_atoms['white'][1].reshape(1,3)
            disco_atoms['black_right'] = disco_atoms['black'][1].reshape(1,3)
            return disco_atoms

        twinpy_structure = self._twinboundary.output_structure
        w_lp = twinpy_structure['lattice_points']['white']
        wt_lp = twinpy_structure['lattice_points']['white_tb']
        b_lp = twinpy_structure['lattice_points']['black']
        bt_lp = twinpy_structure['lattice_points']['black_tb']

        # w_vec = (w_lp[1] - w_lp[0]) % 1
        b_vec = (b_lp[1] - b_lp[0]) % 1

        dich_w_lp = []
        dich_wt_lp = []
        dich_b_lp = []
        dich_bt_lp = []
        dich_w_left_lp = []
        dich_b_right_lp = []

        for i in range(b_replicate):
            seg_w_lp = deepcopy(w_lp)
            seg_wt_lp = deepcopy(wt_lp)
            seg_b_lp = deepcopy(b_lp)
            seg_bt_lp = deepcopy(bt_lp)
            seg_w_left_lp = []
            seg_b_right_lp = []

            if i == step_start:
                arr = (bt_lp + np.array([0,i+1,0]) - b_vec) % 1
                seg_b_lp = np.vstack((seg_b_lp, arr))
                seg_bt_lp = np.vstack((bt_lp, w_lp[-2]))
                seg_w_left_lp = w_lp[-1].reshape(1,3)
                seg_w_lp = w_lp[:-2]

            elif step_start < i < step_start + step_range:
                seg_b_lp = np.vstack((b_lp, bt_lp, (bt_lp-b_vec)%1))
                seg_bt_lp = w_lp[-2].reshape(1,3)
                seg_w_lp = w_lp[:-2]

            elif i == step_start + step_range:
                seg_b_lp = np.vstack((b_lp, bt_lp))
                seg_b_right_lp = ((bt_lp-b_vec)%1).reshape(1,3)
                seg_bt_lp = w_lp[-2].reshape(1,3)
                seg_w_lp = w_lp[:-2]

            for dich, seg in zip([dich_w_lp,
                                  dich_wt_lp,
                                  dich_b_lp,
                                  dich_bt_lp,
                                  dich_w_left_lp,
                                  dich_b_right_lp],
                                 [seg_w_lp,
                                  seg_wt_lp,
                                  seg_b_lp,
                                  seg_bt_lp,
                                  seg_w_left_lp,
                                  seg_b_right_lp]
                                 ):
                dich.append(_get_scaled_positions(seg, i, b_replicate))

        lattice_points = {
                'white': dich_w_lp,
                'white_tb': dich_wt_lp,
                'white_left': dich_w_left_lp,
                'black': dich_b_lp,
                'black_tb': dich_bt_lp,
                'black_right': dich_b_right_lp,
                }
        atoms_from_lattice_points = \
            _get_atom_positions(twinpy_structure['atoms_from_lattice_points'],
                                b_replicate)
        lattice = twinpy_structure['lattice'] \
                      * np.array([1,b_replicate,1])

        self._output_structure = {
                'lattice': lattice,
                'lattice_points': lattice_points,
                'atoms_from_lattice_points': atoms_from_lattice_points,
                }
        self._b_replicate = b_replicate
        self._step_start = step_start
        self._step_range = step_range

    def get_cell_for_export(self):
        scaled_positions = []
        for key in self._output_structure['lattice_points'].keys():
            lps = self._output_structure['lattice_points'][key]
            atoms = self._output_structure['atoms_from_lattice_points'][key]
            for lp in lps:
                if lp != []:
                    posi = get_atom_positions_from_lattice_points(lp, atoms)
                scaled_positions.extend(posi.tolist())
        scaled_positions = np.round(np.array(scaled_positions), decimals=8)
        scaled_positions %= 1.

        twinpy_structure = self._twinboundary.output_structure
        lattice = self._output_structure['lattice']
        symbol = twinpy_structure['symbols'][0]
        symbols = [ symbol ] * len(scaled_positions)

        return (lattice, scaled_positions, symbols)

    def show_dichromatic_lattice(self, scale=0.3):
        colors = ['r', 'green', 'b', 'brown', 'cyan', 'grey']
        b = self._twinboundary.output_structure['lattice'][1,1]
        c = self._twinboundary.output_structure['lattice'][2,2]
        fig = plt.figure(figsize=(b*scale*self._b_replicate, c*scale))
        ax = fig.add_subplot(111)
        for i in range(self._b_replicate):
            for j, key in enumerate(['white', 'white_tb', 'white_left',
                                     'black', 'black_tb', 'black_right']):
                lp = self._output_structure['lattice_points'][key]
                if lp[i] != []:
                    ax.scatter(lp[i][:,1], lp[i][:,2], c=colors[j])
