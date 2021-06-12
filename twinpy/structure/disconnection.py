#!/usr/bin/env python

"""
This module deals with disconnection.
"""

from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
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
            if arr:
                return ( arr + np.array(1,ix,1) ) / np.array(1,rep,1)
            return []

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
                seg_w_lp = w_lp[:-1]

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

        self._w_lp = dich_w_lp
        self._wt_lp = dich_wt_lp
        self._b_lp = dich_b_lp
        self._bt_lp = dich_bt_lp
        self._w_left_lp = dich_w_left_lp
        self._b_right_lp = dich_b_right_lp
        self._b_replicate = b_replicate
        self._step_start = step_start
        self._step_range = step_range

    def show_dichromatic_lattice(self, scale=0.3):
        colors = ['r', 'green', 'b', 'brown', 'cyan', 'grey']
        b = self._twinboundary.output_structure['lattice'][1,1]
        c = self._twinboundary.output_structure['lattice'][2,2]
        fig = plt.figure(figsize=(b*scale*self._b_replicate, c*scale))
        ax = fig.add_subplo(111)
        for i in range(self._b_replicate):
            for j, lp in enumerate((self._w_lp,
                                    self._wt_lp,
                                    self._b_lp,
                                    self._bt_lp,
                                    self._w_left_lp,
                                    self._b_right_lp,
                                    )):
                if lp[i]:
                    ax.scatter(lp[i][:,1], lp[i][:,2], c=colors[j])
