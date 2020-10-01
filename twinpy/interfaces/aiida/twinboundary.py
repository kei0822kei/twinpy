#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aiida interface for twinpy.
"""
import numpy as np
from copy import deepcopy
from pprint import pprint
import warnings
from matplotlib import pyplot as plt
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.orm import (load_node,
                       Node,
                       KpointsData,
                       Float,
                       Int,
                       Bool)
from twinpy.interfaces.aiida import (check_process_class,
                                     get_aiida_structure,
                                     get_cell_from_aiida,
                                     _WorkChain,
                                     AiidaRelaxWorkChain,
                                     AiidaPhonopyWorkChain)
from twinpy.structure.base import check_same_cells
from twinpy.structure.diff import get_structure_diff
from twinpy.structure.standardize import (get_standardized_cell,
                                          StandardizeCell)
from twinpy.api_twinpy import get_twinpy_from_cell
from twinpy.common.utils import print_header
from twinpy.lattice.lattice import Lattice
from twinpy.plot.base import line_chart
from twinpy.analysis import (RelaxAnalyzer,
                             PhononAnalyzer,
                             TwinBoundaryAnalyzer)


@with_dbenv()
class AiidaTwinBoudnaryRelaxWorkChain(_WorkChain):
    """
    TwinBoundaryRelax work chain class.
    """

    def __init__(
            self,
            node:Node,
            ):
        """
        Args:
            node: TwinBoundaryRelaxWorkChain node
        """
        process_class = 'TwinBoundaryRelaxWorkChain'
        check_process_class(node, process_class)
        super().__init__(node=node)
        self._twinboundary_settings = None
        self._cells = None
        self._structure_pks = None
        self._set_twinboundary()
        self._twinpy = None
        self._standardize = None
        self._set_twinpy()
        self._additional_relax_pks = []

    def _set_twinboundary(self):
        """
        Set twinboundary from vasp.
        """
        parameters = self._node.called[-1].outputs.parameters.get_dict()
        aiida_hexagonal = self._node.inputs.structure
        aiida_tb = self._node.called[-1].outputs.twinboundary
        aiida_tb_original = self._node.called[-1].outputs.twinboundary_orig
        aiida_tb_relax = self._node.called[-2].outputs.relax__structure
        tb = get_cell_from_aiida(aiida_tb)
        tb_original = get_cell_from_aiida(aiida_tb_original)
        tb_relax = get_cell_from_aiida(aiida_tb_relax)

        round_cells = []
        for cell in [ tb, tb_original, tb_relax ]:
            round_lattice = np.round(cell[0], decimals=8)
            round_lattice = cell[0]
            round_atoms = np.round(cell[1], decimals=8) % 1
            round_cell = (round_lattice, round_atoms, cell[2])
            round_cells.append(round_cell)

        self._twinboundary_settings = parameters
        self._cells = {
                'hexagonal': get_cell_from_aiida(aiida_hexagonal),
                'twinboundary': round_cells[0],
                'twinboundary_original': round_cells[1],
                'twinboundary_relax': round_cells[2],
                }
        self._structure_pks = {
                'hexagonal_pk': aiida_hexagonal.pk,
                'twinboundary_pk': aiida_tb.pk,
                'twinboundary_original_pk': aiida_tb_original.pk,
                'twinboundary_relax_pk': aiida_tb_relax.pk,
                }

    @property
    def twinboundary_settings(self):
        """
        Twinboundary settings.
        """
        return self._twinboundary_settings

    @property
    def cells(self):
        """
        cells.
        """
        return self._cells

    @property
    def structure_pks(self):
        """
        Twinboundary structure pks
        """
        return self._structure_pks

    def _set_twinpy(self):
        """
        Set twinpy structure object and standardize object.
        """
        params = self._twinboundary_settings
        cell = get_cell_from_aiida(
                load_node(self._structure_pks['hexagonal_pk']))
        twinpy = get_twinpy_from_cell(
                cell=cell,
                twinmode=params['twinmode'])
        twinpy.set_twinboundary(
                layers=params['layers'],
                delta=params['delta'],
                twintype=params['twintype'],
                xshift=params['xshift'],
                yshift=params['yshift'],
                shear_strain_ratio=params['shear_strain_ratio'],
                )
        self._twinpy = twinpy

    @property
    def twinpy(self):
        """
        Twinpy structure object.
        """
        return self._twinpy

    @property
    def standardize(self):
        """
        Stadardize object of twinpy original cell.
        """
        return self._standardize

    def _get_additional_relax_final_structure_pk(self):
        """
        Get additional relax final structure.
        """
        if self._additional_relax_pks == []:
            raise RuntimeError("additional_relax_pks is not set.")
        else:
            aiida_relax = AiidaRelaxWorkChain(
                    load_node(self._additional_relax_pks[-1]))
            final_pk = aiida_relax.get_pks()['current_final_structure_pk']
        return final_pk

    def set_additional_relax(self, aiida_relax_pks:list):
        """
        Set additional_relax_pks in the case final structure of
        TwinBoundaryRelax WorkChain is further relaxed.

        Args:
            aiida_relaxes (list): list of relax pks

        Raises:
            RuntimeError: Input node is not RelaxWorkChain
            RuntimeError: Output structure and next input structure
                          does not match.
        """
        previous_rlx = self.get_pks()['relax_pk']
        structure_pk = self._structure_pks['twinboundary_relax_pk']
        aiida_relaxes = [ load_node(relax_pk) for relax_pk in aiida_relax_pks ]
        for relax in aiida_relaxes:
            if relax.process_class.get_name() != 'RelaxWorkChain':
                raise RuntimeError(
                        "Input node (pk={}) is not RelaxWorkChain".format(
                            relax.pk))
            if structure_pk == relax.inputs.structure.pk:
                if relax.process_state.value == 'finished':
                    structure_pk = relax.outputs.relax__structure.pk
                else:
                    warnings.warn("RelaxWorkChain (pk={}) state is {}".format(
                        relax.pk, relax.process_state.value))
                    if relax.process_state.value == 'excepted':
                        structure_pk = \
                            relax.called[1].called[0].outputs.structure.pk
                previous_rlx = relax.pk
            else:
                print("previous relax: pk={}".format(previous_rlx))
                print("next relax: pk={}".format(relax.pk))
                raise RuntimeError("Output structure and next input structure "
                                   "does not match.")
        self._additional_relax_pks = aiida_relax_pks
        rlx_structure_pk = self._get_additional_relax_final_structure_pk()
        rlx_cell = get_cell_from_aiida(load_node(rlx_structure_pk))
        self._structure_pks['twinboundary_additional_relax_pk'] = \
                rlx_structure_pk
        self._cells['twinboundary_additional_relax'] = \
                rlx_cell

    @property
    def additional_relax_pks(self):
        """
        Additional relax pks.
        """
        return self._additional_relax_pks

    def get_pks(self):
        """
        Get pks.
        """
        relax_pk = self._node.called[-2].pk
        pks = {}
        pks['twinboundary_pk'] = self._pk
        pks['relax_pk'] = relax_pk
        pks['additional_relax_pks'] = self._additional_relax_pks
        return pks

    def get_twinboundary_analyzer(self,
                                  twinboundary_phonon_pk:int,
                                  hexagonal_phonon_pk:int=None):
        """
        Get TwinBoundaryAnalyzer class object.

        Args:
            twinboundary_phonon_pk (int): Twinboundary phonon pk.
            hexagonal_phonon_pk (int): Hexagonal phonon pk.
        """
        def __get_twinboundary_relax_original_cell(relax_cell):
            std = self._twinpy.get_twinboundary_standardize(
                      get_lattice=False,
                      move_atoms_into_unitcell=True,
                      )
            rotation_matrix = std.rotation_matrix
            lattice = relax_cell[0]
            orig_lattice = np.dot(np.linalg.inv(rotation_matrix),
                                  lattice.T).T
            orig_relax_cell = (orig_lattice, relax_cell[1], relax_cell[2])
            return orig_relax_cell

        tb_aiida_relax = AiidaRelaxWorkChain(
                load_node(self.get_pks()['relax_pk']))
        tb_aiida_relax.set_additional_relax(
                aiida_relax_pks=self._additional_relax_pks)
        relax_cell = tb_aiida_relax.current_final_cell
        print(tb_aiida_relax.get_pks())
        orig_relax_cell = __get_twinboundary_relax_original_cell(
                relax_cell=relax_cell)
        tb_aiida_phonopy = AiidaPhonopyWorkChain(
                load_node(twinboundary_phonon_pk))
        relax_analyzer = tb_aiida_relax.get_relax_analyzer(
                original_cell=orig_relax_cell)
        tb_phonon_analyzer = PhononAnalyzer(
                phonon=tb_aiida_phonopy.get_phonon(),
                relax_analyzer=relax_analyzer)

        hex_aiida_phonopy = AiidaPhonopyWorkChain(
                load_node(hexagonal_phonon_pk))

        twinboundary_analyzer = TwinBoundaryAnalyzer(
                twinboundary_structure=self._twinpy,
                twinboundary_phonon_analyzer=tb_phonon_analyzer,
                hexagonal_phonon_analyzer=hex_phonon_analyzer)

        return twinboundary_analyzer
