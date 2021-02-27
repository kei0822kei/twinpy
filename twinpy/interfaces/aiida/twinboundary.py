#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aiida interface for twinpy.
"""
from pprint import pprint
from copy import deepcopy
import numpy as np
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.orm import (load_node,
                       Node,
                       KpointsData,
                       Float,
                       Int,
                       Bool)
from twinpy.common.utils import print_header
from twinpy.interfaces.aiida.base import (check_process_class,
                                          get_aiida_structure,
                                          get_cell_from_aiida,
                                          _WorkChain)
from twinpy.interfaces.aiida.vasp import (AiidaRelaxWorkChain,
                                          AiidaRelaxCollection)
from twinpy.interfaces.aiida.phonopy import AiidaPhonopyWorkChain
from twinpy.properties.hexagonal import get_wyckoff_from_hcp
from twinpy.structure.standardize import StandardizeCell
from twinpy.structure.twinboundary import get_twinboundary
from twinpy.analysis.twinboundary_analyzer import TwinBoundaryAnalyzer


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
            node: TwinBoundaryRelaxWorkChain node.
        """
        process_class = 'TwinBoundaryRelaxWorkChain'
        check_process_class(node, process_class)
        super().__init__(node=node)
        self._twinboundary_parameters = None
        self._cells = None
        self._structure_pks = None
        self._set_twinboundary()
        self._twinboundary_structure = None
        self._standardize = None
        self._set_twinboundary_structure()

    def _set_twinboundary(self):
        """
        Set twinboundary from vasp.
        """
        tb_parameters = self._node.outputs.twinboundary_parameters.get_dict()
        aiida_hex_structure = self._node.inputs.structure
        aiida_tb_structure = self._node.called[0].outputs.twinboundary
        aiida_tb_orig_structure = \
                self._node.called[0].outputs.twinboundary_orig
        aiida_tb_rlx_structure = \
                self._node.called[-1].outputs.relax__structure

        round_cells = []
        for aiida_structure in [ aiida_tb_orig_structure,
                                 aiida_tb_orig_structure,
                                 aiida_tb_rlx_structure ]:
            cell = get_cell_from_aiida(aiida_structure)
            round_lattice = np.round(cell[0], decimals=8)
            round_lattice = cell[0]
            round_atoms = np.round(cell[1], decimals=8) % 1
            round_cell = (round_lattice, round_atoms, cell[2])
            round_cells.append(round_cell)

        self._twinboundary_parameters = tb_parameters
        self._cells = {
                'hexagonal': get_cell_from_aiida(aiida_hex_structure),
                'twinboundary': round_cells[0],
                'twinboundary_original': round_cells[1],
                'twinboundary_relax': round_cells[2],
                }
        self._structure_pks = {
                'hexagonal_pk': aiida_hex_structure.pk,
                'twinboundary_pk': aiida_tb_structure.pk,
                'twinboundary_original_pk': aiida_tb_orig_structure.pk,
                'twinboundary_relax_pk': aiida_tb_rlx_structure.pk,
                }

    @property
    def twinboundary_parameters(self):
        """
        Twinboundary settings.
        """
        return self._twinboundary_parameters

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

    def _set_twinboundary_structure(self):
        """
        Set twinpy structure object and standardize object.
        """
        params = self._twinboundary_parameters
        cell = get_cell_from_aiida(
                        load_node(self._structure_pks['hexagonal_pk']))
        lattice, _, symbols = cell
        wyckoff = get_wyckoff_from_hcp(cell=cell)
        twinboundary = get_twinboundary(
                           lattice=lattice,
                           symbol=symbols[0],
                           twinmode=params['twinmode'],
                           layers=params['layers'],
                           wyckoff=wyckoff,
                           delta=params['delta'],
                           twintype=params['twintype'],
                           xshift=params['xshift'],
                           yshift=params['yshift'],
                           shear_strain_ratio=params['shear_strain_ratio'])
        tb_cell = twinboundary.get_cell_for_export(
                      get_lattice=False,
                      move_atoms_into_unitcell=True,
                      )
        self._twinboundary_structure = twinboundary
        self._standardize = StandardizeCell(cell=tb_cell)

    @property
    def twinboundary_structure(self):
        """
        Twinpy structure object.
        """
        return self._twinboundary_structure

    @property
    def standardize(self):
        """
        Stadardize object of twinpy original cell.
        """
        return self._standardize

    def get_pks(self):
        """
        Get pks.
        """
        relax_pk = self._node.called[-1].pk
        pks = {}
        pks['twinboundary_pk'] = self._pk
        pks['relax_pk'] = relax_pk
        return pks

    def get_aiida_relax(
            self, additional_relax_pks:list=None) -> AiidaRelaxCollection:
        """
        Get relax collection.

        Args:
            additional_relax_pks: List of additinal relax calculation pks.

        Returns:
            AiidaRelaxCollection: Collection of aiida relax.
        """
        relax_pk = self.get_pks()['relax_pk']
        if additional_relax_pks is None:
            aiida_relax = AiidaRelaxWorkChain(load_node(relax_pk))
        else:
            relax_pks = deepcopy(additional_relax_pks)
            relax_pks.insert(0, relax_pk)
            aiida_relaxes = \
                    [ AiidaRelaxWorkChain(load_node(pk)) for pk in relax_pks ]
            aiida_relax = AiidaRelaxCollection(aiida_relaxes)

        return aiida_relax

    def get_twinboundary_analyzer(
            self,
            twinboundary_phonon_pk:int=None,
            additional_relax_pks:list=None,
            hexagonal_relax_pk:int=None,
            hexagonal_phonon_pk:int=None) -> TwinBoundaryAnalyzer:
        """
        Get TwinBoundaryAnalyzer class object.

        Args:
            twinboudnary_phonon_pk: Twinboundary phonon calculation pk.
            additional_relax_pks: List of additinal relax calculation
                                         pks.
            hexagonal_relax_pk: Hexagonal relax calculation pk.
            hexagonal_phonon_pk: Hexagonal phonon calculation pk.

        Returns:
            TwinBoundaryAnalyzer: TwinBoundaryAnalyzer class object.
        """
        twinboundary_structure = self._twinboundary_structure
        original_cell = self.cells['twinboundary_original']
        aiida_relax = self.get_aiida_relax(
                additional_relax_pks=additional_relax_pks)
        relax_analyzer = aiida_relax.get_relax_analyzer(
                             original_cell=original_cell)

        if twinboundary_phonon_pk is not None:
            aiida_phonopy = AiidaPhonopyWorkChain(
                                load_node(twinboundary_phonon_pk))
            phonon_analyzer = aiida_phonopy.get_phonon_analyzer(
                                  relax_analyzer=relax_analyzer)
        else:
            phonon_analyzer = None

        hex_phonon_analyzer = None
        if hexagonal_relax_pk is not None and hexagonal_phonon_pk is not None:
            aiida_hex_relax = AiidaRelaxWorkChain(
                                  load_node(hexagonal_relax_pk))
            hex_relax_analyzer = aiida_hex_relax.get_relax_analyzer()
            if hexagonal_phonon_pk is not None:
                aiida_hex_phonopy = AiidaPhonopyWorkChain(
                                        load_node(hexagonal_phonon_pk))
                hex_phonon_analyzer = aiida_hex_phonopy.get_phonon_analyzer(
                        relax_analyzer=hex_relax_analyzer)
        else:
            hex_phonon_analyzer = None
        twinboundary_analyzer = TwinBoundaryAnalyzer(
                twinboundary_structure=twinboundary_structure,
                twinboundary_relax_analyzer=relax_analyzer,
                twinboundary_phonon_analyzer=phonon_analyzer,
                hexagonal_phonon_analyzer=hex_phonon_analyzer,
                )
        return twinboundary_analyzer

    def get_shear_relax_builder(self,
                                shear_strain_ratio:float,
                                additional_relax_pks:list=None):
        """
        Get relax builder for shear introduced relax twinboundary structure.

        Args:
            shear_strain_ratio (float): shear strain ratio
        """
        twinboundary_analyzer = self.get_twinboundary_analyzer(
                additional_relax_pks=additional_relax_pks)
        cell = twinboundary_analyzer.get_shear_cell(
                shear_strain_ratio=shear_strain_ratio,
                is_standardize=False)  # in order to get rotation matrix
        std = StandardizeCell(cell=cell)
        std_cell = std.get_standardized_cell(to_primitive=True,
                                             no_idealize=False,
                                             no_sort=True)
        if additional_relax_pks is None:
            rlx_pk = self.get_pks()['relax_pk']
        else:
            rlx_pk = additional_relax_pks[-1]
        rlx_node = load_node(rlx_pk)
        builder = rlx_node.get_builder_restart()

        # fix kpoints
        mesh, offset = map(np.array, builder.kpoints.get_kpoints_mesh())
        orig_mesh = np.abs(np.dot(np.linalg.inv(
            self._standardize.transformation_matrix), mesh).astype(int))
        orig_offset = np.round(np.abs(np.dot(np.linalg.inv(
            std.transformation_matrix), offset)), decimals=2)
        std_mesh = np.abs(np.dot(std.transformation_matrix,
                                 orig_mesh).astype(int))
        std_offset = np.round(np.abs(np.dot(std.transformation_matrix,
                                            orig_offset)), decimals=2)
        kpt = KpointsData()
        kpt.set_kpoints_mesh(std_mesh, offset=std_offset)
        builder.kpoints = kpt

        # fix structure
        builder.structure = get_aiida_structure(cell=std_cell)

        # fix relax conf
        builder.relax.convergence_max_iterations = Int(100)
        builder.relax.positions = Bool(True)
        builder.relax.shape = Bool(False)
        builder.relax.volume = Bool(False)
        builder.relax.convergence_positions = Float(1e-4)
        builder.relax.force_cutoff = \
                Float(AiidaRelaxWorkChain(node=rlx_node).get_max_force())
        builder.metadata.label = "tbr:{} rlx:{} shr:{} std:{}".format(
                self._pk, rlx_node.pk, shear_strain_ratio, True)
        builder.metadata.description = \
                "twinboundary_relax_pk:{} relax_pk:{} " \
                "shear_strain_ratio:{} standardize:{}".format(
                    self._pk, rlx_node.pk, shear_strain_ratio, True)
        return builder

    def get_description(self):
        """
        Get description.
        """
        self._print_common_information()
        print_header('PKs')
        pprint(self.get_pks())
