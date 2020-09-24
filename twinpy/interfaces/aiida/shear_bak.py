#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aiida interface for twinpy.
"""
import numpy as np
from pprint import pprint
import warnings
from matplotlib import pyplot as plt
from twinpy.analysis.shear_analyzer import ShearAnalyzer
from twinpy.interfaces.phonopy import get_phonopy_structure
from twinpy.structure.base import check_same_cells, check_same_cells
from twinpy.structure.diff import get_structure_diff
from twinpy.api_twinpy import get_twinpy_from_cell
from twinpy.common.kpoints import get_mesh_offset_from_direct_lattice
from twinpy.common.utils import print_header
from twinpy.plot.base import line_chart, DEFAULT_COLORS, DEFAULT_MARKERS
# from twinpy.plot.twinboundary import plane_diff
from twinpy.lattice.lattice import Lattice
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.common import NotExistentAttributeError
from aiida.orm import (load_node,
                       Node,
                       QueryBuilder,
                       StructureData,
                       CalcFunctionNode,
                       WorkChainNode)
from aiida.plugins import WorkflowFactory
from aiida.common.exceptions import NotExistentAttributeError
from phonopy import Phonopy


@with_dbenv()
class AiidaShearWorkChain():
    """
    Shear work chain class.
    """

    def __init__(
            self,
            node:Node,
            ):
        """
        Args:
            relax_pk (int): relax pk
        """
        process_class = 'ShearWorkChain'
        check_process_class(node, process_class)
        super().__init__(node=node)
        node = load_node(pk)
        check_process_class(node, 'ShearWorkChain')
        is_phonon = node.inputs.is_phonon.value

        self._shear_conf = node.inputs.shear_conf.get_dict()
        self._shear_ratios = \
            node.called[-1].outputs.shear_settings.get_dict()['shear_ratios']
        self._hexagonal_cell = get_cell_from_aiida(node.inputs.structure)
        self._gamma = node.outputs.gamma.value

        self._node = node
        self._pk = pk

        self._create_shears_pk = None
        self._original_cell_pks = None
        self._original_cells = None
        self._set_shear_structures()

        self._relax_pks = None
        self._relaxes = None
        self._set_relaxes()

        self._phonon_pks = None
        self._phonons = None
        if is_phonon:
            self._set_phonons()

    @property
    def node(self):
        """
        ShearWorkChain node.
        """
        return self._node

    @property
    def pk(self):
        """
        ShearWorkChain pk.
        """
        return self._pk

    @property
    def shear_conf(self):
        """
        Input shear conf
        """
        return self._shear_conf

    @property
    def shear_ratios(self):
        """
        Output shear ratios
        """
        return self._shear_ratios

    @property
    def hexagonal_cell(self):
        """
        Input hexagonal cell
        """
        return self._hexagonal_cell

    @property
    def original_cells(self):
        """
        Output shear original cells
        """
        return self._original_cells

    @property
    def gamma(self):
        """
        Output gamma
        """
        return self._gamma

    def _set_shear_structures(self):
        """
        Set original cells in ShearWorkChain.
        """
        qb = QueryBuilder()
        qb.append(Node, filters={'id':{'==': self._pk}}, tag='wf')
        qb.append(CalcFunctionNode,
                  filters={'label':{'==': 'get_shear_structures'}},
                  with_incoming='wf',
                  project=['id'])
        create_shears_pk = qb.all()[0][0]

        qb = QueryBuilder()
        qb.append(Node, filters={'id':{'==': create_shears_pk}}, tag='cs')
        qb.append(StructureData,
                  with_incoming='cs',
                  project=['id', 'label'])
        structs = qb.all()
        orig_cell_pks = [ struct[0] for struct in structs
                                    if 'shear_orig' in struct[1] ]
        orig_cell_pks.sort(key=lambda x: x)

        self._create_shears_pk = create_shears_pk
        self._original_cell_pks = orig_cell_pks
        self._original_cells = [ get_cell_from_aiida(load_node(pk))
                                     for pk in self._original_cell_pks ]

    def _set_relaxes(self):
        """
        Set relax in ShearWorkChain.
        """
        self._relax_pks = get_workflow_pks(pk=self._pk,
                                           workflow_name='vasp.relax')
        self._relaxes = [ RelaxWorkChain(pk=pk) for pk in self._relax_pks ]

    @property
    def relax_pks(self):
        """
        Output relax pks.
        """
        return self._relax_pks

    @property
    def relaxes(self):
        """
        Output relaxes in ShearWorkChain.
        """
        return self._relaxes

    def _set_phonons(self):
        """
        Set phonon_pks in ShearWorkChain.
        """
        self._phonon_pks = get_workflow_pks(pk=self._pk,
                                            workflow_name='phonopy.phonopy')
        self._phonons = [ PhonopyWorkChain(pk=pk) for pk in self._phonon_pks ]

    @property
    def phonon_pks(self):
        """
        Output phonon pks.
        """
        return self._phonon_pks

    @property
    def phonons(self):
        """
        Output phonons in ShearWorkChain.
        """
        return self._phonons

    def get_analyzer(self) -> ShearAnalyzer:
        """
        Get ShearAnalyzer class object.
        """
        original_cells = self._original_cells
        input_cells = [ relax.initial_cell for relax in self._relaxes ]
        relax_cells = [ relax.final_cell for relax in self._relaxes ]
        analyzer = ShearAnalyzer(original_cells=original_cells,
                                 input_cells=input_cells,
                                 relax_cells=relax_cells)
        phns = [ phonon_wf.get_phonon() for phonon_wf in self._phonons ]
        analyzer.set_phonons(phonons=phns)
        return analyzer

    def get_pks(self) -> dict:
        """
        Get pks.

        Returns:
            dict: keys and corresponding pks
        """
        pks = {
                'shear_pk': self._pk,
                'get_shear_structures_pk': self._create_shears_pk,
                'original_cell_pks': self._original_cell_pks,
                'relax_pks': self._relax_pks,
                'phonon_pks': self._phonon_pks,
              }
        return pks
