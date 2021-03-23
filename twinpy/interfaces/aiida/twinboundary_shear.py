#!/usr/bin/env python

"""
Aiida interface for twinpy.
"""

from pprint import pprint
from copy import deepcopy
import numpy as np
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.orm import (load_node,
                       Bool,
                       Float,
                       Int,
                       KpointsData,
                       Node,
                       QueryBuilder,
                       )
from aiida.plugins import WorkflowFactory
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
class AiidaTwinBoudnaryShearWorkChain(_WorkChain):
    """
    TwinBoundaryShear work chain class.
    """

    def __init__(
            self,
            node:Node,
            ):
        """
        Args:
            node: TwinBoundaryShearWorkChain node.
        """
        process_class = 'TwinBoundaryShearWorkChain'
        check_process_class(node, process_class)
        super().__init__(node=node)
        self._shear_strain_ratios = None
        self._set_shear_strain_ratios()
        self._structure_pks = None
        self._set_structure_pks()
        self._aiida_relaxes = None
        self._set_aiida_relaxes()

    def _set_shear_strain_ratios(self):
        """
        Set shear strain ratios.
        """
        conf = self._node.inputs.twinboundary_shear_conf.get_dict()
        self._shear_strain_ratios = conf['shear_strain_ratios']

    @property
    def shear_strain_ratios(self):
        """
        Shear strain ratios.
        """
        return self._shear_strain_ratios

    def _set_structure_pks(self):
        """
        Set structure pks.
        """
        qb = QueryBuilder()
        qb.append(Node, filters={'id':{'==': self._pk}}, tag='wf')
        qb.append(
            Node,
            filters={'label': {'==': 'get_twinboundary_shear_structure'}},
            project=['id'],
            with_incoming='wf')
        cf_pks = [ q[0] for q in qb.all() ]
        orig_pks = []
        input_pks = []
        for pk in cf_pks:
            cf = load_node(pk)
            orig_pks.append(cf.outputs.twinboundary_shear_structure_orig.pk)
            input_pks.append(cf.outputs.twinboundary_shear_structure.pk)
        self._structure_pks = {
                'original_structures': orig_pks,
                'structures': input_pks,
                }

    @property
    def structure_pks(self):
        """
        Structure pks.
        """
        return self._structure_pks

    def _set_aiida_relaxes(self):
        """
        Set list of AiidaRelaxWorkChain objects.
        """
        relax_wf = WorkflowFactory('vasp.relax')
        qb = QueryBuilder()
        qb.append(Node, filters={'id':{'==': self._pk}}, tag='wf')
        qb.append(relax_wf, with_incoming='wf', project=['id'])
        rlx_pks = [ q[0] for q in qb.all() ]
        self._aiida_relaxes = [ AiidaRelaxWorkChain(load_node(pk)) for pk in rlx_pks ]

    @property
    def aiida_relaxes(self):
        """
        List of AiidaRelaxWorkChain class objects.
        """
        return self._aiida_relaxes
