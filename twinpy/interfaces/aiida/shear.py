#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aiida interface for twinpy.
"""
from twinpy.api_twinpy import get_twinpy_from_cell
from twinpy.analysis.relax_analyzer import RelaxAnalyzer
from twinpy.analysis.shear_analyzer import ShearAnalyzer
from twinpy.interfaces.aiida import (check_process_class,
                                     get_cell_from_aiida,
                                     get_workflow_pks,
                                     _WorkChain,
                                     AiidaRelaxWorkChain,
                                     AiidaPhonopyWorkChain)
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.plugins import WorkflowFactory
from aiida.orm import (load_node,
                       Node,
                       QueryBuilder,
                       StructureData,
                       CalcFunctionNode)


@with_dbenv()
class AiidaShearWorkChain(_WorkChain):
    """
    Shear work chain class.
    """

    def __init__(
            self,
            node:Node,
            ):
        """
        Args:
            node: ShearWorkChain node

        Todo:
            Replace shear_conf ot shear_settings as
            AiidaTwinBoundaryRelaxWorkChain class.
        """
        process_class = 'ShearWorkChain'
        check_process_class(node, process_class)
        super().__init__(node=node)

        self._shear_conf = node.inputs.shear_conf.get_dict()
        self._shear_ratios = \
            node.called[-1].outputs.shear_settings.get_dict()['shear_ratios']
        self._gamma = node.outputs.gamma.value
        self._is_phonon = node.inputs.is_phonon.value

        self._create_shears_pk = None
        self._cells = None
        self._structure_pks = None
        self._set_shear()
        self._twinpy = None
        self._set_twinpy()

        self._relax_pks = None
        self._relaxes = None
        self._set_relaxes()

        self._phonon_pks = None
        self._phonons = None
        if self._is_phonon:
            self._set_phonons()

    def _set_twinpy(self):
        """
        Set twinpy structure object.
        """
        twinmode = self._shear_conf['twinmode']
        cell = self._cells['hexagonal']
        twinpy = get_twinpy_from_cell(
                cell=cell,
                twinmode=twinmode)
        twinpy.set_shear(is_primitive=True)
        self._twinpy = twinpy

    @property
    def shear_conf(self):
        """
        Input shear conf.
        """
        return self._shear_conf

    @property
    def shear_ratios(self):
        """
        Output shear ratios.
        """
        return self._shear_ratios

    @property
    def gamma(self):
        """
        Output gamma.
        """
        return self._gamma

    @property
    def is_phonon(self):
        """
        Input is_phonon.
        """
        return self._is_phonon

    @property
    def cells(self):
        """
        Cells.
        """
        return self._cells

    @property
    def twinpy(self):
        """
        Twinpy structure class object.
        """
        return self._twinpy

    @property
    def structure_pks(self):
        """
        Structure pks.
        """
        return self._structure_pks

    def _set_shear(self):
        """
        Set ShearWorkChain data.
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
        self._structure_pks = {}
        self._structure_pks['shear_original_pks'] = orig_cell_pks
        self._structure_pks['hexagonal_pk'] = self._node.inputs.structure
        self._cells = {}
        self._cells['hexagonal'] = \
                get_cell_from_aiida(self._node.inputs.structure)
        self._cells['shear_original'] = \
                [ get_cell_from_aiida(load_node(pk))
                      for pk in orig_cell_pks ]

    def _set_relaxes(self):
        """
        Set relax in ShearWorkChain.
        """
        relax_wf = WorkflowFactory('vasp.relax')
        rlx_pks = get_workflow_pks(node=self._node,
                                   workflow=relax_wf)
        self._relax_pks = rlx_pks
        self._relaxes = [ AiidaRelaxWorkChain(node=load_node(pk))
                              for pk in self._relax_pks ]

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
        phonon_wf = WorkflowFactory('phonopy.phonopy')
        self._phonon_pks = get_workflow_pks(node=self._node,
                                            workflow=phonon_wf)
        self._phonons = [ AiidaPhonopyWorkChain(node=load_node(pk))
                              for pk in self._phonon_pks ]

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

    def get_shear_analyzer(self) -> ShearAnalyzer:
        """
        Get ShearAnalyzer class object.
        """
        original_cells = self._cells['shear_original']
        relax_analyzers = []
        for i, relax in enumerate(self._relaxes):
            relax_analyzer = relax.get_relax_analyzer(
                    original_cell=original_cells[i])
            relax_analyzers.append(relax_analyzer)
        phns = [ phonon_wf.get_phonon() for phonon_wf in self._phonons ]
        analyzer = ShearAnalyzer(shear_structure=self._twinpy,
                                 relax_analyzers=relax_analyzers,
                                 phonons=phns)
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
                'relax_pks': self._relax_pks,
                'phonon_pks': self._phonon_pks,
              }
        return pks
