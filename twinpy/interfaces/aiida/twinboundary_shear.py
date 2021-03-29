#!/usr/bin/env python

"""
Aiida interface for twinpy.
"""

from aiida.cmdline.utils.decorators import with_dbenv
from aiida.orm import (load_node,
                       Node,
                       QueryBuilder,
                       )
from aiida.plugins import WorkflowFactory
from aiida_twinpy.common.utils import get_create_node
from twinpy.interfaces.aiida.base import (check_process_class,
                                          _WorkChain)
from twinpy.interfaces.aiida.vasp import (AiidaRelaxWorkChain)
from twinpy.interfaces.aiida.twinboundary \
        import AiidaTwinBoudnaryRelaxWorkChain


RLX_WF = WorkflowFactory('vasp.relax')

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
        self._shear_aiida_relaxes = None
        self._set_shear_aiida_relaxes()
        self._structure_pks = None
        self._set_structure_pks()
        self._aiida_twinboundary_relax = None
        self._set_aiida_twinboundary_relax()
        self._additional_relax_pks = None
        self._set_additional_relax_pks()

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

        rlx_pks = []
        for aiida_rlx, i_struct_pk in zip(self.shear_aiida_relaxes, input_pks):
            pks = aiida_rlx.get_pks()
            assert pks['initial_structure_pk'] == i_struct_pk, \
                    "Input structure does not match."
            rlx_pks.append(pks['final_structure_pk'])

        self._structure_pks = {
                'original_structures': orig_pks,
                'input_structures': input_pks,
                'relax_structures': rlx_pks,
                }

    @property
    def structure_pks(self):
        """
        Structure pks.
        """
        return self._structure_pks

    def _set_aiida_twinboundary_relax(self):
        """
        Set twinboundary relax pk.
        """
        tb_RLX_WF = WorkflowFactory('twinpy.twinboundary_relax')
        tb_rlx_struct_pk = self._node.inputs.twinboundary_relax_structure.pk
        tb_rlx = get_create_node(tb_rlx_struct_pk, tb_RLX_WF)
        self._aiida_twinboundary_relax \
                = AiidaTwinBoudnaryRelaxWorkChain(tb_rlx)

    def _set_shear_aiida_relaxes(self):
        """
        Set list of AiidaRelaxWorkChain objects.
        """
        RLX_WF = WorkflowFactory('vasp.relax')
        qb = QueryBuilder()
        qb.append(Node, filters={'id':{'==': self._pk}}, tag='wf')
        qb.append(RLX_WF, with_incoming='wf', project=['id'])
        rlx_pks = [ q[0] for q in qb.all() ]
        self._shear_aiida_relaxes = [ AiidaRelaxWorkChain(load_node(pk))
                                         for pk in rlx_pks ]

    def _set_additional_relax_pks(self):
        """
        Set additional relax pks.
        """
        addi_struct_pks = [ self._node.inputs.__getattr__(key).pk
                              for key in dir(self._node.inputs)
                                if 'additional_relax__structure' in key ]
        self._additional_relax_pks = \
                [ get_create_node(pk, RLX_WF).pk for pk in addi_struct_pks ]

    @property
    def shear_aiida_relaxes(self):
        """
        List of AiidaRelaxWorkChain class objects.
        """
        return self._shear_aiida_relaxes

    def set_twinboundary_analyzer(self,
                                  twinboundary_phonon_pk:int=None,
                                  hexagonal_relax_pk:int=None,
                                  hexagonal_phonon_pk:int=None,
                                  ):
        """
        Set twinboundary analyzer.

        Args:
            twinboudnary_phonon_pk: Twinboundary phonon calculation pk.
            hexagonal_relax_pk: Hexagonal relax calculation pk.
            hexagonal_phonon_pk: Hexagonal phonon calculation pk.
        """
        conf = self._node.inputs.twinboundary_shear_conf.get_dict()
        tb_rlx_pk = self._aiida_twinboundary_relax.pk
        addi_rlx_pks = self._additional_relax_pks

        aiida_tb = AiidaTwinBoudnaryRelaxWorkChain(load_node(tb_rlx_pk))
        self._twinboundary_analyzer = aiida_tb.get_twinboundary_analyzer(
                twinboundary_phonon_pk=twinboundary_phonon_pk,
                additional_relax_pks=addi_rlx_pks,
                hexagonal_relax_pk=hexagonal_relax_pk,
                hexagonal_phonon_pk=hexagonal_phonon_pk,
                )

    @property
    def twinboundary_analyzer(self):
        """
        TwinBoundaryAnalyzer class object.
        """
        return self._twinboundary_analyzer

    def get_twinboundary_shear_analyzer(self,
                                        shear_phonon_pks:list,
                                        ):
        """
        Get twinboundary shear analyzer.

        Args:
            shaer_phonon_pks: List of phonon pks.

        Raises:
            RuntimeError: Property twinboundary_analyzer is not set.

        Note:
            Length of phono_pks list must be the same as that of shear strain
            ratios. If there is no phonon result, set please set None.
        """
        if self._twinboundary_analyzer is None:
            raise RuntimeError("Please set twinboundary_analyzer before.")

        tb_anal = self._twinboundary_analyzer
        shr_rlx_pks = [ aiida_rlx.pk for aiida_rlx in self._shear_aiida_relaxes ]
        ratios = self.shear_strain_ratios

        tb_shear_analyzer = \
            tb_anal.get_twinboundary_shear_analyzer_from_relax_pks(
                shear_relax_pks=shr_rlx_pks,
                shear_strain_ratios=ratios,
                shear_phonon_pks=shear_phonon_pks,
                )
        return tb_shear_analyzer

    def get_pks(self):
        """
        Get workflow pks.

        Returns:
            dict: Workflow pks.
        """
        wf_pks = {
            'twinboundary_relax_pk': self._aiida_twinboundary_relax.pk,
            'additional_relax_pks': self._additional_relax_pks,
            'shear_aiida_relax_pks': [ shr_rlx.pk for shr_rlx
                                           in self._shear_aiida_relaxes ],
            }
        return wf_pks
