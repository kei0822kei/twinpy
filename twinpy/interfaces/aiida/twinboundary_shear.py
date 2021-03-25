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
from twinpy.interfaces.aiida.base import (check_process_class,
                                          _WorkChain)
from twinpy.interfaces.aiida.vasp import (AiidaRelaxWorkChain)
from twinpy.interfaces.aiida.twinboundary \
        import AiidaTwinBoudnaryRelaxWorkChain


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
        self._shear_relax_pks = None
        self._aiida_relaxes = None
        self._set_aiida_relaxes()
        self._twinboundary_analyzer = None

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
        self._shear_relax_pks = rlx_pks
        self._aiida_relaxes = [ AiidaRelaxWorkChain(load_node(pk))
                                for pk in self._shear_relax_pks ]

    @property
    def aiida_relaxes(self):
        """
        List of AiidaRelaxWorkChain class objects.
        """
        return self._aiida_relaxes

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
        tb_rlx_pk = conf['twinboundary_relax_pk']
        addi_rlx_pks = conf['additional_relax_pks']

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
                                        twinboundary_phonon_pk:int=None,
                                        hexagonal_relax_pk:int=None,
                                        hexagonal_phonon_pk:int=None,
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
        tb_shear_analyzer = \
            tb_anal.get_twinboundary_shear_analyzer_from_relax_pks(
                shear_relax_pks=self._shear_relax_pks,
                shear_strain_ratios=self._shear_strain_ratios,
                shear_phonon_pks=shear_phonon_pks,
                )
        return tb_shear_analyzer
