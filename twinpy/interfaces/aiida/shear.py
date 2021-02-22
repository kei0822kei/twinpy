#!/usr/bin/env python

"""
Aiida interface for twinpy.
"""
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.plugins import WorkflowFactory
from aiida.orm import (load_node,
                       Node,
                       QueryBuilder,
                       StructureData,
                       CalcFunctionNode)
from twinpy.analysis.phonon_analyzer import PhononAnalyzer
from twinpy.analysis.shear_analyzer import ShearAnalyzer
from twinpy.interfaces.aiida.base import (check_process_class,
                                          get_cell_from_aiida,
                                          get_workflow_pks,
                                          _WorkChain)
from twinpy.interfaces.aiida.vasp import AiidaRelaxWorkChain
from twinpy.interfaces.aiida.phonopy import AiidaPhonopyWorkChain
from twinpy.properties.hexagonal import get_wyckoff_from_hcp
from twinpy.structure.shear import get_shear


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
        self._shear_ratios = node.outputs.shear_ratios['shear_ratios']
        self._gamma = node.outputs.gamma.value
        self._is_phonon = node.inputs.is_phonon.value

        self._create_shears_pk = None
        self._cells = None
        self._structure_pks = None
        self._set_shear()
        self._shear_structure = None
        self._set_shear_structure()

        self._relax_pks = None
        self._relaxes = None
        self._set_relaxes()

        self._phonon_pks = None
        self._phonons = None
        if self._is_phonon:
            self._set_phonons()

    def _set_shear_structure(self):
        """
        Set twinpy structure object.
        """
        twinmode = self._shear_conf['twinmode']
        lattice, _, symbols = self._cells['hexagonal']
        wyckoff = get_wyckoff_from_hcp(self._cells['hexagonal'])
        shear = get_shear(lattice=lattice,
                          symbol=symbols[0],
                          twinmode=twinmode,
                          wyckoff=wyckoff,
                          xshift=0.,
                          yshift=0.,
                          dim=[1,1,1],
                          shear_strain_ratio=0.0,
                          is_primitive=True)
        self._shear_structure = shear

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
    def shear_structure(self):
        """
        Twinpy structure class object.
        """
        return self._shear_structure

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
        phonon_analyzers = []
        for i, relax in enumerate(self._relaxes):
            relax_analyzer = relax.get_relax_analyzer(
                    original_cell=original_cells[i])
            phn = self._phonons[i].get_phonon()
            phonon_analyzer = PhononAnalyzer(phonon=phn,
                                             relax_analyzer=relax_analyzer)
            phonon_analyzers.append(phonon_analyzer)
        shear_analyzer = ShearAnalyzer(shear_structure=self._shear_structure,
                                       phonon_analyzers=phonon_analyzers)
        return shear_analyzer

    def get_pks(self) -> dict:
        """
        Get pks.

        Returns:
            dict: keys and corresponding pks
        """
        pks = {
                'shear_pk': self._pk,
                'shear_structures_pk': self._create_shears_pk,
                'relax_pks': self._relax_pks,
                'phonon_pks': self._phonon_pks,
              }
        return pks
