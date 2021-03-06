#!/usr/bin/env python

"""
Aiida interface for twinpy.
"""
from pprint import pprint
import warnings
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.common import NotExistentAttributeError
from aiida.orm import (load_node,
                       Node)
from phonopy import Phonopy
from twinpy.interfaces.aiida.base import (check_process_class,
                                          get_cell_from_aiida,
                                          _WorkChain)
from twinpy.interfaces.phonopy import get_phonopy_structure
from twinpy.common.utils import print_header
from twinpy.analysis.phonon_analyzer import (RelaxAnalyzer,
                                             PhononAnalyzer)


@with_dbenv()
class AiidaPhonopyWorkChain(_WorkChain):
    """
    Phononpy work chain class.
    """
    def __init__(
            self,
            node:Node,
            is_nac:bool=True,
            external_nac_node:int=None,
            ):
        """
        Args:
            node: Aiida Node.
            external_nac_node: If is_nac=True and external_nac_node is
                               specified, nac params are gotten from
                               external_nac_node.
        """
        process_class = 'PhonopyWorkChain'
        check_process_class(node, process_class)
        super().__init__(node=node)
        self._unitcell = get_cell_from_aiida(
                load_node(node.inputs.structure.pk))
        self._phonon_settings = node.inputs.phonon_settings.get_dict()
        self._phonon_setting_info = node.outputs.phonon_setting_info.get_dict()
        self._unitcell = None
        self._primitive = None
        self._supercell = None
        self._structure_pks = None
        self._set_structures()
        self._force_sets = None
        self._set_force_sets()
        self._nac_params = None
        if external_nac_node:
            self._set_nac_params_from_external(external_nac_node)
        elif 'nac_params' in self._node.outputs and is_nac:
            self._set_nac_params()

    def _set_structures(self):
        """
        Set structures.
        """
        self._unitcell = get_cell_from_aiida(self._node.inputs.structure)
        self._primitive = get_cell_from_aiida(self._node.outputs.primitive)
        self._supercell = get_cell_from_aiida(self._node.outputs.supercell)
        self._structure_pks = {
                'unitcell_pk': self._node.inputs.structure.pk,
                'primitive_pk': self._node.outputs.primitive.pk,
                'supercell_pk': self._node.outputs.supercell.pk,
                }

    @property
    def unitcell(self):
        """
        Input unitcell.
        """
        return self._unitcell

    @property
    def primitive(self):
        """
        Output primitive cell.
        """
        return self._primitive

    @property
    def supercell(self):
        """
        Output supercell.
        """
        return self._supercell

    @property
    def structure_pks(self):
        """
        Structure pks.
        """
        return self._structure_pks

    def _set_force_sets(self):
        """
        Set force_sets.
        """
        try:
            self._force_sets = \
                    self._node.outputs.force_sets.get_array('force_sets')
        except NotExistentAttributeError:
            warnings.warn("Could not find force sets. Probably, "
                          "this job still running or finished improperly.\n"
                          "process state: {} (pk={})".format(
                              self._process_state, self._pk))

    @property
    def phonon_settings(self):
        """
        Input phonon settings.
        """
        return self._phonon_settings

    @property
    def phonon_setting_info(self):
        """
        Output phonon setting_info.
        """
        return self._phonon_setting_info

    @property
    def force_sets(self):
        """
        Output force sets.
        """
        return self._force_sets

    def _set_nac_params_from_external(self, nac_node):
        """
        Set nac params from external nac node.
        """
        from aiida_phonopy.common.utils import get_nac_params
        from aiida.orm import Float
        params = get_nac_params(
            born_charges=nac_node.outputs.born_charges,
            epsilon=nac_node.outputs.dielectrics,
            nac_structure=nac_node.inputs.structure,
            symmetry_tolerance=Float(1e-5),
            )
        # borns = nac_node.outputs.born_charges.get_array('born_charges')
        # epsilon = nac_node.outputs.dielectrics.get_array('epsilon')
        borns = params.get_array('born_charges')
        epsilon = params.get_array('epsilon')
        nac_params = {'born': borns,
                      'factor': 14.399652,
                      'dielectric': epsilon}
        self._nac_params = nac_params

    def _set_nac_params(self):
        """
        Set nac params.
        """
        borns = self._node.outputs.nac_params.get_array('born_charges')
        epsilon = self._node.outputs.nac_params.get_array('epsilon')
        nac_params = {'born': borns,
                      'factor': 14.399652,
                      'dielectric': epsilon}
        self._nac_params = nac_params

    @property
    def nac_params(self):
        """
        Nac params.
        """
        return self._nac_params


    def get_pks(self) -> dict:
        """
        Get pks.

        Returns:
            dict: Containing phonopy pk and structure pk.
        """
        pks = self._structure_pks.copy()
        pks.update({'phonopy_pk': self._pk})

        return pks

    def get_phonon(self) -> Phonopy:
        """
        Get phonopy object.

        Returns:
            Phonopy: Phonopy class object.
        """
        phonon = Phonopy(
                get_phonopy_structure(self._unitcell),
                supercell_matrix=self._phonon_setting_info['supercell_matrix'],
                primitive_matrix=self._phonon_setting_info['primitive_matrix'],
                nac_params=self._nac_params,
                )
        phonon.set_displacement_dataset(
                self._phonon_setting_info['displacement_dataset'])
        phonon.set_forces(self._force_sets)
        phonon.produce_force_constants()

        return phonon

    def get_phonon_analyzer(self,
                            relax_analyzer:RelaxAnalyzer=None):
        """
        Get PhononAnalyzer class object.

        Args:
            relax_analyzer: RelaxAnalyzer class object.
        """
        analyzer = PhononAnalyzer(phonon=self.get_phonon(),
                                  relax_analyzer=relax_analyzer)
        return analyzer

    def get_description(self):
        """
        Get description.
        """
        self._print_common_information()
        print_header('PKs')
        pprint(self.get_pks())
        print("\n\n")
        print_header('phonopy settings')
        pprint(self._phonon_settings)
