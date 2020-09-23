#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aiida interface for twinpy.
"""
from pprint import pprint
import warnings
from twinpy.interfaces.aiida import (check_process_class,
                                     get_cell_from_aiida)
from twinpy.interfaces.phonopy import get_phonopy_structure
from twinpy.common.utils import print_header
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.common import NotExistentAttributeError
from aiida.orm import (load_node,
                       Node)
from phonopy import Phonopy

from twinpy.interfaces.aiida import _WorkChain


@with_dbenv()
class AiidaPhonopyWorkChain(_WorkChain):
    """
    Phononpy work chain class.
    """
    def __init__(
            self,
            node:Node,
            ):
        """
        Args:
            node: aiida Node
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
        self._relax = None

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

    def set_relax(self, relax:Node):
        """
        Set relax

        Args:
            relax: AiidaRelaxWorkChain class object.
        """
        relax_structure_pk = relax.get_pks()['current_final_structure_pk']
        phonopy_structure_pk = self._structure_pks['unitcell_pk']
        assert relax_structure_pk == phonopy_structure_pk, \
                 "relax structure and phonopy input structure is not match. " \
                 "relax structure: {}, phonopy structure: {}".format(
                     relax_structure_pk, phonopy_structure_pk)
        self._relax = relax

    @property
    def relax(self):
        """
        Output AiidaRelaxWorkChain class.
        """
        return self._relax

    def get_pks(self) -> dict:
        """
        Get pks.

        Returns:
            dict: containing relax pk and structure pk
        """
        pks = self._structure_pks.copy()
        pks.update({'phonopy_pk': self._pk})
        if self._relax is not None:
            pks.update({'relax_pk': self._relax.pk})
        return pks

    def get_phonon(self) -> Phonopy:
        """
        Get phonopy object.

        Returns:
            Phonopy: phonopy object
        """
        phonon = Phonopy(
                get_phonopy_structure(self._unitcell),
                supercell_matrix=self._phonon_setting_info['supercell_matrix'],
                primitive_matrix=self._phonon_setting_info['primitive_matrix'])
        phonon.set_displacement_dataset(
                self._phonon_setting_info['displacement_dataset'])
        phonon.set_forces(self._force_sets)
        phonon.produce_force_constants()
        return phonon

    def export_phonon(self, filename:str=None):
        """
        Export phonopy object to yaml file.

        Args:
            filename (str): Output filename. If None, filename becomes
                            pk<number>_phonopy.yaml.
        """
        phonon = self.get_phonon()
        if filename is None:
            filename = 'pk%d_phonopy.yaml' % self._pk
        phonon.save(filename)

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
