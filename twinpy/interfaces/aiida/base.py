#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interface for Aiida Node.
"""
import numpy as np
import warnings
from aiida.orm import Node, QueryBuilder, StructureData
from twinpy.common.utils import print_header


def load_aiida_profile():
    """
    Load aiida profile.

    Raises:
        ProfileConfigurationError: Fail to load aiida profile.
    """
    from aiida import load_profile
    from aiida.common.exceptions import ProfileConfigurationError

    try:
        load_profile()
    except ProfileConfigurationError:
        err_msg = "Failed to load aiida profile. " \
                + "Please check your aiida configuration."
        warnings.warn(err_msg)


def check_process_class(node:Node,
                        expected_process_class:str):
    """
    Check process class of node is the same as the expected.

    Args:
        node: aiida node
        expected_process_class (str): expected process class

    Raises:
        AssertionError: input node is not the same as expected
    """
    assert node.process_class.get_name() == expected_process_class, \
            "input node: {}, expected: {} (NOT MATCH)". \
            format(node.process_class, expected_process_class)


def get_aiida_structure(cell:tuple) -> StructureData:
    """
    Get aiida structure from input cell.

    Args:
        cell (tuple): (lattice, scaled_positions, symbols).

    Returns:
        StructureData: Aiida structure data.
    """
    structure = StructureData(cell=cell[0])
    for symbol, scaled_position in zip(cell[2], cell[1]):
        position = np.dot(cell[0].T,
                          scaled_position.reshape(3,1)).reshape(3)
        structure.append_atom(position=position, symbols=symbol)
    return structure


def get_workflow_pks(node, workflow) -> list:
    """
    Get workflow pks in the node.

    Args:
        node: node
        workflow: Workflow, ex. workflow = WorkflowFactory('vasp.relax').

    Returns:
        list: PKs.
    """
    qb = QueryBuilder()
    qb.append(Node, filters={'id':{'==': node.pk}}, tag='wf')
    qb.append(workflow, with_incoming='wf', project=['id'])
    pks = [ wf[0] for wf in qb.all() ]
    pks.sort(key=lambda x: x)
    return pks


def get_cell_from_aiida(structure:StructureData,
                        get_scaled_positions:bool=True) -> tuple:
    """
    Get cell from input aiida structure.

    Args:
        structure (StructureData): Aiida structure data.
        get_scaled_positions (bool): If True, return scaled positions.

    Returns:
        tuple: (lattice, positions, symbols).
    """
    lattice = np.array(structure.cell)
    positions = np.array([ site.position for site in structure.sites ])
    if get_scaled_positions:
        positions = np.dot(np.linalg.inv(lattice.T), positions.T).T
    symbols = [ site.kind_name for site in structure.sites ]
    return (lattice, positions, symbols)


class _WorkChain():
    """
    Aiida WorkChain class.
    """

    def __init__(
            self,
            node:Node,
            ):
        """
        Args:
            node: Aiida Node.
        """
        self._node = node
        self._process_class = self._node.process_class.get_name()
        self._process_state = self._node.process_state.value
        self._pk = node.pk
        self._label = self._node.label
        self._description = self._node.description
        self._exit_status = self._node.exit_status
        if self._process_state != 'finished':
            warnings.warn("process state: %s" % self._process_state)
        else:
            if self._exit_status != 0:
                warnings.warn(
                    "Warning: exit status was {}".format(self._exit_status))

    @property
    def process_class(self):
        """
        Process class.
        """
        return self._process_class

    @property
    def process_state(self):
        """
        Process state.
        """
        return self._process_state

    @property
    def node(self):
        """
        WorkChain node.
        """
        return self._node

    @property
    def pk(self):
        """
        WorkChain pk.
        """
        return self._pk

    @property
    def label(self):
        """
        Label.
        """
        return self._label

    @property
    def description(self):
        """
        Description.
        """
        return self._description

    @property
    def exit_status(self):
        """
        Exit status.
        """
        return self._exit_status

    def _print_common_information(self):
        print_header('About This Node')
        print("process class:%s" % self._process_class)
        print("process state:%s" % self._process_state)
        print("pk:%s" % self._pk)
        print("label:%s" % self._label)
        print("description:%s" % self._description)
        print("exit status:%s" % self._exit_status)
        print("\n\n")
