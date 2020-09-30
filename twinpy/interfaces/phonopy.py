#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interfaces for Phonopy
"""

from phonopy.structure.atoms import PhonopyAtoms


def get_phonopy_structure(cell:tuple) -> PhonopyAtoms:
    """
    Return phonopy structure.

    Args:
        cell: tuple (lattice, scaled_positions, symbols)

    Returns:
        PhonopyAtoms: structure
    """
    ph_structure = PhonopyAtoms(cell=cell[0],
                                scaled_positions=cell[1],
                                symbols=cell[2])
    return ph_structure


def get_cell_from_phonopy_structure(ph_structure:PhonopyAtoms,
                                    use_atomic_number:bool=False) -> tuple:
    """
    Get cell from phonopy structure

    Args:
        ph_structure: PhonopyAtoms object
        use_atomic_number: if True, use atomic number intead of atomic symbol

    Returns:
        tuple: cell
    """
    lattice = ph_structure.get_cell()
    scaled_positions = ph_structure.get_scaled_positions()
    if use_atomic_number:
        elements = list(ph_structure.get_atomic_numbers())
    else:
        elements = ph_structure.get_chemical_symbols()
    return (lattice, scaled_positions, elements)
