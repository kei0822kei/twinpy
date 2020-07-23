#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pymatgen.core.structure import Structure
from twinpy.structure.base import is_hcp


def get_pymatgen_structure(cell:tuple) -> Structure:
    """
    Get pymatgen structure from cell.

    Args:
        cell (tuple): cell
    """
    return Structure(lattice=cell[0],
                     coords=cell[1],
                     species=cell[2])


def get_cell_from_pymatgen_structure(pmgstructure:Structure) -> tuple:
    """
    Get cell from pymatgen.

    Args:
        pmgstructure (Structure): pymatgen structure

    Returns:
        tuple: cell
    """
    lattice = pmgstructure.lattice.matrix
    scaled_positions = pmgstructure.frac_coords
    symbols = [ specie.value for specie in pmgstructure.species ]
    return (lattice, scaled_positions, symbols)


def get_hexagonal_cell_wyckoff_from_pymatgen(pmgstructure:Structure) -> tuple:
    """
    Get hexagonal cell and wyckoff letter from pyamtgen structure.

    Args:
        pmgstructure: pymatgen structure object

    Returns:
        tuple: (lattice, scaled_positions, symbol, wyckoff)
    """
    lattice, scaled_positions, symbols = get_pymatgen_structure(pmgstructure)
    wyckoff = is_hcp(lattice=lattice,
                     scaled_positions=scaled_positions,
                     symbols=symbols,
                     get_wyckoff=True)
    return (lattice, scaled_positions, symbols[0], wyckoff)
