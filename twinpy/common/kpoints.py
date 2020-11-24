#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deals with kpoints.
"""

import numpy as np
from twinpy.lattice.lattice import Lattice


def _get_mesh_from_interval(lattice:np.array,
                            interval:float) -> dict:
    """
    Get mesh from interval.

    Args:
        lattice (np.array): lattice matrix
        interval (float): grid interval

    Returns:
        dict: containing abc norms, mesh

    Note:
        mesh * interval => abc
        If even becomes zero, fix to one.
    """
    lat = Lattice(lattice=lattice)
    abc = lat.abc
    mesh_float = abc / interval
    mesh = np.int64(np.round(mesh_float))
    fixed_mesh = np.where(mesh==0, 1, mesh)
    return {'abc': abc, 'mesh': fixed_mesh}


def _get_intervals_from_mesh(lattice:np.array,
                             mesh:list) -> dict:
    """
    Get interval from mesh.

    Args:
        lattice (np.array): lattice matrix
        mesh (list): the number of mesh of each axis

    Returns:
        dict: containing abc norms, intervals
    """
    lat = Lattice(lattice=lattice)
    abc = lat.abc
    intervals = abc / np.array(mesh)
    return {'abc': abc, 'intervals': intervals}


def get_mesh_offset_from_direct_lattice(lattice:np.array,
                                        interval:float=None,
                                        mesh:list=None,
                                        include_two_pi:bool=True) -> dict:
    """
    Get kpoints mesh and offset from input lattice and interval.

    Args:
        lattice (np.array): lattice matrix
        interval (float): grid interval
        mesh (list): mesh
        include_two_pi (bool): if True, include 2 * pi

    Returns:
        dict: containing abc norms, mesh, offset

    Raises:
        ValueError: both mesh and interval are not specified
        ValueError: both mesh and interval are specified

    Note:
        Please input 'interval' or 'mesh', not both.
        If the angles of input lattice is (90., 90., 120.),
        offset is set (0., 0., 0.5) and mesh is set
        (odd, odd, even or 1).
        If even becomes zero, fix to one.
        Otherwise, set (0.5, 0.5, 0.5) and mesh is set
        (even or 1, even or 1, even or 1).
        If you use this function, it is better to input
        STANDARDIZED CELL.
    """
    if interval is None and mesh is None:
        raise ValueError("both mesh and interval are not specified")
    if interval is not None and mesh is not None:
        raise ValueError("both mesh and interval are specified")

    lat = Lattice(lattice=lattice)

    if include_two_pi:
        recip_lat = Lattice(2*np.pi*lat.reciprocal_lattice)
    else:
        recip_lat = Lattice(lat.reciprocal_lattice)

    if interval is not None:
        mesh = _get_mesh_from_interval(lattice=recip_lat.lattice,
                                       interval=interval)['mesh']
    else:
        mesh = np.int64(mesh)

    # is hexagonal standardized cell
    recip_a, recip_b, _ = recip_lat.abc
    is_hexagonal = (np.allclose(recip_lat.angles, (90., 90., 60.),
                                rtol=0., atol=1e-5)
                    and np.allclose(recip_a, recip_b,
                                    rtol=0., atol=1e-5)
                    )

    # fix mesh from get_mesh_from_interval
    if is_hexagonal:
        offset = (0., 0., 0.5)
        # If True, get 1, if False get 0.
        condition = lambda x: int(x%2==0)
        arr = [ condition(m) for m in mesh[:2] ]
        if (mesh[2]!=1 and mesh[2]%2==1):
            arr.append(1)
        else:
            arr.append(0)
        arr = np.array(arr)
    else:
        offset = (0.5, 0.5, 0.5)
        # condition = lambda x: int(x!=1 and x%2==1)
        condition = lambda x: int(x%2==1)
        arr = np.array([ condition(m) for m in mesh ])
    fixed_mesh = mesh + arr

    kpts = _get_intervals_from_mesh(
               lattice=recip_lat.lattice, mesh=fixed_mesh)
    kpts['reciprocal_lattice'] = recip_lat.lattice
    kpts['reciprocal_volume'] = recip_lat.volume
    kpts['mesh'] = fixed_mesh
    kpts['total_mesh'] = fixed_mesh[0] * fixed_mesh[1] * fixed_mesh[2]
    kpts['offset'] = offset
    kpts['is_hexagonal'] = is_hexagonal
    kpts['include_two_pi'] = include_two_pi

    return kpts
