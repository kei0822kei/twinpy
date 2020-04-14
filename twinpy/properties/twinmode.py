#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
provide properties of twin modes
"""

import numpy as np
from pymatgen.core.lattice import Lattice
from twinpy.hexagonal import (get_atom_positions,
                              HexagonalPlane,
                              HexagonalDirection)
from twinpy.utils import get_ratio

def is_supported_twinmode(twinmode):
    """
    check input twinmode is supported

    Args:
        twinmode (str): choose from '10-12', '10-11', '11-22' or '11-21'
        (which are supported)
    """
    supported_twinmodes = ['10-12', '10-11', '11-22', '11-21']
    assert twinmode in supported_twinmodes, '%s is not supported' % twinmode

def get_shear_strain_function(twinmode:str) -> 'function':
    """
    get shear strain

    Args:
        twinmode (str): choose from '10-12', '10-11', '11-22' or '11-21'
            (which are supported)

    Returns:
        function: function which returns shear strain
                  input args: r
    """
    if twinmode == '10-12':
        func = lambda r: \
                ( (r**2-3) / (r*np.sqrt(3)) )
    elif twinmode == '10-11':
        func = lambda r: \
                ( (4*r**2-9) / 4*r*np.sqrt(3) )
    elif twinmode == '11-22':
        func = lambda r: \
                ( (2*(r**2-2)) / 3*r )
    elif twinmode == '11-21':
        func = lambda r: \
                ( 1 / r )
    else:
        is_supported_twinmode(twinmode)
    return func

def get_twin_indices(twinmode:str,
                     lattice:Lattice,
                     wyckoff:str) -> dict:
    """
    get twin indices of input twinmode
    twinmode must be '10-11', '10-12', '11-21' or '11-22'

    Args:
        this is used when K1 plane is choosed
        twinmode (str): currently supported
        '10-11', '10-12', '11-22' and '11-21'
        lattice (pymatgen.core.lattice.Lattice): lattice

    Returns:
        dict: twin indices

    Note:
        You can find customs how to set the four indices
        in the paper 'DEFORMATION TWINNING' by Christian (1995).
        Abstruct is as bellow:
        (1) Vector m normal to shear plane,
        eta1 and eta2 form right hand system.
        (2) The angle between eta1 and eta2 are abtuse.
        (3) The angle between eta1 and k2 are acute.
        (4) The angle between eta2 and k1 are acute.
        Additional rule:
        (5) K1 plane depends on wyckoff 'c' or 'd'
        because in the case of {10-12}, either (10-12) or (-1012)
        is choosed based on which plane are nearer to the neighbor
        atoms

        in this algorithm, indices are set in order of:
        => k1    condition(5)
        => eta2  condition(4)
        => eta1  condition(2)
        => k2    condition(3)
        => m     condition(1)

    Todo:
        _twin_indices_orig is a little bit different from the ones 1981. Yoo
        especially 'S'
    """
    def _twin_indices_orig(lattice, twinmode, wyckoff):
        dataset = {
                '10-12': {
                            'S'    : np.array([  1.,  1., -2.,  0.]),
                            'K1'   : np.array([  1.,  0., -1.,  2.]),
                            'K2'   : np.array([  1.,  0., -1., -2.]),
                            'eta1' : np.array([  1.,  0., -1., -1.]),
                            'eta2' : np.array([ -1.,  0.,  1., -1.]),
                         },

                '10-11': {
                            'S'    : np.array([  1.,  1., -2.,  0.]),
                            'K1'   : np.array([  1.,  0., -1.,  1.]),
                            'K2'   : np.array([  1.,  0., -1., -3.]),
                            'eta1' : np.array([  1.,  0., -1., -2.]),
                            'eta2' : np.array([  3.,  0., -3.,  2.]),
                         },

                '11-21': {
                            'S'    : np.array([  1.,  0., -1.,  0.]),
                            'K1'   : np.array([  1.,  1., -2.,  1.]),
                            'K2'   : np.array([  0.,  0.,  0.,  2.]),
                            'eta1' : np.array([-1/3,-1/3, 2/3,  2.]),
                            'eta2' : np.array([ 1/3, 1/3,-2/3,  0.]),
                         },
                '11-22': {
                            'S'    : np.array([  1.,  0., -1.,  0.]),
                            'K1'   : np.array([  1.,  1., -2.,  2.]),
                            'K2'   : np.array([  1.,  1., -2., -4.]),
                            'eta1' : np.array([ 1/3, 1/3,-2/3, -1.]),
                            'eta2' : np.array([ 2/3, 2/3,-4/3,  1.]),
                         },
                  }
        array = np.array([-1,-1,-1,1])
        atoms = get_atom_positions(wyckoff)
        K1_1 = HexagonalPlane(lattice=lattice,
                              four=dataset[twinmode]['K1'])
        K1_2 = HexagonalPlane(lattice=lattice,
                              four=dataset[twinmode]['K1'] * array )
        if K1_1.get_distance_from_plane(atoms[0]) < \
               K1_2.get_distance_from_plane(atoms[0]):
            data = dataset[twinmode]
        else:
            keys = [ 'S', 'K1', 'K2', 'eta1', 'eta2' ]
            data = {}
            for key in keys:
                data[key] = dataset[twinmode][key] * array

        indices = {
                     'S'    :     HexagonalPlane(lattice, four=data['S']),
                     'K1'   :     HexagonalPlane(lattice, four=data['K1']),
                     'K2'   :     HexagonalPlane(lattice, four=data['K2']),
                     'eta1' : HexagonalDirection(lattice, four=data['eta1']),
                     'eta2' : HexagonalDirection(lattice, four=data['eta2']),
                  }
        indices['k1'] = indices['K1'].get_direction_normal_to_plane()
        indices['k2'] = indices['K2'].get_direction_normal_to_plane()
        return indices

    def _set_K1(lattice, indices, wyckoff):
        # reset K1
        atoms = get_atom_positions(wyckoff=wyckoff)
        candidate_three = indices['K1'].three * np.array([-1, -1, 1])
        candidate_plane = HexagonalPlane(lattice=lattice,
                                         three=candidate_three)
        candidate_distancee = candidate_plane.get_distance_from_plane(atoms[0])
        distance = indices['K1'].get_distance_from_plane(atoms[0])
        if candidate_distancee < distance:
            indices['K1'] = candidate_plane
            indices['k1'] = indices['K1'].get_direction_normal_to_plane()
        return indices

    def _reset_indices(lattice, indices):
        # reset eta2
        if lattice.dot(indices['k1'].three,
                       indices['eta2'].three,
                       frac_coords=True)[0] < 0:
            indices['eta2'].inverse()

        # reset eta1
        if lattice.dot(indices['eta1'].three,
                       indices['eta2'].three,
                       frac_coords=True)[0] > 0:
            indices['eta1'].inverse()

        # reset K2
        if lattice.dot(indices['eta1'].three,
                       indices['k2'].three,
                       frac_coords=True)[0] < 0:
            indices['K2'].inverse()
            indices['k2'].inverse()

        # check k1 is orthogonal to eta1
        np.testing.assert_allclose(
                actual=lattice.dot(indices['k1'].three,
                                   indices['eta1'].three,
                                   frac_coords=True)[0],
                desired=0,
                atol=1e-6,
                err_msg="k1 is not orthogonal to eta1")

        # check k2 is orthogonal to eta2
        np.testing.assert_allclose(
                actual=lattice.dot(indices['k2'].three,
                                   indices['eta2'].three,
                                   frac_coords=True)[0],
                desired=0,
                atol=1e-6,
                err_msg="k2 is not orthogonal to eta2")

        return indices

    def _set_shear_plane(lattice, indices):
        S_four = indices['S'].four
        trial_S_fours = [
              [S_four[0],S_four[1],S_four[2],S_four[3]],
              [S_four[0],S_four[2],S_four[1],S_four[3]],
              [S_four[1],S_four[0],S_four[2],S_four[3]],
              [S_four[1],S_four[2],S_four[0],S_four[3]],
              [S_four[2],S_four[0],S_four[1],S_four[3]],
              [S_four[2],S_four[1],S_four[0],S_four[3]]
            ]
        normal_directions = ['k1', 'k2', 'eta1', 'eta2']
        flag = 1
        for trial_S_four in trial_S_fours:
            trial_S = HexagonalPlane(lattice, four=trial_S_four)
            trial_m = trial_S.get_direction_normal_to_plane()
            dots = [ lattice.dot(trial_m.three,
                                 indices[direction].three,
                                 frac_coords=True)[0]
                        for direction in normal_directions ]
            if np.allclose(np.array(dots), np.zeros(4)):
                flag = 0
                break
        if flag == 1:
            raise ValueError("could not find shear plane")

        triple_product = np.dot(trial_m.get_cartesian(),
                                np.cross(indices['eta1'].get_cartesian(),
                                         indices['eta2'].get_cartesian()))
        if triple_product < 0:
            trial_S.inverse()
            trial_m.inverse()
        indices['S'] = trial_S
        indices['m'] = trial_m
        return indices

    def _supercell_matrix(lattice, indices):
        tf1 = np.array(get_ratio(indices['m'].three))
        tf2 = np.array(get_ratio(indices['eta1'].three))
        tf3 = np.array(get_ratio(indices['eta2'].three))
        supercell_matrix = np.vstack([tf1, tf2, tf3]).T
        return supercell_matrix

    assert lattice.is_hexagonal(), "lattice is not hexagonal"
    is_supported_twinmode(twinmode)
    indices = _twin_indices_orig(lattice, twinmode, wyckoff)
    indices = _reset_indices(lattice, indices)
    indices = _set_shear_plane(lattice, indices)
    return indices
