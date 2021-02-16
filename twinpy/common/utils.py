#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module deals wtih various convenient tools
used by other scripts in twinpy.
"""

import math
from decimal import Decimal, ROUND_HALF_UP
import numpy as np


def print_header(string:str):
    """
    Print heading with specified string.

    Args:
        string: Head string.

    Raises:
        TypeError: input is not str object
    """
    if not isinstance(string, str):
        raise TypeError("Input must be str object not %s." % type(string))
    print("# " + '-' * len(string))
    print("# %s" % string)
    print("# " + '-' * len(string))


def float2frac(var:float, denominator:int=10, atol=1.e-3, rtol=0.) -> str:
    """
    Transform float to fractional (str object).

    Args:
        var: Input value.
        denominator: If 'denominator=10', check from 1/2 to 12/13.
        atol: Used when checking two values are the same or not.
              See documentation for np.allclose.
        rtol: Used when checking two values are the same or not.
              See documentation for np.allclose.

    Returns:
        str: Fractional value.

    Raises:
        RuntimeError: Could not find fractional representation of input 'var'.

    Examples:
        >>> float2frac(-1.333333, 3)
          '-4/3'
    """
    def _decimal_part2frac(deci_part, denominator, atol, rtol):
        flag = 1
        for i in range(2, denominator+1):

            for j in range(1, i):
                if np.allclose(deci_part, j/i, rtol=rtol, atol=atol):
                    flag = 0
                    pair = (j, i)
                    break
                continue

            if flag == 0:
                break
            continue

        if flag == 1:
            raise RuntimeError(
                    "Could not find fractional representation of %s."
                    % deci_part)
        return pair

    if var < 0:
        sign = '-'
    else:
        sign = ''
    int_part = int(abs(var))
    deci_part = abs(var) - int_part

    if np.allclose(deci_part, 0.):

        if int_part == 0:
            return str(int_part)

        return sign+str(int_part)

    pair = _decimal_part2frac(deci_part, denominator, atol, rtol)
    return sign+str(int_part*pair[1]+pair[0])+'/'+str(pair[1])


def get_ratio(nums:list,
              threshold:float=1e-3,
              maxmultiply:int=100) -> list:
    """
    Get ration of input nums.

    Args:
        nums: Numbers.
        threshold: If abs(a - b) < threshold, a and b are considered the same.
        maxmultiply: Max multiply for looking for ratio.

    Returns:
        list: Ratio.

    Raises:
        RuntimeError: Could not find multiply number
                      for making input number integer.

    Examples:
        >>> get_ratio([ 1.33333, 5, 7.5])
          [8, 30, 45]
        >>> get_ratio([ -1.33333, -5, -7.5])
          [-8, -30, -45]

    Note:
        As you can find in 'Examples', 'get_ratio' keeps sign.
    """
    def _get_multinum_for_int(num, threshold, maxmultiply):
        for i in range(1,maxmultiply+1):
            trial_num = num * i
            trial_num_round_off = round_off(trial_num)
            if abs(trial_num - trial_num_round_off) < threshold:
                multinum = i
                break

            if i == 100:
                raise RuntimeError("could not find multiply number "
                                   "for make input number integer")
        return multinum

    inputs = np.array(nums)
    abs_max_ix = np.argmax(np.abs(inputs))
    inputs_div = inputs / abs(inputs[abs_max_ix])
    common_multi = 1
    for num in inputs_div:
        multi = _get_multinum_for_int(num,
                                      threshold=threshold,
                                      maxmultiply=maxmultiply)
        common_multi = (common_multi * multi) // math.gcd(common_multi, multi)
    integers = list(map(round_off, inputs_div * common_multi))
    return integers


def round_off(x:float):
    """
    Round off for input value 'x'.

    Args:
        x: Some value.

    Returns:
        int: Rouned off value.

    Examples:
        >>> round_off(4.5)
            5
        >>> round_off(-4.5)
            -5
    """
    return int(Decimal(str(x)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))


def reshape_dimension(dim:np.array) -> np.array:
    """
    If dim.shape == (3,), reshape to (3,3) numpy array.

    Raises:
        ValueError: Input dimension is neither (3,) or (3,3) np.array.

    Returns:
        np.array: 3x3 dimention matrix.
    """
    if np.array(dim).shape == (3,3):
        dim_matrix = np.array(dim)
    elif np.array(dim).shape == (3,):
        dim_matrix = np.diag(dim)
    else:
        raise ValueError("Input dimension is neither (3,) or (3,3) np.array.")
    return dim_matrix
