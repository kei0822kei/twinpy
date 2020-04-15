#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
this script provide various convenient tools used by other scripts in twinpy
"""

import numpy as np
import math
from decimal import Decimal, ROUND_HALF_UP

def float2frac(var, accuracy=3, denominator=10):
    """
    transform float to fractional (str object)

    Args:
        var (float): input value
        accuracy (int): threshold of whether two values are the same of not, \
                        In the case of 'accuracy=3', \
                        check the third decimal place.
        denominator (int): if 'denominator=10', \
                           check from 1/2 to 12/13

    Returns:
        dict: description

    Raises:
        ValueError: could not find fractional representation of input 'var'

    Examples:
        >>> float2frac(-1.333333, 3)
          '-4/3'

    Note:
        description
    """
    def _decimal_part2frac(deci_part, accuracy, denominator):
        flag = 1
        for i in range(2, denominator+1):
            for j in (range(1, i)):
                if np.allclose(deci_part, j/i):
                    flag = 0
                    pair = (j, i)
                    break
                else:
                    continue
            if flag == 0:
                break
            else:
                continue
        if flag == 1:
            raise ValueError("could not find fractional representation of %s"
                    % deci_part)
        else:
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
        else:
            return sign+str(int_part)
    else:
        pair = _decimal_part2frac(deci_part, accuracy, denominator)
        return sign+str(int_part*pair[1]+pair[0])+'/'+str(pair[1])

def get_ratio(nums:list,
              threshold:float=1e-3,
              maxmultiply:int=100) -> list:
    """
    get ration of input nums

    Args:
        nums (list): numbers

    Returns:
        list: ratio

    Raises:
        ValueError: could not find multiply number
                    for making input number integer

    Examples:
        >>> get_ratio([ 1.33333, 5, 7.5])
          [8, 30, 45]
        >>> get_ratio([ -1.33333, -5, -7.5])
          [-8, -30, -45]

    Note:
        as you can find in 'Examples', 'get_ratio' keeps sign
    """
    def _get_multinum_for_int(num, threshold, maxmultiply):
        for i in range(1,maxmultiply+1):
            trial_num = num * i
            trial_num_round_off = round_off(trial_num)
            if abs(trial_num - trial_num_round_off) < threshold:
                multinum = i
                break
            else:
                if i == 100:
                    raise ValueError("could not find multiply number "
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
    round off for input value 'x'

    Args:
        x (float): some value

    Returns:
        int: rouned off value

    Examples:
        >>> round_off(4.5)
            5
        >>> round_off(-4.5)
            -5
    """
    return int(Decimal(str(x)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
