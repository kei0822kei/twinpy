#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module deals with disconnection.
"""

from copy import deepcopy
import math
import numpy as np
from twinpy.structure.twinboundary import TwinBoundaryStructure


class Disconnection():
    """
    Disconnection generation class.
    """

    def __init__(
           self,
           twinboundary:TwinBoundaryStructure,
           ):
        """
        Init.
        """
        if twinboundary.twinmode != '10-12':
            raise RuntimeError("Only support 10-12 twinmode.")

        _check_support_twinmode(twinboundary)

