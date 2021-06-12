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
        _check_support_twinmode(twinboundary)

    def _check_support_twinmode(self, twinmode):
        """
        Check support twinmode.
        """
