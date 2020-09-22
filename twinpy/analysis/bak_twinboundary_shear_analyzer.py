#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analize twinboundary shear calculation.
"""
import numpy as np
from twinpy.analysis import ShearAnalyzer, TwinBoundaryAnalyzer


class TwinBoundaryShearAnalyzer(ShearAnalyzer):
    """
    Analize twinboudnary shear result.
    """

    def __init__(
           self,
           twinboundary_analyzers:list
           ):
        """
        Args:
            twinboundary_analyzer (list): list of TwinBoundaryAnalyzer objects.
        """
        self._twinboundary_analyzers = twinboundary_analyzers

    @property
    def twinboundary_analyzers(self):
        """
        List of TwinBoundaryAnalyzer objects.
        """
        return self._twinboundary_analyzers

