#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test twinpy/structure.py
"""

def test_assert_three(return_three):
    """
    Never write 'import conftest'
    """
    print("test return val is 3")
    assert return_three == 3
