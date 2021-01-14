"""
This is for pytest fixtures.
"""

import pytest

@pytest.fixture(autouse=True, scope='function')
def return_three():
    """
    This is sample fixture.
    """
    print("This is sample funciton.")
    print("return 3")
    return 3
