#!/usr/bin/env python

"""
Interface for Aiida Node.
"""


def load_aiida_profile():
    """
    Load aiida profile.
    """

    try:
        from aiida import load_profile
        from aiida.common.exceptions import ProfileConfigurationError
        load_profile()
    except ProfileConfigurationError as err:
        err_msg = "Failed to load aiida profile. " \
                + "Please check your aiida configuration."
        print(err_msg)
        raise ImportError from err
    except ImportError:
        err_msg = "Aiida is not installed. Skip loading aiida profile."
        print(err_msg)
        raise
