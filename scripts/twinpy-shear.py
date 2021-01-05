#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Get twinpy strucuture
"""

import argparse
from twinpy.api_twinpy import Twinpy


# argparse
def get_argparse():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-pk', '--shear_pk', type=int,
                        help="AiidaShearWorkChain PK.")
    args = parser.parse_args()

    return args


def main(shear_pk,
         ):
    twinpy = Twinpy.initialize_from_aiida_shear(shear_pk)

if __name__ == '__main__':
    args = get_argparse()
    main(shear_pk=args.shear_pk,
         )
