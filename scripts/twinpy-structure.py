#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
get structure from twinpy
"""

import argparse
import numpy as np
from twinpy.structure.hexagonal import (get_hexagonal_structure_from_a_c,
                                        get_hexagonal_structure_from_pymatgen)
from twinpy.lattice.lattice import Lattice
from twinpy.lattice.brillouin import show_brillouin_zone
from pymatgen.io.vasp import Poscar
from pymatgen.core.structure import Structure

# argparse
def get_argparse():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-r', '--ratio', type=float, default=0.,
        help="strain ratio")
    parser.add_argument('--twinmode',
        help="twinmode")
    parser.add_argument('--twintype', type=int, default=None,
        help="twintype, when you specify this, twin boundary mode is evoked")
    parser.add_argument('--is_primitive', action='store_true',
        help="primitive, do not specify when you evoke twin boundary mode")
    parser.add_argument('--xshift', type=float, default=0.,
        help="x shift")
    parser.add_argument('--yshift', type=float, default=0.,
        help="y shift")
    parser.add_argument('--dim', type=str, default='1 1 1',
        help="dimension")
    parser.add_argument('-c', '--posfile', default=None,
        help="POSCAR file")
    parser.add_argument('--get_poscar', action='store_true',
        help="get poscar")
    parser.add_argument('--get_lattice', action='store_true',
        help="get lattice not structure")
    parser.add_argument('-o', '--filename', default=None,
        help="POSCAR filename")
    parser.add_argument('--show_BZ', action='store_true',
        help="show brillouin zone")
    args = parser.parse_args()
    return args

def main(
         ratio,
         twinmode,
         twintype,
         is_primitive,
         xshift,
         yshift,
         dim,
         posfile,
         get_poscar,
         get_lattice,
         filename,
         show_BZ,
        ):
    if posfile is None:
        print("POSCAR file did not specify.")
        print("set automatically, a=2.93, c=4.65, symbol='Ti', wyckoff='c'")
        structure = get_hexagonal_structure_from_a_c(
                a=2.93, c=4.65, symbol='Ti', wyckoff='c')
    else:
        poscar = Poscar.from_file(posfile)
        pmgstructure = poscar.structure
        structure = get_hexagonal_structure_from_pymatgen(pmgstructure)
    structure.set_parent(twinmode)
    if twintype is not None:
        structure.set_twintype(twintype)
    structure.set_shear_ratio(ratio)
    structure.set_xshift(xshift)
    structure.set_yshift(yshift)
    structure.set_dimension(dim)
    structure.set_is_primitive(is_primitive)
    structure.run()
    if get_poscar:
        if filename is None:
            filename = ''
            if twintype is None:
                if np.allclose(ratio, 0.):
                    filename += 'parent'
                else:
                    filename += 'shear'
            else:
                filename += 'tb'
            filename += '_%s' % twinmode
            if get_lattice:
                filename += '_lat.poscar'
            else:
                filename += '.poscar'
        structure.get_poscar(filename=filename, get_lattice=get_lattice)
    if show_BZ:
        output_lattice = Lattice(structure.output_structure['lattice'])
        reciprocal_lattice = output_lattice.reciprocal_lattice
        show_brillouin_zone(reciprocal_lattice)

if __name__ == '__main__':
    args = get_argparse()
    dimension = list(map(int, args.dim.split()))
    main(
         ratio=args.ratio,
         twinmode=args.twinmode,
         twintype=args.twintype,
         is_primitive=args.is_primitive,
         xshift=args.xshift,
         yshift=args.yshift,
         dim=dimension,
         posfile=args.posfile,
         get_poscar=args.get_poscar,
         get_lattice=args.get_lattice,
         filename=args.filename,
         show_BZ=args.show_BZ,
        )
