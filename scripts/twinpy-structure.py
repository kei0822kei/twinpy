#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
get structure from twinpy
"""

import argparse
from twinpy.structure import hexagonal
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
    parser.add_argument('-c', '--posfile',
        help="POSCAR file")
    parser.add_argument('--get_poscar', action='store_true',
        help="get poscar")
    parser.add_argument('--show_BZ', action='store_true',
        help="show brillouin zone")
    args = parser.parse_args()
    return args

def main(posfile, twinmode, ratio,
         get_poscar=False, show_BZ=False):
    poscar = Poscar.from_file(posfile)
    pmgstructure = poscar.structure
    structure = hexagonal.get_hexagonal_structure_from_pymatgen(pmgstructure)
    structure.set_parent(twinmode)
    structure.set_shear_ratio(ratio)
    structure.run(is_primitive=True)
    if get_poscar:
        filename = 't'+twinmode+'_r'+str(ratio)+'.poscar'
        structure.get_poscar(filename=filename)
    if show_BZ:
        output_lattice = Lattice(structure.output_structure[0])
        reciprocal_lattice = output_lattice.reciprocal_lattice
        show_brillouin_zone(reciprocal_lattice)

if __name__ == '__main__':
    args = get_argparse()
    main(posfile=args.posfile,
         twinmode=args.twinmode,
         ratio=args.ratio,
         get_poscar=args.get_poscar,
         show_BZ=args.show_BZ)
