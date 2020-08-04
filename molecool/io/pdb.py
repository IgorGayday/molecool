"""
I/O functions for pdb files.
"""

import numpy as np


def open_pdb(f_loc):
    # This function reads in a pdb file and returns the atom names and coordinates.
    with open(f_loc) as f:
        data = f.readlines()

    coordinates = []
    sym = []
    for line in data:
        if 'ATOM' in line[0:6] or 'HETATM' in line[0:6]:
            sym.append(line[76:79].strip())
            c2 = [float(x) for x in line[30:55].split()]
            coordinates.append(c2)

    coords = np.array(c)

    return sym, coords


