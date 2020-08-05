"""
molecool
A Python package for analyzing and visualizing xyz files. For MolSSI Workshop Python Package development workshop.
"""

# Add imports here
from .functions import *
from .measure import calculate_distance, calculate_angle
from .molecule import calculate_molecular_mass, calculate_center_of_mass, build_bond_list
from .visualize import draw_molecule, bond_histogram

import molecool.io

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
