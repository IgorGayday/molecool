Getting Started
===============
molecool is currently under development, and can not yet be installed from PyPI
or conda. 

Installation
------------
**These instructions assume you have the python package manager `conda` installed.**
To install molecool, navigate to the `GitHub <https://www.github.com/IgorGayday/molecool>`_ and clone it.

Naviate to the project directory. To do a developmental install, type

``pip install -e .``

Dependencies
^^^^^^^^^^^^
You need to install `numpy` and `matplotlib`

Usage
-----
Once installed, you can use the package. This example draws a benzene molecule from an xyz file.::

    import molecool
    
    benzene_symbols, benzene_coords = molecool.open_xyz('benzene.xyz')
    benzene_bonds = molecool.build_bond_list(benzene_coords)
    molecool.draw_molecule(benzene_coords, benzene_symbols, draw_bonds=benzene_bonds)

