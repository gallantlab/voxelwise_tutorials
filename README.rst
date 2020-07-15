.. raw:: html

   <h1>Voxelwise modeling tutorial</h1>


|Github| |Python| 


This repository contains some tutorials describing how to perform voxelwise
modeling, based for instance on visual imaging experiments [1]_ [2]_.

The best way to explore these tutorials is to go to the
`website <https://gallantlab.github.io/tutorials/>`_.

Installation
------------

This repository also contains a small Python package called `voxelwise`, which
contains useful fonctions to download the data, load the files, process the
data, and visualize the results.

To install this package, run

.. code-block:: bash

   git clone https://github.com/gallantlab/tutorials.git
   cd tutorials
   pip install .


Developers can also install the package in editable mode via:

.. code-block:: bash

   pip install --editable .


Requirements
------------

The Python package contained in this repository, `voxelwise`, requires the
following Python packages:

- numpy
- scipy
- scikit-learn
- matplotlib

Each tutorial requires additional packages, as listed in their respective
documentations, such as:

- h5py (https://github.com/h5py/h5py)
- himalaya (https://github.com/gallantlab/himalaya)
- pymoten (https://github.com/gallantlab/pymoten)
- voxelwise (this repository)


.. |Github| image:: https://img.shields.io/badge/github-tutorials-blue
   :target: https://github.com/gallantlab/tutorials

.. |Python| image:: https://img.shields.io/badge/python-3.7%2B-blue
   :target: https://www.python.org/downloads/release/python-370
