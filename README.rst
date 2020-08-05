.. raw:: html

   <h1>Voxelwise modeling tutorials</h1>


|Github| |Python|

.. raw:: html

   <h2>Tutorials</h2>


This repository contains some tutorials describing how to perform voxelwise
modeling, based for instance on visual imaging experiments.
The best way to explore these tutorials is to go to the
`website <https://gallantlab.github.io/voxelwise_tutorials/>`_.

The ``voxelwise_tutorials`` package
=========================

On top of tutorials, this repository also contains a small Python package
called ``voxelwise_tutorials``, which contains useful fonctions to download the
data sets, load the files, process the data, and visualize the results.

Installation
------------

To install this package, run

.. code-block:: bash

   git clone https://github.com/gallantlab/voxelwise_tutorials.git
   cd voxelwise_tutorials
   pip install .


Developers can also install the package in editable mode via:

.. code-block:: bash

   pip install --editable .


Requirements
------------

The Python package ``voxelwise_tutorials`` requires the following dependencies:

- `numpy <https://github.com/numpy/numpy>`_
- `scipy <https://github.com/scipy/scipy>`_
- `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_
- `matplotlib <https://github.com/matplotlib/matplotlib>`_

Each tutorial requires additional dependencies, as listed in their respective
documentations, such as:

- `h5py <https://github.com/h5py/h5py>`_
- `himalaya <https://github.com/gallantlab/himalaya>`_
- `pymoten <https://github.com/gallantlab/pymoten>`_
- `voxelwise_tutorials <https://github.com/gallantlab/voxelwise_tutorials>`_
   (this repository)


.. |Github| image:: https://img.shields.io/badge/github-tutorials-blue
   :target: https://github.com/gallantlab/voxelwise_tutorials

.. |Python| image:: https://img.shields.io/badge/python-3.7%2B-blue
   :target: https://www.python.org/downloads/release/python-370
