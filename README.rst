============================
Voxelwise modeling tutorials
============================

|Github| |Python| |License|

Welcome to the voxelwise modeling tutorial from the
`Gallantlab <https://gallantlab.org>`_.

Tutorials
=========

This repository contains tutorials describing how to use the voxelwise modeling
framework. Voxelwise modeling is a framework to perform functional magnetic
resonance imaging (fMRI) data analysis, fitting encoding models at the voxel
level.

To explore these tutorials, one can:

- read the rendered examples in the tutorials
  `website <https://gallantlab.github.io/voxelwise_tutorials/>`_ (recommended)
- run the Python scripts located in the `tutorials <tutorials>`_ directory
- run the Jupyter notebooks located in the
  `tutorials/notebooks <tutorials/notebooks>`_ directory

To run the tutorials yourself, first download this repository, and install the
dependencies (see below). Then, run either the Python scripts or the
Jupyter notebooks located in the "tutorials" directory. The tutorials are
best explored in order, starting with the "Movies 3T tutorial".

Helper Python package
=====================

To run the tutorials, this repository contains a small Python package
called ``voxelwise_tutorials``, with useful fonctions to download the
data sets, load the files, process the data, and visualize the results.

Installation
------------

To install the ``voxelwise_tutorials`` package, run

.. code-block:: bash

   git clone https://github.com/gallantlab/voxelwise_tutorials.git
   cd voxelwise_tutorials
   pip install .


Developers can also install the package in editable mode via:

.. code-block:: bash

   pip install --editable .


Requirements
------------

The package ``voxelwise_tutorials`` has the following dependencies:

- `numpy <https://github.com/numpy/numpy>`_
- `scipy <https://github.com/scipy/scipy>`_
- `h5py <https://github.com/h5py/h5py>`_
- `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_
- `matplotlib <https://github.com/matplotlib/matplotlib>`_
- `networkx <https://github.com/networkx/networkx>`_
- `nltk <https://github.com/nltk/nltk>`_
- `pycortex <https://github.com/gallantlab/pycortex>`_
- `himalaya <https://github.com/gallantlab/himalaya>`_
- `pymoten <https://github.com/gallantlab/pymoten>`_


.. |Github| image:: https://img.shields.io/badge/github-voxelwise_tutorials-blue
   :target: https://github.com/gallantlab/voxelwise_tutorials

.. |Python| image:: https://img.shields.io/badge/python-3.7%2B-blue
   :target: https://www.python.org/downloads/release/python-370

.. |License| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
