Helper Python package
=====================

|Github| |Python|

To run the tutorials, the `gallantlab/voxelwise_tutorials
<https://github.com/gallantlab/voxelwise_tutorials>`_ repository contains a
Python package called ``voxelwise_tutorials``, with useful fonctions to
download the data sets, load the files, process the data, and visualize the
results.

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

Each tutorial requires additional packages, as listed in their respective
documentations, such as:

- `voxelwise_tutorials <https://github.com/gallantlab/voxelwise_tutorials>`_
- `himalaya <https://github.com/gallantlab/himalaya>`_
- `pymoten <https://github.com/gallantlab/pymoten>`_
- `pycortex <https://github.com/gallantlab/pycortex>`_


.. |Github| image:: https://img.shields.io/badge/github-voxelwise_tutorials-blue
   :target: https://github.com/gallantlab/voxelwise_tutorials

.. |Python| image:: https://img.shields.io/badge/python-3.7%2B-blue
   :target: https://www.python.org/downloads/release/python-370
