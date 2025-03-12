==================================
Voxelwise Encoding Model tutorials
==================================

|Github| |Python| |License| |Build| |Build Tutorials| |Downloads|

Welcome to the Voxelwise Encoding Model tutorials, brought to you by the
`Gallant Lab <https://gallantlab.org>`_.

Paper
=====

If you use these tutorials for your work, consider citing the corresponding paper:

   Dupré la Tour, T., Visconti di Oleggio Castello, M., & Gallant, J. L. (2024). The Voxelwise Encoding Model framework: a tutorial introduction to fitting encoding models to fMRI data. https://doi.org/10.31234/osf.io/t975e

You can find a copy of the paper `here <paper/voxelwise_tutorials_paper.pdf>`_.

Tutorials
=========

This repository contains tutorials describing how to use the Voxelwise Encoding Model 
(VEM) framework. `VEM
<https://gallantlab.github.io/voxelwise_tutorials/pages/voxelwise_modeling.html>`_ is
a framework to perform functional magnetic resonance imaging (fMRI) data
analysis, fitting encoding models at the voxel level.

To explore these tutorials, one can:

- Read the rendered examples in the tutorials
  `website <https://gallantlab.github.io/voxelwise_tutorials/>`_ (recommended).
- Run the merged notebook in
  `Colab <https://colab.research.google.com/github/gallantlab/voxelwise_tutorials/blob/main/tutorials/notebooks/shortclips/vem_tutorials_merged_for_colab.ipynb>`_.
- Run the Jupyter notebooks (`tutorials/notebooks <tutorials/notebooks>`_ directory) locally.

The tutorials are best explored in order, starting with the "shortclips"
tutorial. The "vim2" tutorial is optional and redundant with the "shortclips" one.


Dockerfiles
===========

This repository contains Dockerfiles to run the tutorials locally. Please see the
instructions in the `docker <docker>`_ directory.


Helper Python package
=====================

To run the tutorials, this repository contains a small Python package
called ``voxelwise_tutorials``, with useful functions to download the
data sets, load the files, process the data, and visualize the results.

Installation
------------

To install the ``voxelwise_tutorials`` package, run:

.. code-block:: bash

   pip install voxelwise_tutorials


To also download the tutorial scripts and notebooks, clone the repository via:

.. code-block:: bash

   git clone https://github.com/gallantlab/voxelwise_tutorials.git
   cd voxelwise_tutorials
   pip install .


Developers can also install the package in editable mode via:

.. code-block:: bash

   pip install --editable .


Requirements
------------

The tutorials are not compatible with Windows.
If you are using Windows, we recommend running the tutorials on Google Colab or 
in the provided Docker containers.

`git-annex <https://git-annex.branchable.com/>`_ is required to download the
data sets. Please follow the instructions in the
`git-annex documentation <https://git-annex.branchable.com/install/>`_ to install
it on your system.

The tutorials and the package ``voxelwise_tutorials`` require Python 3.9 or higher.

The package ``voxelwise_tutorials`` has the following Python dependencies:
`numpy <https://github.com/numpy/numpy>`_,
`scipy <https://github.com/scipy/scipy>`_,
`h5py <https://github.com/h5py/h5py>`_,
`scikit-learn <https://github.com/scikit-learn/scikit-learn>`_,
`matplotlib <https://github.com/matplotlib/matplotlib>`_,
`networkx <https://github.com/networkx/networkx>`_,
`nltk <https://github.com/nltk/nltk>`_,
`pycortex <https://github.com/gallantlab/pycortex>`_,
`himalaya <https://github.com/gallantlab/himalaya>`_,
`pymoten <https://github.com/gallantlab/pymoten>`_,
`datalad <https://github.com/datalad/datalad>`_.


.. |Github| image:: https://img.shields.io/badge/github-voxelwise_tutorials-blue
   :target: https://github.com/gallantlab/voxelwise_tutorials

.. |Python| image:: https://img.shields.io/badge/python-3.9%2B-blue
   :target: https://www.python.org/downloads/release/python-390

.. |License| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause

.. |Build| image:: https://github.com/gallantlab/voxelwise_tutorials/actions/workflows/run_tests.yml/badge.svg
   :target: https://github.com/gallantlab/voxelwise_tutorials/actions/workflows/run_tests.yml

.. |Build Tutorials| image:: https://github.com/gallantlab/voxelwise_tutorials/actions/workflows/run_tutorials.yml/badge.svg
   :target: https://github.com/gallantlab/voxelwise_tutorials/actions/workflows/run_tutorials.yml

.. |Downloads| image:: https://pepy.tech/badge/voxelwise_tutorials
   :target: https://pepy.tech/project/voxelwise_tutorials


Cite as
=======

If you use one of our packages in your work (``voxelwise_tutorials`` [1]_,
``himalaya`` [2]_, ``pycortex`` [3]_, or ``pymoten`` [4]_), please cite the
corresponding publications:

.. [1] Dupré la Tour, T., Visconti di Oleggio Castello, M., & Gallant, J. L. (2024).
   The Voxelwise Modeling framework: a tutorial introduction to fitting encoding models to fMRI data.
   https://doi.org/10.31234/osf.io/t975e

.. [2] Dupré la Tour, T., Eickenberg, M., Nunez-Elizalde, A.O., & Gallant, J. L. (2022).
   Feature-space selection with banded ridge regression. NeuroImage.
   https://doi.org/10.1016/j.neuroimage.2022.119728

.. [3] Gao, J. S., Huth, A. G., Lescroart, M. D., & Gallant, J. L. (2015).
   Pycortex: an interactive surface visualizer for fMRI. Frontiers in
   neuroinformatics, 23. https://doi.org/10.3389/fninf.2015.00023

.. [4] Nunez-Elizalde, A.O., Deniz, F., Dupré la Tour, T., Visconti di Oleggio
   Castello, M., and Gallant, J.L. (2021). pymoten: scientific python package
   for computing motion energy features from video. Zenodo.
   https://doi.org/10.5281/zenodo.6349625
