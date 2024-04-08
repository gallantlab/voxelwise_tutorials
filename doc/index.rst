Voxelwise modeling tutorials
============================


Welcome to the voxelwise modeling tutorials from the
`GallantLab <https://gallantlab.org>`_.

If you use these tutorials for your work, consider citing the corresponding paper:

Dupré la Tour, T., Visconti di Oleggio Castello, M., & Gallant, J. L. (2024). The Voxelwise Modeling framework: a tutorial introduction to fitting encoding models to fMRI data. https://doi.org/10.31234/osf.io/t975e

You can find a copy of the paper `here <https://github.com/gallantlab/voxelwise_tutorials/blob/main/paper/voxelwise_tutorials_paper.pdf>`_.

Getting started
---------------

This website contains tutorials describing how to use the
`voxelwise modeling framework <voxelwise_modeling.html>`_.

To explore these tutorials, one can:

- read the rendered examples in the tutorials
  `gallery of examples <_auto_examples/index.html>`_ (recommended)
- run the Python scripts (`tutorials <https://github.com/gallantlab/voxelwise_tutorials/tree/main/tutorials>`_ directory)
- run the Jupyter notebooks (`tutorials/notebooks
  <https://github.com/gallantlab/voxelwise_tutorials/tree/main/tutorials/notebooks>`_
  directory)
- run the notebooks in Google Colab: 
  `all notebooks <https://colab.research.google.com/github/gallantlab/voxelwise_tutorials/blob/main/tutorials/notebooks/shortclips/merged_for_colab.ipynb>`_ or
  `only the notebooks about model fitting <https://colab.research.google.com/github/gallantlab/voxelwise_tutorials/blob/main/tutorials/notebooks/shortclips/merged_for_colab_model_fitting.ipynb>`_

The tutorials are best explored in order, starting with the `Shortclips
tutorial <_auto_examples/index.html>`_.

The project is available on GitHub at `gallantlab/voxelwise_tutorials
<https://github.com/gallantlab/voxelwise_tutorials>`_. On top of the tutorials
scripts, the GitHub repository contains a Python package called
``voxelwise_tutorials``, which contains useful functions to download the data
sets, load the files, process the data, and visualize the results. Install
instructions are available `here <voxelwise_package.html>`_.

Navigation
----------
.. toctree::
   :includehidden:
   :maxdepth: 1

   _auto_examples/index

.. toctree::
   :maxdepth: 1

   voxelwise_modeling
   voxelwise_package
   references

Cite as
-------

If you use one of our packages in your work (``voxelwise_tutorials``
:ref:`[p1]<dup2023>`, ``himalaya`` :ref:`[p2]<dup2022>`, ``pycortex``
:ref:`[p3]<gao2015>`, or ``pymoten`` :ref:`[p4]<nun2021>`), please cite the
corresponding publications.

If you use one of our public datasets in your work (vim-2
:ref:`[3b]<nis2011data>`, shortclips :ref:`[4b]<hut2012data>`), please cite the
corresponding publications.
