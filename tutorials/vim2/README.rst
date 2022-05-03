|
|

Vim-2 tutorial
==============

.. Note::
    This tutorial is redundant with the "Shortclips tutorial". It uses the
    "vim-2" data set, a data set with brain responses limited to the occipital
    lobe, and with no mappers to plot the data on flatmaps.
    Using the "Shortclips tutorial" with full brain responses is recommended.

This tutorial describes how to perform voxelwise modeling on a visual
imaging experiment.

**Data set:**
This tutorial is based on publicly available data published on
`CRCNS <https://crcns.org/data-sets/vc/vim-2/about-vim-2>`_ :ref:`[3b]<nis2011data>`.
The data is briefly described in the dataset description
`PDF <https://crcns.org/files/data/vim-2/crcns-vim-2-data-description.pdf>`_,
and in more details in the original publication :ref:`[3]<nis2011>`.
If you publish work using this data set, please cite the original
publication :ref:`[3]<nis2011>`, and the CRCNS data set :ref:`[3b]<nis2011data>`.


**Requirements:**
This tutorial requires the following Python packages:

- voxelwise_tutorials  (this repository) and its dependencies
- cupy or pytorch  (optional, to use a GPU backend in himalaya)

**Gallery of scripts:**
Click on each thumbnail below to open the corresponding page:
