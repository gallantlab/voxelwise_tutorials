|
|

Movies 4T tutorial
==================

This tutorial describes how to perform voxelwise modeling on a visual
imaging experiment.

**Data set:**
This tutorial is based on publicly available data published on
`CRCNS <https://crcns.org/data-sets/vc/vim-2/about-vim-2>`_ [6]_.
The data is briefly described in the dataset description
`PDF <https://crcns.org/files/data/vim-2/crcns-vim-2-data-description.pdf>`_,
and in more details in the original publication [5]_.
If you publish work using this data set, please cite the original
publication [5]_, and the CRCNS data set [6]_.

.. Note::
    This tutorial is redundant with the "Movies 3T tutorial". It uses a
    different data set, with brain responses limited to the occipital lobe,
    and with no mappers to plot the data on flatmaps.
    Using the "Movies 3T tutorial" with full brain responses is recommended.


**Requirements:**
This tutorial requires the following Python packages:

- voxelwise_tutorials  (this repository) and its dependencies
- cupy or pytorch  (optional, to use a GPU backend in himalaya)

**References:**

.. [5] Nishimoto, S., Vu, A. T., Naselaris, T., Benjamini, Y., Yu,
    B., & Gallant, J. L. (2011). Reconstructing visual experiences from brain
    activity evoked by natural movies. Current Biology, 21(19), 1641-1646.

.. [6] Nishimoto, S., Vu, A. T., Naselaris, T., Benjamini, Y., Yu,
    B., & Gallant, J. L. (2014): Gallant Lab Natural Movie 4T fMRI Data.
    CRCNS.org. http://dx.doi.org/10.6080/K00Z715X