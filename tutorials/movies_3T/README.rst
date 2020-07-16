Movies 3T tutorial
==================

This tutorial describes how to perform voxelwise modeling on a visual
imaging experiment.

This tutorial is based on publicly available data `published on CRCNS <TBD>`_
[4]_.
The data is briefly described in the dataset `description PDF <TBD>`_.
The experiment is described in more details in the original publication
[1]_.

This tutorial implements different voxelwise models:

- the wordnet model described in [1]_
- the motion-energy model described in [2]_
- the banded-ridge model described in [3]_


**Requirements**

This tutorial requires the following Python packages:

- numpy  (for the data array)
- scipy  (for motion energy extraction)
- h5py  (for loading the data files)
- scikit-learn  (for preprocessing and modeling)
- himalaya  (for modeling)
- pymoten  (for extracting motion energy)
- voxelwise  (this repository)


**References**

If you publish work using this data set, please cite the original
publication [1]_, and the CRCNS data set [4]_.

.. [1] Huth, A. G., Nishimoto, S., Vu, A. T., & Gallant, J. L. (2012).
    A continuous semantic space describes the representation of thousands of
    object and action categories across the human brain. Neuron, 76(6),
    1210-1224.

.. [2] Nishimoto, S., Vu, A. T., Naselaris, T., Benjamini, Y., Yu,
    B., & Gallant, J. L. (2011). Reconstructing visual experiences from brain
    activity evoked by natural movies. Current Biology, 21(19), 1641-1646.

.. [3] Nunez-Elizalde, A. O., Huth, A. G., & Gallant, J. L. (2019). 
    Voxelwise encoding models with non-spherical multivariate normal priors.
    Neuroimage, 197, 482-492.

.. [4] Huth, A. G., Nishimoto, S., Vu, A. T., & Gallant, J. L. (2020):
    Gallant Lab Natural Movie 3T fMRI Data. CRCNS.org.
    http://dx.doi.org/10.6080/TBD
