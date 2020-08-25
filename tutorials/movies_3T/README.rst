Movies 3T tutorial
==================

This tutorial describes how to perform voxelwise modeling on a visual
imaging experiment.

**Data set:**
This tutorial is based on publicly available data
`published on CRCNS <TBD>`_ [4]_.
The data is briefly described in the dataset `description PDF <TBD>`_,
and in more details in the original publication [1]_.
If you publish work using this data set, please cite the original
publication [1]_, and the CRCNS data set [4]_.

**Models:**
This tutorial implements different voxelwise models:

- a ridge model with wordnet semantic features as described in [1]_.
- a ridge model with motion-energy features as described in [2]_.
- a banded-ridge model with both feature spaces as described in [3]_.

**Scikit-learn API:** These tutorials use `scikit-learn
<https://github.com/scikit-learn/scikit-learn>`_ to define the preprocessing
steps, the modeling pipeline, and the cross-validation scheme. If you are not
familiar with the scikit-learn API, we recommend the `getting started guide
<https://scikit-learn.org/stable/getting_started.html>`_. We also use a lot of
the scikit-learn terminology, which is explained in great details in the
`glossary of common terms and API elements
<https://scikit-learn.org/stable/glossary.html#glossary>`_.

**Running time:**
Most of these tutorials can be run in a very reasonable time (under 1 minute
for most examples, ~7 minutes for the banded ridge example) with a GPU backend
in `himalaya <https://github.com/gallantlab/himalaya>`_. Using a CPU backend is
usually slower (typically 10 times slower).

**Requirements:**
This tutorial requires the following Python packages:

- voxelwise_tutorials  (this repository) and its dependencies
- cupy or pytorch  (optional, to use a GPU backend in himalaya)

**References:**

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
