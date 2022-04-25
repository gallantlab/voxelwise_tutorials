Shortclips tutorial
===================

This tutorial describes how to perform voxelwise modeling on a visual
imaging experiment.

**Data set:** This tutorial is based on publicly available data `published on
GIN <https://gin.g-node.org/gallantlab/shortclips>`_ :ref:`[4b]<hut2012data>`.
This data set contains BOLD fMRI responses in human subjects viewing a set of
natural short movie clips. The functional data were collected in five subjects,
in three sessions over three separate days for each subject. Details of the
experiment are described in the original publication :ref:`[4]<hut2012>`. If
you publish work using this data set, please cite the original publications
:ref:`[4]<hut2012>`, and the GIN data set :ref:`[4b]<hut2012data>`.

**Models:**
This tutorial implements different voxelwise encoding models:

- a ridge model with wordnet semantic features as described in :ref:`[4]<hut2012>`.
- a ridge model with motion-energy features as described in :ref:`[3]<nis2011>`.
- a banded-ridge model with both feature spaces as described in :ref:`[12]<nun2019>`.

**Scikit-learn API:** These tutorials use `scikit-learn
<https://github.com/scikit-learn/scikit-learn>`_ to define the preprocessing
steps, the modeling pipeline, and the cross-validation scheme. If you are not
familiar with the scikit-learn API, we recommend the `getting started guide
<https://scikit-learn.org/stable/getting_started.html>`_. We also use a lot of
the scikit-learn terminology, which is explained in great details in the
`glossary of common terms and API elements
<https://scikit-learn.org/stable/glossary.html#glossary>`_.

**Running time:** Most of these tutorials can be run in a reasonable time
(under 1 minute for most examples, ~7 minutes for the banded ridge example)
with a GPU backend in `himalaya <https://github.com/gallantlab/himalaya>`_.
Using a CPU backend is slower (typically 10 times slower).

**Requirements:**
This tutorial requires the following Python packages:

- voxelwise_tutorials  (this repository) and its dependencies
- cupy or pytorch  (optional, to use a GPU backend in himalaya)

**Gallery of scripts:**
Click on each thumbnail below to open the corresponding page:
