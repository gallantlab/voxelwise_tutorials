# Shortclips tutorial

This tutorial describes how to use the Voxelwise Encoding Model framework in a visual
imaging experiment.

## Dataset

This tutorial is based on publicly available data [published on
GIN](https://gin.g-node.org/gallantlab/shortclips) {cite}`huth2022data`.
This data set contains BOLD fMRI responses in human subjects viewing a set of
natural short movie clips. The functional data were collected in five subjects,
in three sessions over three separate days for each subject. Details of the
experiment are described in the original publication {cite}`huth2012`. If
you publish work using this data set, please cite the original publications
{cite}`huth2012`, and the GIN data set {cite}`huth2022data`.

## Models
This tutorial shows and compares analyses based on three different voxelwise encoding models:

1. A ridge model with [wordnet semantic features](03_plot_wordnet_model.ipynb) as described in {cite}`huth2012`.
2. A ridge model with [motion-energy features](05_plot_motion_energy_model.ipynb) as described in {cite}`nishimoto2011`.
3. A banded-ridge model with [both feature spaces](06_plot_banded_ridge_model.ipynb) as described in {cite}`nunez2019`.

## Scikit-learn API
These tutorials use [scikit-learn](https://github.com/scikit-learn/scikit-learn) to define the preprocessing steps, the modeling pipeline, and the cross-validation scheme. If you are not
familiar with the scikit-learn API, we recommend the [getting started guide](https://scikit-learn.org/stable/getting_started.html). We also use a lot of
the scikit-learn terminology, which is explained in great details in the
[glossary of common terms and API elements](https://scikit-learn.org/stable/glossary.html#glossary).

## Running time
Most of these tutorials can be run in a reasonable time with a GPU (under 1 minute for most examples, ~7 minutes for the banded ridge example). Running these examples on a CPU is much slower (typically 10 times slower).

## Requirements
This tutorial requires the following Python packages:

- `voxelwise_tutorials` and its dependencies (see [this page](../../voxelwise_package.rst) for installation instructions)
- `cupy` or `pytorch` (optional, required to use a GPU backend in himalaya)

## References
```{bibliography}
:filter: docname in docnames
```