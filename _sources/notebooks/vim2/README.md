# Vim-2 tutorial

:::{note}
This tutorial is redundant with the [Shortclips tutorial](../shortclips/README.md). 
It uses the "vim-2" data set, a data set with brain responses limited to the occipital
lobe, and with no mappers to plot the data on flatmaps.
Using the "Shortclips tutorial" with full brain responses is recommended.
:::

This tutorial describes how to use the Voxelwise Encoding Model framework in a visual
imaging experiment.

## Data set

This tutorial is based on publicly available data published on
[CRCNS](https://crcns.org/data-sets/vc/vim-2/about-vim-2) {cite}`nishimoto2014data`.
The data is briefly described in the dataset description
[PDF](https://crcns.org/files/data/vim-2/crcns-vim-2-data-description.pdf),
and in more details in the original publication {cite}`nishimoto2011`.
If you publish work using this data set, please cite the original
publication {cite}`nishimoto2011`, and the CRCNS data set {cite}`nishimoto2014data`.

## Requirements
This tutorial requires the following Python packages:

- `voxelwise_tutorials` and its dependencies (see [this page](../../voxelwise_package.rst) for installation instructions)
- `cupy` or `pytorch` (optional, required to use a GPU backend in himalaya)

## References
```{bibliography}
:filter: docname in docnames
```