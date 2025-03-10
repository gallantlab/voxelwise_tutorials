# Voxelwise Encoding Model (VEM) tutorials

Welcome to the tutorials on the Voxelwise Encoding Model framework from the
[GallantLab](https://gallantlab.org).

If you use these tutorials for your work, consider citing the corresponding paper:

> T. Dupré La Tour, M. Visconti di Oleggio Castello, and J. L. Gallant. The voxelwise modeling framework: a tutorial introduction to fitting encoding models to fMRI data. PsyArXiv, 2024. [doi:10.31234/osf.io/t975e.](https://doi.org/10.31234/osf.io/t975e)

You can find a copy of the paper [here](https://github.com/gallantlab/voxelwise_tutorials/blob/main/paper/voxelwise_tutorials_paper.pdf).

## How to use the tutorials

To explore the VEM tutorials, one can:

1. Read the tutorials on this website (recommended)
2. Run the notebooks in Google Colab (clicking on the following links opens Colab):
  [all notebooks](https://colab.research.google.com/github/gallantlab/voxelwise_tutorials/blob/main/tutorials/notebooks/shortclips/merged_for_colab.ipynb) or [only the notebooks about model fitting](https://colab.research.google.com/github/gallantlab/voxelwise_tutorials/blob/main/tutorials/notebooks/shortclips/merged_for_colab_model_fitting.ipynb)
3. Use the provided [Dockerfiles](https://github.com/gallantlab/voxelwise_tutorials/tree/main/docker) to run the notebooks locally (recommended for Windows users, as some of the packages used do not support Windows)

The code of this project is available on GitHub at [gallantlab/voxelwise_tutorials
](https://github.com/gallantlab/voxelwise_tutorials). 

The GitHub repository also contains a Python package called
`voxelwise_tutorials`, which contains useful functions to download the data
sets, load the files, process the data, and visualize the results. Install
instructions are available [here](voxelwise_package.rst)

## Cite as

Please cite the corresponding publications if you use the code or data in your work:
- `voxelwise_tutorials` {cite}`dupre2023`
- `himalaya` {cite}`dupre2022`
- `pycortex` {cite}`gao2015`
- `pymoten` {cite}`nunez2021software`
- `shortclips` dataset {cite}`huth2022data`
- `vim-2` dataset {cite}`nishimoto2014data`

## References
```{bibliography}
:filter: docname in docnames
```