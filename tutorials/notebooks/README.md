# Jupyter notebooks

This directory contains a copy of each python script as a jupyter notebook.
These notebooks are generated automatically by `sphinx-gallery`, and do not
render any content. To see rendered versions of the tutorials, go to the
[tutorials website](https://gallantlab.github.io/voxelwise_tutorials)

## Run the notebooks
To run the notebooks yourself, you first need to download this repository and
install the `voxelwise_tutorials` package:

```bash
    git clone https://github.com/gallantlab/voxelwise_tutorials.git
    cd voxelwise_tutorials
    pip install .
```

You also need the following dependencies:

- numpy
- scipy
- h5py
- scikit-learn
- jupyter
- matplotlib
- himalaya
- pycortex  (optional, for using 3D brain viewers)
- pymoten  (optional, for extracting motion energy)
- cupy/pytorch (optional, to use a GPU backend in himalaya)

For more info about how to run jupyter notebooks, see
[the official documentation](https://jupyter-notebook.readthedocs.io/en/stable/).
