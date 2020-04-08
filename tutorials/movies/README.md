# Voxelwise modeling tutorial

This tutorial describes how to perform voxelwise modeling on a visual
imaging experiment [1].

This tutorial is based on publicly available data
[published on CRCNS](https://crcns.org/data-sets/vc/vim-2/about-vim-2).
The data is briefly described in the dataset
[description PDF](https://crcns.org/files/data/vim-2/crcns-vim-2-data-description.pdf). The experiment is described in more details in the original publication [1].

> [1] Nishimoto, S., Vu, A. T., Naselaris, T., Benjamini, Y., Yu, B., & Gallant,
    J. L. (2011). Reconstructing visual experiences from brain activity evoked
    by natural movies. Current Biology, 21(19), 1641-1646.


## Requirements

This tutorial requires the following Python packages:

- numpy  (for the data array)
- scipy  (for motion energy extraction)
- h5py  (for loading the data files)
- scikit-image  (for motion energy extraction)
- scikit-learn  (for preprocessing and modeling)
- glabtools  (TODO: update to the motion energy repo)
