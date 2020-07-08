# Voxelwise modeling tutorial

This tutorial describes how to perform voxelwise modeling on a visual
imaging experiment [1].

This tutorial is based on publicly available data
[published on CRCNS](TBD).
The data is briefly described in the dataset
[description PDF](TBD).
The experiment is described in more details in the original publication [1].

This tutorial also implements the motion energy voxelwise model described in
[2].

> [1] Huth, A. G., Nishimoto, S., Vu, A. T., & Gallant, J. L. (2012). A
    continuous semantic space describes the representation of thousands of
    object and action categories across the human brain. Neuron, 76(6),
    1210-1224.

> [2] Nishimoto, S., Vu, A. T., Naselaris, T., Benjamini, Y., Yu, B., & Gallant,
    J. L. (2011). Reconstructing visual experiences from brain activity evoked
    by natural movies. Current Biology, 21(19), 1641-1646.

## Requirements

This tutorial requires the following Python packages:

- numpy  (for the data array)
- scipy  (for motion energy extraction)
- h5py  (for loading the data files)
- scikit-learn  (for preprocessing and modeling)
- himalaya  (for modeling)
- pymoten  (for extracting motion energy)
- voxelwise  (this repository)
