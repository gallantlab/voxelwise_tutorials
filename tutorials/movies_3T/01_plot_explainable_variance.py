"""
================================
Compute the explainable variance
================================

"""

###############################################################################

# path of the data directory
directory = '/data1/tutorials/vim-4/'

# modify to use another subject
subject = "S01"

###############################################################################
import os
import numpy as np

from voxelwise.io import load_hdf5_array

# Load fMRI responses on the test set
file_name = os.path.join(directory, 'responses', f'{subject}_responses.hdf')
Y_test = load_hdf5_array(file_name, "Y_test")

# compute the explainable variance per voxel, based on the test set repeats
mean_var = np.mean(np.var(Y_test, axis=1), axis=0)
var_mean = np.var(np.mean(Y_test, axis=0), axis=0)
explainable_variance = var_mean / mean_var

###############################################################################
# Map to subject flatmap
import matplotlib.pyplot as plt
from voxelwise.viz import plot_flatmap_from_mapper

mapper_file = os.path.join(directory, 'mappers', f'{subject}_mappers.hdf')
plot_flatmap_from_mapper(explainable_variance, mapper_file, vmin=0, vmax=0.7)
plt.show()
