"""
================================
Compute the explainable variance
================================

Before fitting voxelwise models to the fMRI responses, we can compute the
explainable variance on the test set repeats.

The explainable variance is the part of the fMRI responses that can be
explained by voxelwise modeling. It is thus the upper bound of voxelwise
modeling performances.

Indeed, we can decompose the signal into a sum of two components, one
component that is repeated if we repeat the same experiment, and one component
that changes for each repeat. Because voxelwise modeling would use the same
features for each repeat, it can only model the component that is common to
all repeats. This shared component can be estimated by taking the mean over
repeats of the same experiment.
"""
# sphinx_gallery_thumbnail_number = 2
###############################################################################

# path of the data directory
directory = '/data1/tutorials/vim-4/'

# modify to use another subject
subject = "S01"

###############################################################################
# Compute the explainable variance
# --------------------------------
import os
import numpy as np

from voxelwise.io import load_hdf5_array

###############################################################################
# First, we load the fMRI responses on the test set, which contains 10 repeats.
file_name = os.path.join(directory, 'responses', f'{subject}_responses.hdf')
Y_test = load_hdf5_array(file_name, "Y_test")

###############################################################################
# Then, we compute the explainable variance per voxel.
# The variance of the signal is estimated by taking the variance over
# time (``var``). The variance of the component shared across repeats
# is estimated by taking the variance of the average response (``var_mean``).
# Then, we can compute the explainable variance by dividing the two quantities.

var = np.var(Y_test.reshape(-1, Y_test.shape[-1]), axis=0)
var_mean = np.var(np.mean(Y_test, axis=0), axis=0)
explainable_variance = var_mean / var

###############################################################################
# Plot the distribution of explainable variance over voxels.
import matplotlib.pyplot as plt

plt.hist(explainable_variance, bins=np.linspace(0, 1, 100), log=True,
         histtype='step')
plt.xlabel("Explainable variance")
plt.ylabel("Number of voxels")
plt.title('Histogram of explainable variance')
plt.grid('on')
plt.show()

###############################################################################
# Map to subject flatmap
# ----------------------
from voxelwise.viz import plot_flatmap_from_mapper

mapper_file = os.path.join(directory, 'mappers', f'{subject}_mappers.hdf')
plot_flatmap_from_mapper(explainable_variance, mapper_file, vmin=0, vmax=0.7)
plt.show()

###############################################################################
# We can see that the explainable variance is mainly located in the visual
# cortex, which was expected since this is a purely visual experiment.
