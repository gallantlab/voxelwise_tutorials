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

from voxelwise_tutorials.io import load_hdf5_array

###############################################################################
# First, we load the fMRI responses on the test set, which contains 10 repeats.
file_name = os.path.join(directory, 'responses', f'{subject}_responses.hdf')
Y_test = load_hdf5_array(file_name, key="Y_test")

###############################################################################
# Then, we compute the explainable variance per voxel.
# The variance of the signal is estimated by taking the average variance over
# repeats. The variance of the component shared across repeats is estimated by
# taking the variance of the average response. Then, we compute the
# explainable variance by dividing these two quantities.
# Finally, a correction can be applied to account for small numbers of repeat
# (parameter ``bias_correction``).

from voxelwise_tutorials.utils import explainable_variance
ev = explainable_variance(Y_test, bias_correction=False)

###############################################################################
# We can plot the distribution of explainable variance over voxels.

import matplotlib.pyplot as plt

plt.hist(ev, bins=np.linspace(0, 1, 100), log=True, histtype='step')
plt.xlabel("Explainable variance")
plt.ylabel("Number of voxels")
plt.title('Histogram of explainable variance')
plt.grid('on')
plt.show()

###############################################################################
# We see that most voxels have a rather low explainable variance, around 0.1
# (when not using the bias correction). This is expected, since most voxels are
# not directly driven by a visual stimulus.
# We also see that some voxels reach an explainable variance of 0.7, which is
# quite high. It means that these voxels consistently record the same activity
# across a repeated stimulus, and thus are good targets for encoding models.

###############################################################################
# Map to subject flatmap
# ----------------------
#
# To better understand the distribution of explainable variance, we map the
# values to the subject brain. This can be done with
# `pycortex <https://gallantlab.github.io/pycortex/>`_, which can create
# interactive 3D viewers displayed in any modern browser.
# ``Pycortex`` can also display flatten maps of the cortical surface, to
# visualize the entire cortical surface at once.
#
# Here, we do not share the anatomical information of the subjects for privacy
# concerns. Instead, we provide two mappers, (i) to map the voxels to a
# subject-specific flatmap, or (ii) to map the voxels to the Freesurfer average
# cortical surface ("fsaverage").
#
# The first mapper is a sparse CSR matrix that map each voxel to a set of pixel
# in a flatmap. To ease its use, we provide here an example function
# ``plot_flatmap_from_mapper``.

from voxelwise_tutorials.viz import plot_flatmap_from_mapper

mapper_file = os.path.join(directory, 'mappers', f'{subject}_mappers.hdf')
plot_flatmap_from_mapper(ev, mapper_file, vmin=0, vmax=0.7)
plt.show()

###############################################################################
# We can see that the explainable variance is mainly located in the visual
# cortex, in early regions like V1, V2, V3, or in higher-level regions like
# EBA, FFA or IPS. This was expected since this is a purely visual experiment.

###############################################################################
# Map to fsaverage
# ----------------
#
# The second mapper we provide maps the voxel data to a Freesurfer
# average surface ("fsaverage"), that can be used in ``pycortex``.
# First, let's download the fsaverage surface if it does not exist

import cortex

surface = "fsaverage_pycortex"  # ("fsaverage" outside the Gallant lab)

if not hasattr(cortex.db, surface):
    cortex.utils.download_subject(subject_id=surface)

###############################################################################
# Then, we load the fsaverage mapper. The mapper is a sparse CSR matrix, which
# map each voxel to some vertices in the fsaverage surface.
# The mapper is applied with a dot product ``@``.
from voxelwise_tutorials.io import load_hdf5_sparse_array
voxel_to_fsaverage = load_hdf5_sparse_array(mapper_file,
                                            key='voxel_to_fsaverage')
ev_projected = voxel_to_fsaverage @ ev

###############################################################################
# We can then create a ``Vertex`` object with the projected data.
# This object can be used either in a ``pycortex`` interactive 3D viewer, or
# in a ``matplotlib`` figure showing directly the flatmap.

vertex = cortex.Vertex(ev_projected, surface, vmin=0, vmax=0.7, cmap='inferno')

###############################################################################
# To start an interactive 3D viewer in the browser, use the following function:
if False:
    cortex.webshow(vertex, open_browser=True)

###############################################################################
# Alternatively, to plot a flatmap in a ``matplotlib`` figure, use the
# following function:

fig = cortex.quickshow(vertex, colorbar_location='right')
plt.show()
