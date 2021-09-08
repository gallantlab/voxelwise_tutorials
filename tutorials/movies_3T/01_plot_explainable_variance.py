"""
================================
Compute the explainable variance
================================

Before fitting voxelwise models to the fMRI responses, we can estimate the
*explainable variance*. The explainable variance is the part of the fMRI
responses that can be explained by the voxelwise modeling framework.

Indeed, we can decompose the signal into a sum of two components, one component
that is repeated if we repeat the same experiment, and one component that
changes for each repeat. Because voxelwise modeling would use the same features
for each repeat, it can only model the component that is common to all repeats.
This shared component can be estimated by taking the mean over repeats of the
same experiment. The variance of this shared component, that we call the
explainable variance, is the upper bound of the voxelwise modeling
performances. The explainable variance is also sometimes called the *noise
ceiling*.
"""
# sphinx_gallery_thumbnail_number = 1
###############################################################################
# Path of the data directory
import os
from voxelwise_tutorials.io import get_data_home
directory = os.path.join(get_data_home(), "vim-5")
print(directory)

###############################################################################

# modify to use another subject
subject = "S01"

###############################################################################
# Compute the explainable variance
# --------------------------------
import numpy as np
from voxelwise_tutorials.io import load_hdf5_array

###############################################################################
# First, we load the fMRI responses on the test set, which contains ten (10)
# repeats.
file_name = os.path.join(directory, 'responses', f'{subject}_responses.hdf')
Y_test = load_hdf5_array(file_name, key="Y_test")
print("(n_repeats, n_samples_test, n_voxels) =", Y_test.shape)

###############################################################################
# Then, we compute the explainable variance per voxel.
# The variance of the signal is estimated by taking the average variance over
# repeats. The variance of the component shared across repeats is estimated by
# taking the variance of the average response. Then, we compute the
# explainable variance by dividing these two quantities.
# Finally, a correction can be applied to account for small numbers of repeat
# (through the parameter ``bias_correction``).

from voxelwise_tutorials.utils import explainable_variance
ev = explainable_variance(Y_test, bias_correction=False)
print("(n_voxels,) =", ev.shape)

###############################################################################
# To better understand the explainable variance, we can plot the time-courses
# of a voxel with large explainable variance...

import matplotlib.pyplot as plt

voxel_1 = np.argmax(ev)
time = np.arange(Y_test.shape[1]) * 2  # one time point every 2 seconds
plt.figure(figsize=(10, 3))
plt.plot(time, Y_test[:, :, voxel_1].T, color='C0', alpha=0.5)
plt.plot(time, Y_test[:, :, voxel_1].mean(0), color='C1', label='average')
plt.xlabel("Time (sec)")
plt.title("Voxel with large explainable variance (%.2f)" % ev[voxel_1])
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.show()

###############################################################################
# ... and of a voxel with low explainable variance.
voxel_2 = np.argmin(ev)
plt.figure(figsize=(10, 3))
plt.plot(time, Y_test[:, :, voxel_2].T, color='C0', alpha=0.5)
plt.plot(time, Y_test[:, :, voxel_2].mean(0), color='C1', label='average')
plt.xlabel("Time (sec)")
plt.title("Voxel with low explainable variance (%.2f)" % ev[voxel_2])
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.show()

###############################################################################
# We can also plot the distribution of explainable variance over voxels.

plt.hist(ev, bins=np.linspace(0, 1, 100), log=True, histtype='step')
plt.xlabel("Explainable variance")
plt.ylabel("Number of voxels")
plt.title('Histogram of explainable variance')
plt.grid('on')
plt.show()

###############################################################################
# We see that most voxels have a rather low explainable variance, around 0.1
# (when not using the bias correction). This is expected, since most voxels are
# not directly driven by a visual stimulus, and their activity change over
# repeats. We also see that some voxels reach an explainable variance of 0.7,
# which is quite high. It means that these voxels consistently record the same
# activity across a repeated stimulus, and thus are good targets for encoding
# models. Of course, this set of explainable voxels changes from task to
# task, depending on what you are trying to model.

###############################################################################
# Map to subject flatmap
# ----------------------
#
# To better understand the distribution of explainable variance, we map the
# values to the subject brain. This can be done with `pycortex
# <https://gallantlab.github.io/pycortex/>`_, which can create interactive 3D
# viewers to be displayed in any modern browser. ``pycortex`` can also display
# flattened maps of the cortical surface, to visualize the entire cortical
# surface at once.
#
# Here, we do not share the anatomical information of the subjects for privacy
# concerns. Instead, we provide two mappers:
#
# - to map the voxels to a (subject-specific) flatmap
# - to map the voxels to the Freesurfer average cortical surface ("fsaverage")
#
# The first mapper is 2D matrix of shape (n_pixels, n_voxels), that map each
# voxel to a set of pixel in a flatmap. The matrix is efficient stored using a
# ``scipy`` sparse CSR matrix format. The function ``plot_flatmap_from_mapper``
# provides an example of how to use the mapper and visualize the flatmap.

from voxelwise_tutorials.viz import plot_flatmap_from_mapper

mapper_file = os.path.join(directory, 'mappers', f'{subject}_mappers.hdf')
plot_flatmap_from_mapper(ev, mapper_file, vmin=0, vmax=0.7)
plt.show()

###############################################################################
# This figure is a flatten map of the cortical surface. A number of regions of
# interest (ROIs) have been labeled to ease the interpretation. If you have
# never seen such a flatmap, we recommend taking a look at a `pycortex brain
# viewer <https://www.gallantlab.org/brainviewer/Deniz2019>`_, which displays
# the brain in 3D. In this viewer, press "I" to inflate the brain, "F" to
# flatten the surface, and "R" to reset the view (or use the ``surface/unfold``
# cursor on the right menu). Press "H" for a list of all keyboard shortcuts.
# This viewer should help you understand the correspondance between the flatten
# and the folded cortical surface of the brain.

###############################################################################
# On this flatmap, we can see that the explainable variance is mainly located
# in the visual cortex, in early visual regions like V1, V2, V3, or in
# higher-level regions like EBA, FFA or IPS. This was expected since this is a
# purely visual experiment.

###############################################################################
# Map to "fsaverage"
# ------------------
#
# The second mapper we provide maps the voxel data to a Freesurfer
# average surface ("fsaverage"), that can be used in ``pycortex``.
# First, let's download the "fsaverage" surface.

import cortex

surface = "fsaverage"

if not hasattr(cortex.db, surface):
    cortex.utils.download_subject(subject_id=surface)

###############################################################################
# If you are running the notebook on Colab, you might need to update the
# pycortex filestore as following:

try:
    import google.colab  # noqa
    in_colab = True
except ImportError:
    in_colab = False
print(in_colab)

if in_colab:
    filestore = cortex.options.config['basic']['filestore']
    cortex.database.db = cortex.database.Database(filestore)
    cortex.db = cortex.database.db
    cortex.utils.db = cortex.database.db
    cortex.dataset.braindata.db = cortex.database.db

###############################################################################
# Then, we load the "fsaverage" mapper. The mapper is a matrix of shape
# (n_vertices, n_voxels), which maps each voxel to some vertices in the
# fsaverage surface. It is stored as a sparse CSR matrix. The mapper is applied
# with a dot product ``@`` (equivalent to ``np.dot``).
from voxelwise_tutorials.io import load_hdf5_sparse_array
voxel_to_fsaverage = load_hdf5_sparse_array(mapper_file,
                                            key='voxel_to_fsaverage')
ev_projected = voxel_to_fsaverage @ ev
print("(n_vertices,) =", ev_projected.shape)

###############################################################################
# We can then create a ``Vertex`` object in ``pycortex``, containing the
# projected data. This object can be used either in a ``pycortex`` interactive
# 3D viewer, or in a ``matplotlib`` figure showing only the flatmap.

vertex = cortex.Vertex(ev_projected, surface, vmin=0, vmax=0.7, cmap='inferno')

###############################################################################
# To start an interactive 3D viewer in the browser, use the following function:
if False:
    cortex.webshow(vertex, open_browser=True)

###############################################################################
# Alternatively, to plot a flatmap in a ``matplotlib`` figure, use the
# `quickshow` function.
#
# (This function requires Inkscape to be installed. The rest of the tutorial
# does not use this function, so feel free to ignore.)

from cortex.testing_utils import has_installed

if has_installed("inkscape"):
    fig = cortex.quickshow(vertex, colorbar_location='right')
    plt.show()
