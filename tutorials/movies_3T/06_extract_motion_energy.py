"""
===============================================
Extract motion energy features from the stimuli
===============================================

This script describes how to extract motion-energy features from the stimuli.

.. Note:: This public data set already contains precomputed motion-energy.
    Therefore, you do not need to run this script to fit motion-energy models
    in other part of this tutorial.

*Motion-energy features:* Motion-energy features result from filtering a video
stimulus with spatio-temporal Gabor filters. A pyramid of filters is used to
compute the motion-energy features at multiple spatial and temporal scales.
Motion-energy features were introduced in [1]_.

The motion-energy extraction is performed by the package `pymoten
<https://github.com/gallantlab/pymoten>`_. Check the pymoten `gallery of
examples <https://gallantlab.github.io/pymoten/auto_examples/index.html>`_ for
visualizing motion-energy filters, and for pymoten API usage examples.

Running time
------------
Extracting motion energy is a bit longer than the other examples. It typically
takes a couple hours to run.
"""
# sphinx_gallery_thumbnail_path = "static/moten.png"
###############################################################################

# path of the data directory
import os
from voxelwise_tutorials.io import get_data_home
directory = os.path.join(get_data_home(), "vim-5")
print(directory)

###############################################################################
# Load the stimuli images
# -----------------------
#
# Here the data is not loaded in memory, we only take a peak at the data shape.

import h5py

first_file_name = os.path.join(directory, 'stimuli', 'train_00.hdf')
print(f"Content of {first_file_name}:")
with h5py.File(first_file_name, 'r') as f:
    for key in f.keys():
        print(f[key])

###############################################################################
# Compute the luminance
# ---------------------
#
# The motion energy is typically not computed on RGB (color) images,
# but on the luminance channel of the LAB color space.
# To avoid loading the entire simulus array in memory, we use batches of data.
# These batches can be arbitray, since the luminance is computed independently
# on each image.

import numpy as np
from moten.io import imagearray2luminance

from voxelwise_tutorials.progress_bar import bar
from voxelwise_tutorials.io import load_hdf5_array


def compute_luminance(run_name, size=(96, 96), batch_size=100):

    stimuli_file = os.path.join(directory, 'stimuli', run_name)

    # get the number of images in the stimuli file
    with h5py.File(stimuli_file, 'r') as f:
        n_images = f['stimuli'].shape[0]

    # compute the luminance on each batch
    luminance = np.zeros((n_images, *size))
    for start in bar(range(0, n_images, batch_size),
                     title=f'compute_luminance({run_name})'):
        # load the batch of images
        batch = slice(start, start + batch_size)
        images = load_hdf5_array(stimuli_file, key='stimuli', slice=batch)

        # ``imagearray2luminance`` uses uint8 arrays
        if images.dtype != 'uint8':
            images = np.int_(np.clip(images, 0, 1) * 255).astype(np.uint8)

        # convert RGB images to a single luminance channel
        luminance[batch] = imagearray2luminance(images, size=size)

    return luminance


luminance_train = np.concatenate(
    [compute_luminance(f"train_{ii:02d}.hdf") for ii in range(12)])
luminance_test = compute_luminance("test.hdf")

###############################################################################
# Compute the motion energy
# -------------------------
#
# This is done with a ``MotionEnergyPyramid`` object of the ``pymoten``
# package. The parameters used are the one described in [1]_.
#
# Here we use batches corresponding to run lengths. Indeed, motion energy is
# computed over multiple images, since the filters have a temporal component.
# Therefore, motion-energy is not independent of other images, and we cannot
# arbitrarily split the images.

from scipy.signal import decimate
from moten.pyramids import MotionEnergyPyramid

# fixed experiment settings
N_FRAMES_PER_SEC = 15
N_FRAMES_PER_TR = 30
N_TRS_PER_RUN = 300


def compute_motion_energy(luminance,
                          batch_size=N_TRS_PER_RUN * N_FRAMES_PER_TR,
                          noise=0.1):

    n_frames, height, width = luminance.shape

    # We create a pyramid instance, with the main motion-energy parameters.
    pyramid = MotionEnergyPyramid(stimulus_vhsize=(height, width),
                                  stimulus_fps=N_FRAMES_PER_SEC,
                                  spatial_frequencies=[0, 2, 4, 8, 16, 32])

    # We batch images run by run.
    motion_energy = np.zeros((n_frames, pyramid.nfilters))
    for ii, start in enumerate(range(0, n_frames, batch_size)):
        batch = slice(start, start + batch_size)
        print("run %d" % ii)

        # add some noise to deal with constant black areas
        luminance_batch = luminance[batch].copy()
        luminance_batch += np.random.randn(*luminance_batch.shape) * noise
        luminance_batch = np.clip(luminance_batch, 0, 100)

        motion_energy[batch] = pyramid.project_stimulus(luminance_batch)

    # decimate to the sampling frequency of fMRI responses
    motion_energy_decimated = decimate(motion_energy, N_FRAMES_PER_TR,
                                       ftype='fir', axis=0)
    return motion_energy_decimated


motion_energy_train = compute_motion_energy(luminance_train)
motion_energy_test = compute_motion_energy(luminance_test)

###############################################################################
# We end this script with saving the features. These features should be
# approximately equal to the "motion-energy" features already precomputed in
# the public data set.

from voxelwise_tutorials.io import save_hdf5_dataset

features_directory = os.path.join(directory, "features")
if not os.path.exists(features_directory):
    os.makedirs(features_directory)

save_hdf5_dataset(
    os.path.join(features_directory, "motion_energy_recomputed.hdf"),
    dataset=dict(X_train=motion_energy_train, X_test=motion_energy_test,
                 run_onsets=np.arange(0, 3600, 300)))

###############################################################################
# References
# ----------
#
# .. [1] Nishimoto, S., Vu, A. T., Naselaris, T., Benjamini, Y., Yu,
#     B., & Gallant, J. L. (2011). Reconstructing visual experiences from brain
#     activity evoked by natural movies. Current Biology, 21(19), 1641-1646.
