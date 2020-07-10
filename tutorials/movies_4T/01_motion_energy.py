"""
This script describes how to extract motion-energy features from the stimuli.

Motion-energy features result from filtering a video stimulus with
spatio-temporal Gabor filters. A pyramid of filters is used to compute the
motion-energy features at multiple spatial and temporal scales.

The motion-energy extraction is performed by the package "pymoten", available
at https://github.com/gallantlab/pymoten.

"""

###############################################################################
# We downloaded the files in the previous script, and here we update the path
# variable to link to the directory containing the data.

directory = '/data1/tutorials/vim-2/'

###############################################################################
# Then, we preload the stimuli.
#
# Here the data is not loaded in memory, we only take a peak at the data shape.

import h5py
import os.path as op

with h5py.File(op.join(directory, 'Stimuli.mat'), 'r') as f:
    print(f.keys())  # Show all variables

    for key in f.keys():
        print(f[key])

###############################################################################
# Then, we compute the luminance of the stimulus images.
#
# Indeed, the motion energy is typically not computed on RGB (color) images,
# but on the luminance channel of the LAB color space.
# To avoid loading the entire simulus array in memory, we use batches of data.
# These batches can be arbitray, since the luminance is computed independently
# on each image.

import numpy as np
from moten.io import imagearray2luminance

from voxelwise.progress_bar import bar


def compute_luminance(train_or_test, batch_size=1024):

    with h5py.File(op.join(directory, 'Stimuli.mat'), 'r') as f:

        if train_or_test == 'train':
            data = f['st']
        elif train_or_test == 'test':
            data = f['sv']
        else:
            raise ValueError('Unknown parameter train_or_test=%r.' %
                             train_or_test)

        title = "compute_luminance(%s)" % train_or_test
        luminance = np.zeros((data.shape[0], data.shape[2], data.shape[3]))
        for start in bar(range(0, data.shape[0], batch_size), title):
            batch = slice(start, start + batch_size)

            # transpose to corresponds to rgb2lab inputs
            rgb_batch = np.transpose(data[batch], [0, 2, 3, 1])

            # make sure we use uint8
            if rgb_batch.dtype != 'uint8':
                rgb_batch = np.int_(np.clip(rgb_batch, 0, 1) * 255).astype(
                    np.uint8)

            # convert RGB images to a single luminance channel
            luminance[batch] = imagearray2luminance(rgb_batch)

    return luminance


luminance_train = compute_luminance("train")
luminance_test = compute_luminance("test")

###############################################################################
# Finally, we compute the motion energy features.
#
# This is done with a `MotionEnergyPyramid` object of the `pymoten` package.
# The parameters used are the one described in [Nishimoto et al. 2011].
#
# Here we use batches corresponding to run lengths. Indeed, motion energy is
# computed over multiple images, since the filters have a temporal component.
# Therefore, motion-energy is not independent of other images, and we cannot
# arbitrarily split the images.

from scipy.signal import decimate
from moten.pyramids import MotionEnergyPyramid

# fixed experiment settings
N_FRAMES_PER_SEC = 15
N_FRAMES_PER_TR = 15
N_TRS_PER_RUN = 600


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
        luminance_batch[luminance_batch < 0] = 0
        luminance_batch[luminance_batch > 100] = 100

        motion_energy[batch] = pyramid.project_stimulus(luminance_batch)

    # decimate to the sampling frequency of fMRI responses
    motion_energy_decimated = decimate(motion_energy, N_FRAMES_PER_TR,
                                       ftype='fir', axis=0)
    return motion_energy_decimated


motion_energy_train = compute_motion_energy(luminance_train)
motion_energy_test = compute_motion_energy(luminance_test)

###############################################################################
# We end with saving the features.

import os

features_directory = op.join(directory, "features")
if not op.exists(features_directory):
    os.makedirs(features_directory)
np.save(op.join(features_directory, "motion_energy_train.npy"),
        motion_energy_train)
np.save(op.join(features_directory, "motion_energy_test.npy"),
        motion_energy_test)
