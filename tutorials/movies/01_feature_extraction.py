"""
This script describes how to extract motion energy features from the stimuli.
"""

###############################################################################
# Once we downloaded the files, we update the path variable to link to the
# directory containing the data.

path = '/data1/tutorials/vim2/'

###############################################################################
# Then, we load the stimuli.

import h5py
import os.path as op

# Here the data is not loaded in memory, we only take a peak at the data shape.
with h5py.File(op.join(path, 'Stimuli.mat'), 'r') as f:
    print(f.keys())  # Show all variables

    for key in f.keys():
        print(f[key])

###############################################################################
# Then, we compute the lunminance of the images.
# To avoid loading the entire array in memory, we use (arbitrary) batches of
# data.

import numpy as np
from skimage.color import rgb2lab

from himalaya.progress_bar import bar


def compute_luminance(train_or_test, batch_size=1024):

    with h5py.File(op.join(path, 'Stimuli.mat'), 'r') as f:

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
            rgb_batch = np.transpose(data[batch], [2, 3, 0, 1])

            # convert RGB images to a single luminance channel
            luminance_batch = rgb2lab(rgb_batch)[:, :, :, 0]

            # store in the luminance array
            luminance[batch] = np.transpose(luminance_batch, [2, 0, 1])

    return luminance


luminance_train = compute_luminance("train")
luminance_test = compute_luminance("test")

###############################################################################
# Finally, we compute the motion energy features.
# We use batches corresponding to run lengths.

from scipy.signal import decimate
from glabtools.feature_spaces.moten_gpu import compute_filter_responses

N_FRAMES_PER_SEC = 15
N_FRAMES_PER_TR = 15
N_TRS_PER_RUN = 600


def compute_motion_energy(luminance,
                          batch_size=N_TRS_PER_RUN * N_FRAMES_PER_TR,
                          noise=0.1):

    motion_energy = np.zeros((luminance.shape[0], 6555))
    for ii, start in enumerate(range(0, luminance.shape[0], batch_size)):
        batch = slice(start, start + batch_size)
        print("run %d" % ii)

        # add some noise to deal with constant areas
        luminance_batch = luminance[batch].copy()
        luminance_batch += np.random.randn(*luminance_batch.shape) * noise
        luminance_batch[luminance_batch < 0] = 0
        luminance_batch[luminance_batch > 100] = 100

        motion_energy_batch = compute_filter_responses(
            luminance_batch, stimulus_fps=N_FRAMES_PER_SEC)
        motion_energy[batch] = motion_energy_batch.cpu().numpy()

    # decimate to the sampling frequency of fMRI responses
    motion_energy_decimated = decimate(motion_energy, N_FRAMES_PER_TR,
                                       ftype='fir', axis=0)
    return motion_energy_decimated


motion_energy_train = compute_motion_energy(luminance_train)
motion_energy_test = compute_motion_energy(luminance_test)

###############################################################################
# We end with saving the features.

import os

directory = op.join(path, "features")
if not op.exists(directory):
    os.makedirs(directory)
np.save(op.join(directory, "motion_energy_train.npy"), motion_energy_train)
np.save(op.join(directory, "motion_energy_test.npy"), motion_energy_test)
