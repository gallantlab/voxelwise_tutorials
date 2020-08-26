"""Unit tests on the mappers.

Requires the movies 3T dataset locally.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from voxelwise_tutorials.io import load_hdf5_sparse_array
from voxelwise_tutorials.viz import plot_flatmap_from_mapper
from voxelwise_tutorials.viz import plot_2d_flatmap_from_mapper

dataset_directory = '/data1/tutorials/vim-4/'
subject_id = "S01"


def test_flatmap_mappers():

    # Change to save = True to save the figures locally and check the results
    save_fig = False

    ##################
    # create fake data
    mapper_file = os.path.join(dataset_directory, "mappers",
                               '{}_mappers.hdf'.format(subject_id))

    voxel_to_flatmap = load_hdf5_sparse_array(mapper_file, 'voxel_to_flatmap')
    voxels = np.linspace(0, 1, voxel_to_flatmap.shape[1])

    ######################
    # plot with the mapper
    ax = plot_flatmap_from_mapper(voxels=voxels, mapper_file=mapper_file,
                                  ax=None)
    fig = ax.figure
    if save_fig:
        fig.savefig(f'{subject_id}.png')
    plt.close(fig)


def test_plot_2d_flatmap_from_mapper():

    # Change to save = True to save the figures locally and check the results
    save_fig = False

    ##################
    # create fake data
    mapper_file = os.path.join(dataset_directory, "mappers",
                               '{}_mappers.hdf'.format(subject_id))

    voxel_to_flatmap = load_hdf5_sparse_array(mapper_file, 'voxel_to_flatmap')
    phase = np.linspace(0, 2 * np.pi, voxel_to_flatmap.shape[1])
    sin = np.sin(phase)
    cos = np.cos(phase)

    ######################
    # plot with the mapper
    ax = plot_2d_flatmap_from_mapper(sin, cos, mapper_file=mapper_file,
                                     vmin=-1, vmax=1, vmin2=-1, vmax2=1)
    fig = ax.figure
    if save_fig:
        fig.savefig(f'{subject_id}.png')
    plt.close(fig)
