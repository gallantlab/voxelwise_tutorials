"""Unit tests on the mappers.

Requires the shortclips dataset locally.
"""

import os

import numpy as np
import cortex
from cortex.testing_utils import has_installed
import matplotlib.pyplot as plt

from voxelwise_tutorials.io import load_hdf5_array
from voxelwise_tutorials.io import load_hdf5_sparse_array
from voxelwise_tutorials.viz import plot_flatmap_from_mapper
from voxelwise_tutorials.viz import plot_2d_flatmap_from_mapper

from voxelwise_tutorials.io import get_data_home
from voxelwise_tutorials.io import download_datalad

subject = "S01"
directory = get_data_home(dataset="shortclips")
file_name = os.path.join("mappers", f'{subject}_mappers.hdf')
mapper_file = os.path.join(directory, file_name)

# download mapper if not already present
download_datalad(file_name, destination=directory,
                 source="https://gin.g-node.org/gallantlab/shortclips")

# Change to save = True to save the figures locally and check the results
save_fig = False


def test_flatmap_mappers():

    ##################
    # create fake data
    voxel_to_flatmap = load_hdf5_sparse_array(mapper_file, 'voxel_to_flatmap')
    voxels = np.linspace(0, 1, voxel_to_flatmap.shape[1])

    ######################
    # plot with the mapper
    ax = plot_flatmap_from_mapper(voxels=voxels, mapper_file=mapper_file,
                                  ax=None)
    fig = ax.figure
    if save_fig:
        fig.savefig('test.png')
    plt.close(fig)


def test_plot_2d_flatmap_from_mapper():

    # Change to save = True to save the figures locally and check the results
    save_fig = False

    ##################
    # create fake data
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
        fig.savefig('test_2d.png')
    plt.close(fig)


def test_roi_masks_shape():
    all_mappers = load_hdf5_array(mapper_file, key=None)

    n_pixels, n_voxels = all_mappers['voxel_to_flatmap_shape']
    n_vertices, n_voxels_ = all_mappers['voxel_to_fsaverage_shape']
    assert n_voxels_ == n_voxels

    for key, val in all_mappers.items():
        if 'roi_mask_' in key:
            assert val.shape == (n_voxels, )


def test_fsaverage_mappers():

    # Change to save = True to save the figures locally and check the results
    save_fig = False

    ##################
    # create fake data
    voxel_to_fsaverage = load_hdf5_sparse_array(mapper_file,
                                                'voxel_to_fsaverage')
    voxels = np.linspace(0, 1, voxel_to_fsaverage.shape[1])

    ##################
    # download fsaverage subject
    if not hasattr(cortex.db, "fsaverage"):
        cortex.utils.download_subject(subject_id="fsaverage",
                                      pycortex_store=cortex.db.filestore)
        cortex.db.reload_subjects()  # force filestore reload

    #############################
    # plot with fsaverage mappers
    projected = voxel_to_fsaverage @ voxels
    vertex = cortex.Vertex(projected, 'fsaverage', vmin=0, vmax=0.3,
                           cmap='inferno', alpha=0.7, with_curvature=True)
    fig = cortex.quickshow(vertex, with_rois=has_installed("inkscape"))
    if save_fig:
        fig.savefig('test_fsaverage.png')
    plt.close(fig)
