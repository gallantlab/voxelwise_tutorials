import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .io import load_hdf5_array
from .io import load_hdf5_sparse_array


def plot_hist2d(scores_1, scores_2, bins=100, cmin=1, vmin=None, vmax=None,
                ax=None, norm=LogNorm(), colorbar=True, **kwargs):
    """Plot a 2D histogram to compare voxelwise models performances.

    This is mostly based on matplotlib.pyplot.hist2d, but with more relevant
    default parameters. This plot is typically used to compare the voxelwise
    performances of two voxelwise models.

    Parameters
    ----------
    scores_1 : array of shape (n_voxels, )
        Scores of the first voxelwise model. (horizontal axes)
    scores_2 : array of shape (n_voxels, )
        Scores of the second voxelwise model. (vertical axes)
    bins : int or arrays
        Number of bins of the histogram.
    cmin : int
        Minimum number of elements to plot the bin.
    vmin : float
        Minimum value of the histogram, (in both axes).
    vmax : float
        Maximum value of the histogram, (in both axes).
    ax : Axes or None
        Matplotlib Axes where the histogram will be plotted. If None, the
        current figure is used.
    norm : Matplotlib.colors.Norm or None
        Transform used on the histogram values. Default to log scale.
    colorbar : bool
        If True, plot the colorbar next to the figure.
    kwargs : **kwargs
        Other keyword arguments given to ax.hist2d.

    Returns
    -------
    ax : Axes
        Motplotlib Axes where the histogram was plotted.
    """
    if vmin is None:
        vmin = min(scores_1.min(), scores_2.min())
    if vmax is None:
        vmax = max(scores_1.max(), scores_2.max())
    if isinstance(bins, int):
        bins = np.linspace(vmin, vmax, bins)
    if ax is None:
        ax = plt.gca()

    h = ax.hist2d(scores_1, scores_2, bins=bins, cmin=cmin, norm=norm,
                  **kwargs)
    if colorbar:
        ax.figure.colorbar(h[3], ax=ax)

    ax.plot([vmin, vmax], [vmin, vmax], color='k', linewidth=0.5)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.grid()
    return ax


def plot_flatmap_from_mapper(voxels, mapper_file, ax=None, alpha=0.7,
                             cmap='inferno', vmin=None, vmax=None,
                             with_curvature=True, with_rois=True,
                             with_colorbar=True,
                             colorbar_location=(.4, .9, .2, .05)):
    """Plot a flatmap from a mapper file.

    Note that this function does not have the full capability of pycortex,
    (like cortex.quickshow) since it is based on flatmap mappers and not on the
    original brain surface of the subject.

    Parameters
    ----------
    voxels : array of shape (n_voxels, )
        Data to be plotted.
    mapper_file : str
        File name of the mapper.
    ax : matplotlib Axes or None.
        Axes where the figure will be plotted.
        If None, a new figure is created.
    alpha : float in [0, 1], or array of shape (n_voxels, )
        Transparency of the flatmap.
    cmap : str
        Name of the matplotlib colormap.
    vmin : float or None
        Minimum value of the colormap. If None, use the 1st percentile of the
        `voxels` array.
    vmax : float or None
        Minimum value of the colormap. If None, use the 99th percentile of the
        `voxels` array.
    with_curvature : bool
        If True, show the curvature below the data layer.
    with_rois : bool
        If True, show the ROIs labels above the data layer.
    colorbar_location : [left, bottom, width, height]
        Location of the colorbar. All quantities are in fractions of figure
        width and height.

    Returns
    -------
    ax : matplotlib Axes
        Axes where the figure has been plotted.
    """
    # create a figure
    if ax is None:
        flatmap_mask = load_hdf5_array(mapper_file, key='flatmap_mask')
        figsize = np.array(flatmap_mask.shape) / 100.
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes((0, 0, 1, 1))

    # process plotting parameters
    if vmin is None:
        vmin = np.percentile(voxels, 1)
    if vmax is None:
        vmax = np.percentile(voxels, 99)
    if isinstance(alpha, np.ndarray):
        alpha = map_voxels_to_flatmap(alpha, mapper_file)

    # plot the data
    image = map_voxels_to_flatmap(voxels, mapper_file)
    cimg = ax.imshow(image, aspect='equal', zorder=1, alpha=alpha, cmap=cmap,
                     vmin=vmin, vmax=vmax)

    if with_colorbar:
        cbar = ax.inset_axes(colorbar_location)
        ax.figure.colorbar(cimg, cax=cbar, orientation='horizontal')

    # plot additional layers if present
    with h5py.File(mapper_file, mode='r') as hf:
        if with_curvature and "flatmap_curvature" in hf.keys():
            curvature = load_hdf5_array(mapper_file, key='flatmap_curvature')
            background = np.swapaxes(curvature, 0, 1)[::-1]
        else:
            background = map_voxels_to_flatmap(np.ones_like(voxels),
                                               mapper_file)
        ax.imshow(background, aspect='equal', cmap='gray', vmin=0, vmax=1,
                  zorder=0)

        if with_rois and "flatmap_rois" in hf.keys():
            rois = load_hdf5_array(mapper_file, key='flatmap_rois')
            ax.imshow(
                np.swapaxes(rois, 0, 1)[::-1], aspect='equal',
                interpolation='bicubic', zorder=2)

    return ax


def map_voxels_to_flatmap(voxels, mapper_file):
    """Generate flatmap image from voxel array using a mapper.

    This function maps an array of voxels into a flattened representation
    of an individual subject's brain.

    Parameters
    ----------
    voxels: array of shape (n_voxels, )
        Voxel values to be mapped.
    mapper_file: string
        File containing mapping arrays for a particular subject.

    Returns
    -------
    image : array of shape (width, height)
        Flatmap image.
    """
    voxel_to_flatmap = load_hdf5_sparse_array(mapper_file, 'voxel_to_flatmap')
    flatmap_mask = load_hdf5_array(mapper_file, 'flatmap_mask')

    badmask = np.array(voxel_to_flatmap.sum(1) > 0).ravel()
    img = (np.nan * np.ones(flatmap_mask.shape)).astype(voxels.dtype)
    mimg = (np.nan * np.ones(badmask.shape)).astype(voxels.dtype)
    mimg[badmask] = (voxel_to_flatmap * voxels.ravel())[badmask].astype(
        mimg.dtype)
    img[flatmap_mask] = mimg
    return img.T[::-1]
