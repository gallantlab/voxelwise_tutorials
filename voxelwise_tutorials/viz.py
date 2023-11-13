import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize

import cortex

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
        Matplotlib Axes where the histogram was plotted.
    """
    vmin = min(scores_1.min(), scores_2.min()) if vmin is None else vmin
    vmax = max(scores_1.max(), scores_2.max()) if vmax is None else vmax
    bins = np.linspace(vmin, vmax, bins) if isinstance(bins, int) else bins
    ax = plt.gca() if ax is None else ax

    h = ax.hist2d(scores_1, scores_2, bins=bins, cmin=cmin, norm=norm,
                  **kwargs)
    if colorbar:
        cbar = ax.figure.colorbar(h[3], ax=ax)
        cbar.ax.set_ylabel('number of voxels')

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
    """Plot a flatmap from a mapper file, with 1D data.
    
    This function is equivalent to the pycortex functions:
    cortex.quickshow(cortex.Volume(voxels, ...), ...)

    Note that this function does not have the full capability of pycortex,
    since it is based on flatmap mappers and not on the original brain
    surface of the subject.

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
        ax.axis('off')

    # process plotting parameters
    vmin = np.percentile(voxels, 1) if vmin is None else vmin
    vmax = np.percentile(voxels, 99) if vmax is None else vmax
    if isinstance(alpha, np.ndarray):
        alpha = map_voxels_to_flatmap(alpha, mapper_file)

    # plot the data
    image = map_voxels_to_flatmap(voxels, mapper_file)
    cimg = ax.imshow(image, aspect='equal', zorder=1, alpha=alpha, cmap=cmap,
                     vmin=vmin, vmax=vmax)

    if with_colorbar:
        try:
            cbar = ax.inset_axes(colorbar_location)
        except AttributeError:  # for matplotlib < 3.0
            cbar = ax.figure.add_axes(colorbar_location)
        ax.figure.colorbar(cimg, cax=cbar, orientation='horizontal')

    # plot additional layers if present
    _plot_addition_layers(ax=ax, n_voxels=voxels.shape[0],
                          mapper_file=mapper_file,
                          with_curvature=with_curvature, with_rois=with_rois)

    return ax


def map_voxels_to_flatmap(voxels, mapper_file):
    """Generate flatmap image from voxel array using a mapper.

    This function maps an array of voxels into a flattened representation
    of an individual subject's brain.

    Parameters
    ----------
    voxels: array of shape (n_voxels, ) or (n_voxels, n_channels)
        Voxel values to be mapped.
    mapper_file: string
        File containing mapping arrays for a particular subject.

    Returns
    -------
    image : array of shape (height, width) or (height, width, n_channels)
        Flatmap image.
    """
    voxel_to_flatmap = load_hdf5_sparse_array(mapper_file, 'voxel_to_flatmap')
    flatmap_mask = load_hdf5_array(mapper_file, 'flatmap_mask')

    ndim = voxels.ndim
    if ndim == 1:
        voxels = voxels[:, None]

    # dimensions
    width, height = flatmap_mask.shape
    n_voxels_0, n_channels = voxels.shape
    n_pixels_used, n_voxels_1 = voxel_to_flatmap.shape
    if n_voxels_0 != n_voxels_1:
        raise ValueError(f"Dimension mismatch, {n_voxels_0} voxels given "
                         f"while the mapper expects {n_voxels_1} voxels.")

    # create image with nans, and array with used pixels
    img = np.full(shape=(width, height, n_channels), fill_value=np.nan)
    mimg = np.full(shape=(n_pixels_used, n_channels), fill_value=np.nan)

    # fill array with used pixels
    badmask = np.array(voxel_to_flatmap.sum(1) > 0).ravel()
    mimg[badmask, :] = (voxel_to_flatmap * voxels)[badmask, :]

    # copy used pixels in the image
    img[flatmap_mask] = mimg

    if ndim == 1:
        img = img[:, :, 0]

    # flip the image to match default orientation in plt.imshow
    return np.swapaxes(img, 0, 1)[::-1]


def _plot_addition_layers(ax, n_voxels, mapper_file, with_curvature,
                          with_rois):
    """Helper function to plot additional layers if present."""
    with h5py.File(mapper_file, mode='r') as hf:
        if with_curvature and "flatmap_curvature" in hf.keys():
            curvature = load_hdf5_array(mapper_file, key='flatmap_curvature')
            background = np.swapaxes(curvature, 0, 1)[::-1]
        else:
            background = map_voxels_to_flatmap(np.ones(n_voxels), mapper_file)
        ax.imshow(background, aspect='equal', cmap='gray', vmin=0, vmax=1,
                  zorder=0)

        if with_rois and "flatmap_rois" in hf.keys():
            rois = load_hdf5_array(mapper_file, key='flatmap_rois')
            ax.imshow(
                np.swapaxes(rois, 0, 1)[::-1], aspect='equal',
                interpolation='bicubic', zorder=2)


def plot_2d_flatmap_from_mapper(voxels_1, voxels_2, mapper_file, ax=None,
                                alpha=0.7, cmap='BuOr_2D', vmin=None,
                                vmax=None, vmin2=None, vmax2=None,
                                with_curvature=True, with_rois=True,
                                with_colorbar=True, label_1='', label_2='',
                                colorbar_location=(.45, .85, .1, .1)):
    """Plot a flatmap from a mapper file, with 2D data.

    This function is equivalent to the pycortex functions:
    cortex.quickshow(cortex.Volume2D(voxels_1, voxels_2, ...), ...)

    Note that this function does not have the full capability of pycortex,
    since it is based on flatmap mappers and not on the original brain
    surface of the subject.

    Parameters
    ----------
    voxels_1 : array of shape (n_voxels, )
        Data to be plotted.
    voxels_2 : array of shape (n_voxels, )
        Data to be plotted.
    mapper_file : str
        File name of the mapper.
    ax : matplotlib Axes or None.
        Axes where the figure will be plotted.
        If None, a new figure is created.
    alpha : float in [0, 1], or array of shape (n_voxels, )
        Transparency of the flatmap.
    cmap : str
        Name of the 2D pycortex colormap.
    vmin : float or None
        Minimum value of the colormap for voxels_1.
        If None, use the 1st percentile of the `voxels_1` array.
    vmax : float or None
        Maximum value of the colormap for voxels_1.
        If None, use the 99th percentile of the `voxels_1` array.
    vmin2 : float or None
        Minimum value of the colormap for voxels_2.
        If None, use the 1st percentile of the `voxels_2` array.
    vmax2 : float or None
        Maximum value of the colormap for voxels_2.
        If None, use the 99th percentile of the `voxels_2` array.
    with_curvature : bool
        If True, show the curvature below the data layer.
    with_rois : bool
        If True, show the ROIs labels above the data layer.
    with_colorbar : bool
        If True, show the colorbar.
    label_1 : str
        Label of voxels_1 in the colormap (xlabel).
    label_2 : str
        Label of voxels_2 in the colormap (ylabel).
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
        ax.axis('off')

    # process plotting parameters
    vmin = np.percentile(voxels_1, 1) if vmin is None else vmin
    vmax = np.percentile(voxels_1, 99) if vmax is None else vmax
    vmin2 = np.percentile(voxels_2, 1) if vmin2 is None else vmin2
    vmax2 = np.percentile(voxels_2, 99) if vmax2 is None else vmax2
    if isinstance(alpha, np.ndarray):
        alpha = map_voxels_to_flatmap(alpha, mapper_file)

    # map to the 2D colormap
    mapped_rgba, cmap_image = _map_to_2d_cmap(voxels_1, voxels_2, vmin=vmin,
                                              vmax=vmax, vmin2=vmin2,
                                              vmax2=vmax2, cmap=cmap)

    # plot the data
    image = map_voxels_to_flatmap(mapped_rgba, mapper_file)
    ax.imshow(image, aspect='equal', zorder=1, alpha=alpha)

    if with_colorbar:
        try:
            cbar = ax.inset_axes(colorbar_location)
        except AttributeError:  # for matplotlib < 3.0
            cbar = ax.figure.add_axes(colorbar_location)
        cbar.imshow(cmap_image, aspect='equal',
                    extent=(vmin, vmax, vmin2, vmax2))
        cbar.set(xlabel=label_1, ylabel=label_2)
        cbar.set(xticks=[vmin, vmax], yticks=[vmin2, vmax2])

    # plot additional layers if present
    _plot_addition_layers(ax=ax, n_voxels=voxels_1.shape[0],
                          mapper_file=mapper_file,
                          with_curvature=with_curvature, with_rois=with_rois)

    return ax


def _map_to_2d_cmap(data1, data2, vmin, vmax, vmin2, vmax2, cmap):
    """Helpers, mapping two data arrays to a 2D colormap.

    Parameters
    ----------
    data1 : array of shape (n_voxels, )
        Data to be plotted.
    data2 : array of shape (n_voxels, )
        Data to be plotted.
    vmin : float or None
        Minimum value of the colormap for data1.
    vmax : float or None
        Maximum value of the colormap for data1.
    vmin2 : float or None
        Minimum value of the colormap for data2.
    vmax2 : float or None
        Maximum value of the colormap for data2.
    cmap : str
        Name of the 2D pycortex colormap.

    Returns
    -------
    mapped_rgba : array of shape (n_voxels, 4)
        2D data mapped to a 2D colormap, with 4 (RGBA) color channels.
    cmap_image : array of shape (height, width, 4)
        2D image of the 2D colormap.
    """
    # load 2D cmap image
    cmap_directory = cortex.options.config.get("webgl", "colormaps")
    cmap_image = plt.imread(os.path.join(cmap_directory, "%s.png" % cmap))

    # Normalize the data
    dim1 = np.clip(Normalize(vmin, vmax)(data1), 0, 1)
    dim2 = np.clip(1 - Normalize(vmin2, vmax2)(data2), 0, 1)

    # 2D indices of the data on the 2D cmap
    dim1 = np.round(dim1 * (cmap_image.shape[1] - 1))
    dim1 = np.nan_to_num(dim1).astype(np.uint32)
    dim2 = np.round(dim2 * (cmap_image.shape[0] - 1))
    dim2 = np.nan_to_num(dim2).astype(np.uint32)

    mapped_rgba = cmap_image[dim2.ravel(), dim1.ravel()]

    # Preserve nan values with alpha = 0
    nans = np.logical_or(np.isnan(data1), np.isnan(data2))
    mapped_rgba[nans, 3] = 0

    return mapped_rgba, cmap_image


def plot_3d_flatmap_from_mapper(voxels_1, voxels_2, voxels_3, mapper_file,
                                ax=None, alpha=0.7, vmin=None, vmax=None,
                                vmin2=None, vmax2=None, vmin3=None, vmax3=None,
                                with_curvature=True, with_rois=True):
    """Plot a flatmap from a mapper file, with 3D data.

    This function is equivalent to the pycortex functions:
    cortex.quickshow(cortex.VolumeRGB(voxels_1, voxels_2, voxels_3, ...), ...)

    Note that this function does not have the full capability of pycortex,
    since it is based on flatmap mappers and not on the original brain
    surface of the subject.

    Parameters
    ----------
    voxels_1 : array of shape (n_voxels, )
        Data to be plotted.
    voxels_2 : array of shape (n_voxels, )
        Data to be plotted.
    voxels_3 : array of shape (n_voxels, )
        Data to be plotted.
    mapper_file : str
        File name of the mapper.
    ax : matplotlib Axes or None.
        Axes where the figure will be plotted.
        If None, a new figure is created.
    alpha : float in [0, 1], or array of shape (n_voxels, )
        Transparency of the flatmap.
    cmap : str
        Name of the 2D pycortex colormap.
    vmin : float or None
        Minimum value of the colormap for voxels_1.
        If None, use the 1st percentile of the `voxels_1` array.
    vmax : float or None
        Maximum value of the colormap for voxels_1.
        If None, use the 99th percentile of the `voxels_1` array.
    vmin2 : float or None
        Minimum value of the colormap for voxels_2.
        If None, use the 1st percentile of the `voxels_2` array.
    vmax2 : float or None
        Maximum value of the colormap for voxels_2.
        If None, use the 99th percentile of the `voxels_2` array.
    vmin3 : float or None
        Minimum value of the colormap for voxels_3.
        If None, use the 1st percentile of the `voxels_3` array.
    vmax3 : float or None
        Maximum value of the colormap for voxels_2.
        If None, use the 99th percentile of the `voxels_3` array.
    with_curvature : bool
        If True, show the curvature below the data layer.
    with_rois : bool
        If True, show the ROIs labels above the data layer.

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
        ax.axis('off')

    # process plotting parameters
    vmin = np.percentile(voxels_1, 1) if vmin is None else vmin
    vmax = np.percentile(voxels_1, 99) if vmax is None else vmax
    vmin2 = np.percentile(voxels_2, 1) if vmin2 is None else vmin2
    vmax2 = np.percentile(voxels_2, 99) if vmax2 is None else vmax2
    vmin3 = np.percentile(voxels_3, 1) if vmin3 is None else vmin3
    vmax3 = np.percentile(voxels_3, 99) if vmax3 is None else vmax3
    if isinstance(alpha, np.ndarray):
        alpha = map_voxels_to_flatmap(alpha, mapper_file)

    # Normalize the data
    voxels_1 = np.clip(Normalize(vmin, vmax)(voxels_1), 0, 1)
    voxels_2 = np.clip(Normalize(vmin2, vmax2)(voxels_2), 0, 1)
    voxels_3 = np.clip(Normalize(vmin3, vmax3)(voxels_3), 0, 1)

    # Preserve nan values with alpha = 0
    nans = np.isnan(voxels_1) + np.isnan(voxels_2) + np.isnan(voxels_3)
    alpha_ = nans == 0
    mapped_rgba = np.stack([voxels_1, voxels_2, voxels_3, alpha_]).T

    # plot the data
    image = map_voxels_to_flatmap(mapped_rgba, mapper_file)
    ax.imshow(image, aspect='equal', zorder=1, alpha=alpha)

    # plot additional layers if present
    _plot_addition_layers(ax=ax, n_voxels=voxels_1.shape[0],
                          mapper_file=mapper_file,
                          with_curvature=with_curvature, with_rois=with_rois)

    return ax
