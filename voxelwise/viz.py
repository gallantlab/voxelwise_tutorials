import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


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
