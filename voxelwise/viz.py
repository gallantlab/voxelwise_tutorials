import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_hist2d(scores_1, scores_2, bins=100, cmin=1, vmin=None, vmax=None,
                ax=None, norm=LogNorm(), cmap="viridis", colorbar=True):
    if vmin is None:
        vmin = min(scores_1.min(), scores_2.min())
    if vmax is None:
        vmax = max(scores_1.max(), scores_2.max())
    if isinstance(bins, int):
        bins = np.linspace(vmin, vmax, bins)
    if ax is None:
        ax = plt.gca()

    h = ax.hist2d(scores_1, scores_2, bins=bins, cmin=cmin, norm=norm,
                  cmap=cmap)
    if colorbar:
        ax.figure.colorbar(h[3], ax=ax)

    ax.plot([vmin, vmax], [vmin, vmax], color='k', linewidth=0.5)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.grid()
    return ax
