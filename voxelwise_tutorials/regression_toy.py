"""These helper functions are used in the tutorial on ridge regression
(tutorials/movies/02_plot_ridge_regression.py).
"""
import numpy as np
import matplotlib.pyplot as plt

COEFS = np.array([0.4, 0.3])


def create_regression_toy(n_samples=50, n_features=2, noise=0.3, correlation=0,
                          random_state=0):
    """Create a regression toy dataset."""
    if n_features > 2:
        raise ValueError("n_features must be <= 2.")

    # create features
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    X -= X.mean(0)
    X /= X.std(0)

    # makes correlation(X[:, 1], X[:, 0]) = correlation
    if n_features == 2:
        X[:, 1] -= (X[:, 0] @ X[:, 1]) * X[:, 0] / (X[:, 0] @ X[:, 0])
        X /= X.std(0)
        if correlation != 0:
            X[:, 1] *= np.sqrt(correlation ** (-2) - 1)
            X[:, 1] += X[:, 0] * np.sign(correlation)
        X /= X.std(0)

    # create linear coefficients
    coefs = COEFS[:n_features]

    # create target
    y = X @ coefs
    y += rng.randn(*y.shape) * noise

    return X, y


def l2_loss(X, y, coefs):
    return np.sum((X @ coefs - y[:, None]) ** 2, axis=0)


def plot_1d(X, y, coefs):
    coefs = np.atleast_1d(coefs)

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.5))

    # left plot: y = f(x)
    ax = axes[0]
    ax.scatter(X, y, alpha=0.5, color="C0")
    ylim = ax.get_ylim()
    ax.plot([X.min(), X.max()],
            [X.min() * coefs[0], X.max() * coefs[0]], color="C1")
    ax.set(xlabel="x1", ylabel="y", ylim=ylim)
    ax.grid()
    for xx, yy in zip(X[:, 0], y):
        ax.plot([xx, xx], [yy, xx * coefs[0]], c='gray', alpha=0.5)

    # right plot: loss = f(w)
    ax = axes[1]
    coefs_range = np.linspace(-0.1, 0.8, 100)
    ax.plot(coefs_range, l2_loss(X, y, coefs_range[None]), color="C2")
    ax.scatter([coefs[0]], l2_loss(X, y, coefs[:, None]), color="C1")
    ax.set(xlabel="w1", ylabel="Squared loss")
    ax.grid()

    fig.tight_layout()
    plt.show()


def plot_2d(X, y, coefs, flat=True, alpha=None, show_noiseless=True):
    from mpl_toolkits import mplot3d  # noqa
    coefs = np.array(coefs)

    fig = plt.figure(figsize=(6.7, 2.5))

    # left plot: y = f(x)
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], y, alpha=0.5, color="C0")
    xmin, xmax = X.min(), X.max()
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 10),
                         np.linspace(xmin, xmax, 10))
    ax.plot_surface(xx, yy, xx * coefs[0] + yy * coefs[1], color="C1",
                    alpha=0.4, edgecolor='gray')
    ax.set(xlabel="x1", ylabel="x2", zlabel="y", zlim=[yy.min(), yy.max()])

    # right plot: loss = f(w)
    if flat:
        ax = fig.add_subplot(122)
        ww1, ww2 = np.meshgrid(np.linspace(-0.1, 0.9, 100),
                               np.linspace(-0.1, 0.9, 100))
        coefs_range = np.stack([ww1.ravel(), ww2.ravel()])
        zz = l2_loss(X, y, coefs_range).reshape(ww1.shape)
        ax.imshow(zz, extent=(ww1.min(), ww1.max(), ww2.min(), ww2.max()),
                  origin="lower")
        im = ax.contourf(ww1, ww2, zz, levels=20, vmax=zz.max(),
                         extent=(ww1.min(), ww1.max(), ww2.min(), ww2.max()),
                         origin="lower")
        if alpha is not None:
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            angle = np.linspace(-np.pi, np.pi, 100)
            radius = np.sqrt(np.sum(coefs ** 2))
            ax.plot(np.cos(angle) * radius, np.sin(angle) * radius, c='k')
            ax.set_xlim(xlim), ax.set_ylim(ylim)
        ax.scatter([coefs[0]], [coefs[1]], color="C1", s=[20])
        if show_noiseless:
            ax.scatter([COEFS[0]], [COEFS[1]], color="k", s=[20], marker="x")
            ax.legend(["w/ noise", "wo/ noise"], framealpha=0.2)
        ax.set(xlabel="w1", ylabel="w2")

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set(ylabel="Squared loss")

    else:  # 3D version of the right plot
        ax = fig.add_subplot(122, projection='3d')
        ww1, ww2 = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        coefs_range = np.stack([ww1.ravel(), ww2.ravel()])
        zz = l2_loss(X, y, coefs_range).reshape(ww1.shape)
        ax.plot_surface(ww1, ww2, zz, color="C2", alpha=0.4, edgecolor='gray')
        ax.scatter3D([coefs[0]], [coefs[1]], [l2_loss(X, y, coefs[:, None])],
                     color="C1")
        ax.set(xlabel="ww2", ylabel="w2", zlabel="Squared loss")

    fig.tight_layout()
    plt.show()
