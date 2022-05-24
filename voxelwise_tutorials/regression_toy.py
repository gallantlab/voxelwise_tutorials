"""These helper functions are used in the tutorial on ridge regression
(tutorials/shortclips/02_plot_ridge_regression.py).
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
    w = COEFS[:n_features]

    # create target
    y = X @ w
    y += rng.randn(*y.shape) * noise

    return X, y


def l2_loss(X, y, w):
    if w.ndim == 1:
        w = w[:, None]
    return np.sum((X @ w - y[:, None]) ** 2, axis=0)


def ridge(X, y, alpha):
    n_features = X.shape[1]
    return np.linalg.solve(X.T @ X + np.eye(n_features) * alpha, X.T @ y)


def plot_1d(X, y, w):
    w = np.atleast_1d(w)

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.5))

    # left plot: y = f(x)
    ax = axes[0]
    ax.scatter(X, y, alpha=0.5, color="C0")
    ylim = ax.get_ylim()
    ax.plot([X.min(), X.max()], [X.min() * w[0], X.max() * w[0]], color="C1")
    ax.set(xlabel="X[:, 0]", ylabel="y", ylim=ylim)
    ax.grid()
    for xx, yy in zip(X[:, 0], y):
        ax.plot([xx, xx], [yy, xx * w[0]], c='gray', alpha=0.5)

    # right plot: loss = f(w)
    ax = axes[1]
    w_range = np.linspace(-0.1, 0.8, 100)
    ax.plot(w_range, l2_loss(X, y, w_range[None]), color="C2")
    ax.scatter([w[0]], l2_loss(X, y, w), color="C1")
    ax.set(xlabel="w[0]", ylabel="Squared loss")
    ax.grid()

    fig.tight_layout()
    plt.show()


def plot_2d(X, y, w, flat=True, alpha=None, show_noiseless=True):
    from mpl_toolkits import mplot3d  # noqa
    w = np.array(w)

    fig = plt.figure(figsize=(6.7, 2.5))

    #####################
    # left plot: y = f(x)

    try:  # computed_zorder is only available in matplotlib >= 3.4
        ax = fig.add_subplot(121, projection='3d', computed_zorder=False)
    except AttributeError:
        ax = fig.add_subplot(121, projection='3d')

    # to help matplotlib displays scatter points behind any surface, we
    # first plot the point below, then the surface, then the points above,
    # and use computed_zorder=False.
    above = y > X @ w
    ax.scatter3D(X[~above, 0], X[~above, 1], y[~above], alpha=0.5, color="C0")

    xmin, xmax = X.min(), X.max()
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 10),
                         np.linspace(xmin, xmax, 10))
    ax.plot_surface(xx, yy, xx * w[0] + yy * w[1], color=[0, 0, 0, 0],
                    edgecolor=[1., 0.50, 0.05, 0.50])

    # plot the point above *after* the surface
    ax.scatter3D(X[above, 0], X[above, 1], y[above], alpha=0.5, color="C0")

    ax.set(xlabel="X[:, 0]", ylabel="X[:, 1]", zlabel="y",
           zlim=[yy.min(), yy.max()])

    #########################
    # right plot: loss = f(w)
    if flat:
        ax = fig.add_subplot(122)
        w0, w1 = np.meshgrid(np.linspace(-0.1, 0.9, 100),
                             np.linspace(-0.1, 0.9, 100))
        w_range = np.stack([w0.ravel(), w1.ravel()])
        zz = l2_loss(X, y, w_range).reshape(w0.shape)
        # zz_reg = (w_range ** 2).sum(0).reshape(w0.shape) * alpha
        ax.imshow(zz, extent=(w0.min(), w0.max(), w1.min(), w1.max()),
                  origin="lower")
        im = ax.contourf(w0, w1, zz, levels=20, vmax=zz.max(),
                         extent=(w0.min(), w0.max(), w1.min(), w1.max()),
                         origin="lower")

        ax.scatter([w[0]], [w[1]], color="C1", s=[20], label="w")
        if show_noiseless:
            ax.scatter([COEFS[0]], [COEFS[1]], color="k", s=[20], marker="x",
                       label="w_noiseless")
            if alpha is not None:
                w_ols = np.linalg.solve(X.T @ X, X.T @ y)
                ax.scatter([w_ols[0]], [w_ols[1]], color="C3", s=[20],
                           marker="o", label="w_OLS")
            ax.legend(framealpha=0.2)

        if alpha is not None:
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            angle = np.linspace(-np.pi, np.pi, 100)
            radius = np.sqrt(np.sum(w ** 2))
            ax.plot(np.cos(angle) * radius, np.sin(angle) * radius, c='k')
            ax.set_xlim(xlim), ax.set_ylim(ylim)

        ax.set(xlabel="w[0]", ylabel="w[1]")
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set(ylabel="Squared loss")

    else:  # 3D version of the right plot
        ax = fig.add_subplot(122, projection='3d')
        w0, w1 = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        w_range = np.stack([w0.ravel(), w1.ravel()])
        zz = l2_loss(X, y, w_range).reshape(w0.shape)
        ax.plot_surface(w0, w1, zz, color="C2", alpha=0.4, edgecolor='gray')
        ax.scatter3D([w[0]], [w[1]], [l2_loss(X, y, w)], color="C1")
        ax.set(xlabel="w[0]", ylabel="w[1]", zlabel="Squared loss")

    fig.tight_layout()
    plt.show()


def plot_kfold2(X, y, alpha=0, fit=True, flip=False):
    half = X.shape[0] // 2

    if not fit:
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.5))
        ax.scatter(X[:half], y[:half], alpha=0.5, color="C0")
        ax.scatter(X[half:], y[half:], alpha=0.5, color="C1")
        ax.set(xlabel="x1", ylabel="y")
        ax.grid()
        fig.tight_layout()
        plt.show()
        return None

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.5), sharex=True,
                             sharey=True)

    w_ridge1 = ridge(X[:half], y[:half], alpha)
    w_ridge2 = ridge(X[half:], y[half:], alpha)

    ax = axes[0]
    if not flip:
        ax.scatter(X[:half], y[:half], alpha=0.5, color="C0")
    else:
        ax.scatter(X[half:], y[half:], alpha=0.5, color="C1")
    ax.plot([X.min(), X.max()],
            [X.min() * w_ridge1, X.max() * w_ridge1], color="C0")
    ax.set(xlabel="X[:, 0]", ylabel="y", title='model 1')
    ax.grid()

    ax = axes[1]
    if flip:
        ax.scatter(X[:half], y[:half], alpha=0.5, color="C0")
    else:
        ax.scatter(X[half:], y[half:], alpha=0.5, color="C1")
    ax.plot([X.min(), X.max()],
            [X.min() * w_ridge2, X.max() * w_ridge2], color="C1")
    ax.set(xlabel="X[:, 0]", ylabel="y", title='model 2')
    ax.grid()

    fig.tight_layout()
    plt.show()


def plot_cv_path(X, y):
    losses = []
    alphas = np.logspace(-2, 4, 12)

    half = X.shape[0] // 2
    for alpha in alphas:

        w_ridge1 = ridge(X[:half], y[:half], alpha)
        w_ridge2 = ridge(X[half:], y[half:], alpha)

        losses.append(
            l2_loss(X[half:], y[half:], w_ridge1) +
            l2_loss(X[:half], y[:half], w_ridge2))

    best = np.argmin(losses)

    # final cv plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.semilogx(alphas, losses, '-o', label="candidates")
    ax.set(xlabel="alpha", ylabel="cross-validation error")
    ax.plot([alphas[best]], [losses[best]], "o", c="C3", label="best")
    ax.legend()
    fig.tight_layout()
    plt.show()
