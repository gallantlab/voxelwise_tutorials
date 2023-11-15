import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

cache = dict()


def load_wordnet(directory=None, recache=False):
    """Load the wordnet graph and wordnet categories used in [Huth et al 2012].

    Parameters
    ----------
    directory : str or None
        Directory where the dataset has been downloaded. If None, use
        "shortclips" in ``voxelwise_tutorials.io.get_data_home()``.

    Returns
    -------
    wordnet_graph : networkx MultiDiGraph
        Graph of the wordnet categories (1583 nodes).
    wordnet_categories : list of str
        Names of the wordnet categories (1705 str).

    References
    ----------
    Huth, A. G., Nishimoto, S., Vu, A. T., & Gallant, J. L. (2012). A
    continuous semantic space describes the representation of thousands of
    object and action categories across the human brain. Neuron, 76(6),
    1210-1224.
    """
    if ("wordnet_graph" in cache and "wordnet_categories" in cache
            and not recache):  # noqa
        return cache["wordnet_graph"], cache["wordnet_categories"]

    import os
    import networkx

    if directory is None:
        from voxelwise_tutorials.io import get_data_home
        directory = get_data_home("shortclips")

    dot_file = os.path.join(directory, 'utils', 'wordnet_graph.dot')
    txt_file = os.path.join(directory, 'utils', 'wordnet_categories.txt')

    wordnet_graph = networkx.drawing.nx_pydot.read_dot(dot_file)
    with open(txt_file) as fff:
        wordnet_categories = fff.read().splitlines()

    # Remove nodes in the graph that aren't in the categories list
    for name in list(wordnet_graph.nodes().keys()):
        if name not in wordnet_categories:
            wordnet_graph.remove_node(name)

    cache["wordnet_graph"] = wordnet_graph
    cache["wordnet_categories"] = wordnet_categories

    return wordnet_graph, wordnet_categories


def correct_coefficients(primal_coef, feature_names, norm_by_depth=True):
    """Corrects coefficients across wordnet features as in [Huth et al 2012].

    Parameters
    ----------
    primal_coef : array of shape (n_features, ...)
        Regression coefficient on all wordnet features.
    feature_names : list of str, of length (n_features, )
        Names of the wordnet features.
    norm_by_depth : bool
        If True, normalize the correction by the number of ancestors.

    Returns
    -------
    corrected_coef : array of shape (n_features, ...)
        Corrected coefficient.

    References
    ----------
    Huth, A. G., Nishimoto, S., Vu, A. T., & Gallant, J. L. (2012). A
    continuous semantic space describes the representation of thousands of
    object and action categories across the human brain. Neuron, 76(6),
    1210-1224.
    """
    import itertools
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.corpus import wordnet

    def _get_hypernyms(name):
        hypernyms = set()
        for path in wordnet.synset(name).hypernym_paths():
            hypernyms = hypernyms.union(path)
        return list(hypernyms)

    assert primal_coef.shape[0] == len(feature_names)

    feature_names = list(feature_names)
    corrected_coef = np.zeros_like(primal_coef)

    for ii, name in enumerate(feature_names):
        for hypernym in _get_hypernyms(name):
            if hypernym.name() in feature_names:

                idx = feature_names.index(hypernym.name())
                update = primal_coef[idx]

                if norm_by_depth:
                    ancestors = [
                        hh.name()
                        for hh in itertools.chain(*hypernym.hypernym_paths())
                        if hh.name() in feature_names
                    ]
                    update = update / len(ancestors)

                corrected_coef[ii] += update

    return corrected_coef


DEFAULT_HIGHLIGHTED_NODES = [
    "move.v.03", "turn.v.01", "jump.v.01", "change.v.02", "lean.v.01",
    "bloom.v.01", "travel.v.01", "gallop.v.01", "walk.v.01", "rappel.v.01",
    "touch.v.01", "hit.v.03", "move.v.02", "drag.v.01", "consume.v.02",
    "fasten.v.01", "breathe.v.01", "organism.n.01", "animal.n.01",
    "plant.n.02", "person.n.01", "athlete.n.01", "arthropod.n.01", "fish.n.01",
    "reptile.n.01", "bird.n.01", "placental.n.01", "rodent.n.01",
    "ungulate.n.01", "carnivore.n.01", "plant_organ.n.01",
    "geological_formation.n.01", "hill.n.01", "location.n.01", "city.n.01",
    "grassland.n.01", "body_part.n.01", "leg.n.01", "eye.n.01", "matter.n.03",
    "food.n.01", "sky.n.01", "water.n.01", "material.n.01", "bamboo.n.01",
    "atmospheric_phenomenon.n.01", "mist.n.01", "artifact.n.01", "way.n.06",
    "road.n.01", "clothing.n.01", "structure.n.01", "building.n.01",
    "room.n.01", "shop.n.01", "door.n.01", "implement.n.01", "kettle.n.01",
    "equipment.n.01", "ball.n.01", "vehicle.n.01", "boat.n.01",
    "wheeled_vehicle.n.01", "car.n.01", "furniture.n.01", "device.n.01",
    "weapon.n.01", "gas_pump.n.01", "container.n.01", "bottle.n.01",
    "laptop.n.01", "group.n.01", "herd.n.01", "measure.n.02",
    "communication.n.02", "text.n.01", "attribute.n.02", "dirt.n.02",
    "event.n.01", "rodeo.n.01", "wave.n.01", "communicate.v.02", "talk.v.02",
    "rub.v.03"
]


def plot_wordnet_graph(node_colors, node_sizes, zorder=None, node_scale=200,
                       alpha=1.0, ax=None, extra_edges=None,
                       highlighted_nodes="default", directory=None):
    """Plot a wordnet graph, as in [Huth et al 2012].

    Note: Only plot categories that are in the wordnet graph loaded in the
    function ``load_wordnet``.

    Parameters
    ----------
    node_colors : array of shape (1705, 3)
        RGB colors of each feature. If you want to plot an array of shape
        (1705, ) use ``apply_cmap`` to map it to RGB.
    node_sizes : array of shape (1705, )
        Size of each feature. Values are scaled by the maximum.
    zorder : array of shape (1705, ) or None
        Order of node, larger values are plotted on top.
    node_scale : float
        Scaling factor for the node sizes.
    alpha : float
        Transparency of the nodes.
    ax : Axes or None
        Matplotlib Axes where the grap will be plotted. If None, the current
        figure is used.
    extra_edges : list of (str, str)
        Add extra edges between named nodes. See the function ``load_wordnet``
        to have the list of names.
    highlighted_nodes : list of str, or in {"default", "random_42"}
        List of nodes to be highlighted (with name). If "default", use a fixed
        list of 84 nodes. If "random_42", choose 42 random nodes. See the
        function ``load_wordnet`` to have the list of names.
    directory : str or None
        Directory where the dataset has been downloaded. If None, use
        "shortclips" in ``voxelwise_tutorials.io.get_data_home()``.

    Returns
    -------
    ax : Axes
        Matplotlib Axes where the histogram was plotted.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from voxelwise_tutorials.wordnet import plot_wordnet_graph, apply_cmap
    >>> node_colors = np.random.rand(1705, 3)
    >>> node_sizes = np.random.rand(1705) + 0.5
    >>> plot_wordnet_graph(node_colors=node_colors, node_sizes=node_sizes)
    >>> plt.show()

    References
    ----------
    Huth, A. G., Nishimoto, S., Vu, A. T., & Gallant, J. L. (2012). A
    continuous semantic space describes the representation of thousands of
    object and action categories across the human brain. Neuron, 76(6),
    1210-1224.
    """
    import networkx

    ################
    # initialization

    if ax is None:
        fig = plt.figure(figsize=(8.5, 8.5))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])

    wordnet_graph, wordnet_categories = load_wordnet(directory=directory)

    ################
    # remove features not in the nodes (1705 -> 1583)
    node_names = list(wordnet_graph.nodes().keys())
    indices = [
        ii for ii, name in enumerate(wordnet_categories) if name in node_names
    ]
    order = [
        list(np.array(wordnet_categories)[indices]).index(name)
        for name in node_names
    ]
    np.testing.assert_array_equal(node_names,
                                  np.array(wordnet_categories)[indices][order])
    node_colors = node_colors[indices][order]
    node_sizes = node_sizes[indices][order]
    if zorder is not None:
        zorder = zorder[indices][order]

    assert len(node_sizes) == len(node_names)

    ################
    # Various checks
    if node_colors.min() < 0:
        raise ValueError(
            'Negative value in node_colors, all values should be in [0, 1].')
    if node_colors.max() > 1:
        raise ValueError(
            'Negative value in node_colors, all values should be in [0, 1].')
    if node_sizes.min() < 0:
        raise ValueError(
            'Negative value in node_sizes, all values should be non-negative.')
    node_sizes /= node_sizes.max()

    ################
    # highlighted nodes
    if highlighted_nodes == "default":
        highlighted_nodes = DEFAULT_HIGHLIGHTED_NODES
    elif highlighted_nodes == "random_42":
        highlighted_nodes = [
            node_names[ii] for ii in np.random.choice(len(node_names), 42)
        ]
    elif highlighted_nodes is None:
        highlighted_nodes = []

    ################
    # create node positions and size dictionaries
    node_positions = dict([(key.strip('"'),
                            list(map(float, val['pos'].strip('"').split(","))))
                           for key, val in wordnet_graph.nodes().items()])
    node_sizes = node_sizes * node_scale
    node_sizes_dict = dict(zip(node_names, node_sizes))

    ################
    # plot edges using LineCollection
    edges = wordnet_graph.edges()
    linestyles = [
        ':' if np.isnan([node_sizes_dict[e[0]], node_sizes_dict[e[1]]]).any()
        else '-' for e in edges
    ]
    edge_positions = np.asarray([(node_positions[e[0]], node_positions[e[1]])
                                 for e in edges])
    edge_collection = LineCollection(edge_positions,
                                     colors=(0.7, 0.7, 0.7, 1.0),
                                     linewidths=0.7, antialiaseds=(1, ),
                                     linestyles=linestyles,
                                     transOffset=ax.transData)
    edge_collection.set_zorder(1)  # edges go behind nodes
    ax.add_collection(edge_collection)

    if extra_edges:
        # include these edges that aren't part of the layout
        extra_edge_positions = np.asarray([
            (node_positions[e[0]], node_positions[e[1]]) for e in extra_edges
        ])
        extra_edge_collection = LineCollection(
            extra_edge_positions, colors=(0.7, 0.7, 0.7, 1.0), linewidths=0.5,
            antialiaseds=(1, ), linestyle='-', transOffset=ax.transData)

        extra_edge_collection.set_zorder(1)
        ax.add_collection(extra_edge_collection)

    ################
    # plot nodes with scatter
    xy = np.asarray([node_positions[v] for v in node_names])
    highlighted_node_indices = [
        node_names.index(hn) for hn in highlighted_nodes
    ]

    kind_dict = dict(n="o", v="s")
    node_kinds = np.array([n.split(".")[1] for n in node_names])
    edgecolors = ["none"] * xy.shape[0]
    for hni in highlighted_node_indices:
        edgecolors[hni] = "white"

    for node_kind, marker in kind_dict.items():
        indices = np.nonzero(node_kinds == node_kind)[0]

        # reorder to have highlighted nodes on top
        if highlighted_node_indices:
            hnvec = np.zeros((indices.shape[0], ))
            intersect = np.intersect1d(highlighted_node_indices, indices)
            if intersect.size > 0:
                hnvec[np.array([list(indices).index(hn)
                                for hn in intersect])] = 1
                indices = indices[np.argsort(hnvec)]

        if zorder is not None:
            orders = zorder[indices]
            indices = indices[np.argsort(orders)]

        # normalize area of squares to be same as circles
        if marker == "s":
            norm_sizes = np.nan_to_num(node_sizes[indices]) * (np.pi / 4.0)
        else:
            norm_sizes = np.nan_to_num(node_sizes[indices])

        ax.scatter(xy[indices, 0], xy[indices, 1], s=norm_sizes,
                   c=node_colors[indices], marker=marker,
                   edgecolors=list(np.array(edgecolors)[indices]), alpha=alpha)

    ################
    # add labels for the highlighted nodes
    labels = dict([(name, name.split('.')[0]) for name in highlighted_nodes])
    pos = dict([(n, (x, y - 60)) for (n, (x, y)) in node_positions.items()])
    networkx.draw_networkx_labels(wordnet_graph, font_color='white',
                                  labels=labels, font_weight="bold", pos=pos)

    return ax


def scale_to_rgb_cube(node_colors, clip=2.0):
    """Scale array to RGB cube, as in [Huth et al 2012].

    Parameters
    ----------
    node_colors : array of shape (n_nodes, 3)
        RGB colors of each node, raw values.
    clip : float
        After z-scoring, values outside [-clip, clip] will be clipped.

    Returns
    -------
    node_colors : array of shape (n_nodes, 3)
        Transformed RGB colors of each node, in [0, 1]

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from voxelwise_tutorials.wordnet import plot_wordnet_graph
    >>> from voxelwise_tutorials.wordnet import scale_to_rgb_cube
    >>> node_colors = np.random.randn(1705, 3)
    >>> node_sizes = np.random.rand(1705) + 0.5
    >>> plot_wordnet_graph(node_colors=scale_to_rgb_cube(node_colors),
                           node_sizes=node_sizes)
    >>> plt.show()

    References
    ----------
    Huth, A. G., Nishimoto, S., Vu, A. T., & Gallant, J. L. (2012). A
    continuous semantic space describes the representation of thousands of
    object and action categories across the human brain. Neuron, 76(6),
    1210-1224.
    """
    # z-score and clip
    node_colors = node_colors.copy()
    node_colors -= node_colors.mean(0)
    node_colors /= node_colors.std(0)
    node_colors[node_colors > clip] = clip
    node_colors[node_colors < -clip] = -clip

    # normalize each node by the L_inf norm and the L_2 norm
    l2_norm = np.linalg.norm(node_colors, axis=1)
    linf_norm = np.max(np.abs(node_colors), axis=1)
    node_colors *= (l2_norm / linf_norm)[:, None]

    # clip again
    node_colors[node_colors > clip] = clip
    node_colors[node_colors < -clip] = -clip

    # from [-clip, clip] to [0, 1]
    node_colors = node_colors / clip / 2.0 + 0.5

    return node_colors


def apply_cmap(data, cmap=None, vmin=None, vmax=None, n_colors=None,
               norm=None):
    """Apply a colormap to a 1D array, to get RGB colors.

    Parameters
    ----------
    data : array of shape (n_features, )
        Input array.
    cmap : str or None
        Matplotlib colormap.
    vmin : float or None
        Minimum value of the color mapping. If None, use ``data.min()``.
        Only used if ``norm`` is None.
    vmax : float or None
        Maximum value of the color mapping. If None, use ``data.max()``.
        Only used if ``norm`` is None.
    n_colors : int or None
        If not None, use a discretized version of the colormap.
    norm : matplotlib.colors.Normalize instance, or None
        The normalizing object which scales data, typically into the
        interval ``[0, 1]``. If None, it defaults to a
        ``matplotlib.colors.Normalize`` object using ``vmin`` and ``vmax``.

    Returns
    -------
    data_rgb : array of shape (n_features, 3)
        Input array mapped to RGB.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from voxelwise_tutorials.wordnet import plot_wordnet_graph, apply_cmap
    >>> node_colors = np.random.rand(1705,)
    >>> node_sizes = np.random.rand(1705) + 0.5
    >>> plot_wordnet_graph(node_colors=apply_cmap(node_colors),
                           node_sizes=node_sizes)
    >>> plt.show()
    """
    from matplotlib import cm
    cmap = plt.get_cmap(cmap, lut=n_colors)
    if norm is None:
        from matplotlib import colors
        norm = colors.Normalize(vmin, vmax)
    cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    data_rgb = cmapper.to_rgba(data)
    return data_rgb
