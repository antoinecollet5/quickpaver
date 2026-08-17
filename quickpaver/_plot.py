"""Efficient rendering of an edge list with matplotlib."""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

NDArrayFloat = np.typing.NDArray[np.floating]


def adjacency_to_edges(
    adjacency: Mapping[int, Iterable[int]],
    *,
    sort: bool = True,
) -> list[tuple[int, int]]:
    """Flatten an adjacency mapping into unique undirected edges.

    Each edge is normalised as ``(min, max)`` so that ``(1, 2)`` and ``(2, 1)``
    are considered the same edge and appear only once in the result.

    Parameters
    ----------
    adjacency : Mapping[int, Iterable[int]]
        Mapping of a node to its neighbours, e.g. ``{0: [1, 2], 1: [0]}``.
        Any mapping of int to an iterable of int is accepted, including
        ``dict[int, list[int]]``.
    sort : bool, optional
        If True (default), the returned edges are sorted in ascending
        lexicographic order, which makes the output deterministic. If False,
        the order follows the iteration order of ``adjacency`` and of its
        values, first occurrence winning.

    Returns
    -------
    list[tuple[int, int]]
        Unique undirected edges as ``(u, v)`` tuples with ``u <= v``.

    Notes
    -----
    Self-loops are preserved as ``(n, n)`` and, like any other edge, appear
    only once. Nodes that appear only as keys with an empty neighbour list
    produce no edge, so isolated nodes are absent from the result.

    Examples
    --------
    >>> adjacency_to_edges({0: [1, 2], 1: [0], 2: [0]})
    [(0, 1), (0, 2)]
    >>> adjacency_to_edges({0: [0, 1], 1: []})
    [(0, 0), (0, 1)]

    """
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []

    for node, neighbours in adjacency.items():
        for neighbour in neighbours:
            edge = (node, neighbour) if node <= neighbour else (neighbour, node)
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)

    return sorted(edges) if sort else edges


def _edges_to_segments(
    edges: Sequence[tuple[int, int]],
    pos: Mapping[int, tuple[float, float]],
) -> np.ndarray:
    """Turn an edge list into an array of line segments.

    Parameters
    ----------
    edges : Sequence[tuple[int, int]]
        Undirected edges as ``(u, v)`` node labels.
    pos : Mapping[int, tuple[float, float]]
        Node label to ``(x, y)`` coordinates. Every node referenced in
        ``edges`` must be present.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_edges, 2, 2)``, where ``segments[i]`` holds the
        two endpoints ``[[x0, y0], [x1, y1]]`` of edge ``i``. This is exactly
        the layout expected by `matplotlib.collections.LineCollection`.

    Notes
    -----
    Node labels are mapped to row indices once, then the whole edge list is
    resolved with a single fancy-indexing operation, so the cost is one
    vectorised gather rather than a Python loop per edge.

    """
    labels = list(pos)
    index = {label: i for i, label in enumerate(labels)}
    points = np.asarray([pos[label] for label in labels], dtype=float)

    edge_idx = np.fromiter(
        (index[n] for edge in edges for n in edge),
        dtype=np.intp,
        count=2 * len(edges),
    ).reshape(-1, 2)

    return points[edge_idx]


def _draw_edges(
    ax: Axes,
    edges: Sequence[tuple[int, int]],
    pos: Mapping[int, tuple[float, float]],
    **kwargs: object,
) -> LineCollection:
    """Draw every edge as a single `LineCollection` artist.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    edges : Sequence[tuple[int, int]]
        Undirected edges as ``(u, v)`` node labels.
    pos : Mapping[int, tuple[float, float]]
        Node label to ``(x, y)`` coordinates.
    **kwargs
        Forwarded to `LineCollection`, e.g. ``colors``, ``linewidths``,
        ``alpha``, ``zorder``. Scalar values apply to all edges; sequences of
        length ``n_edges`` give per-edge styling.

    Returns
    -------
    matplotlib.collections.LineCollection
        The artist added to ``ax``, kept so it can be restyled or removed
        later without rebuilding the figure.

    Notes
    -----
    One artist is created for the whole graph instead of one per edge, which
    is what makes this scale: draw time and memory stay roughly flat in the
    number of Python objects. Per-edge widths and colours are still available
    through ``linewidths`` and ``colors``.

    ``ax.add_collection`` does not update the data limits on its own, so
    ``ax.autoscale_view`` is called here after the limits are extended.

    """
    segments = _edges_to_segments(edges, pos)
    collection = LineCollection(segments, **kwargs)  # ty:ignore[invalid-argument-type]
    ax.add_collection(collection)

    if len(segments):
        flat = segments.reshape(-1, 2)
        ax.update_datalim(flat)
        ax.autoscale_view()

    return collection


def draw_adjacency(
    ax: Axes,
    points: NDArrayFloat,
    adjacency_dict: dict[int, list[int]],
    *,
    node_kwargs: Mapping[str, object] | None = None,
    **kwargs: object,
) -> LineCollection:
    """Draw a graph described by an adjacency mapping on ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    points : NDArrayFloat
        Array of shape ``(n_nodes, 2)`` giving the ``(x, y)`` position of
        each node. A node's row index in ``points`` doubles as its integer
        label, i.e. the labels used as keys/values in ``adjacency_dict``.
    adjacency_dict : Dict[int, List[int]]
        Mapping of a node to its neighbours, see `adjacency_to_edges`.
    node_kwargs : Mapping[str, object] or None, optional
        If given, node markers are additionally drawn as a single `Line2D`
        artist built from ``points`` and these keyword arguments (e.g.
        ``marker``, ``markersize``, ``color``). If None (default), no node
        markers are drawn.
    **kwargs
        Forwarded to `LineCollection` for edge styling, e.g. ``colors``,
        ``linewidths``, ``alpha``, ``zorder``.

    Returns
    -------
    matplotlib.collections.LineCollection
        The edge artist added to ``ax``.

    Notes
    -----
    Every neighbour referenced in ``adjacency_dict`` must be a valid row
    index into ``points``; a ``KeyError`` is raised otherwise. As with
    `_draw_edges`, both edges and (optionally) nodes are each drawn as a
    single artist, so rendering cost stays roughly flat in the number of
    Python objects regardless of graph size.

    """
    edges = adjacency_to_edges(adjacency_dict)
    pos = {label: (float(x), float(y)) for label, (x, y) in enumerate(points)}
    collection = _draw_edges(ax, edges, pos, **kwargs)

    if node_kwargs is not None:
        line = Line2D(points[:, 0], points[:, 1], linestyle="None", **node_kwargs)  # ty:ignore[invalid-argument-type]
        ax.add_line(line)

    return collection
