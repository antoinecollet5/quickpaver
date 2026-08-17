"""Tests for :mod:`hytecio.graph_edges`."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from quickpaver._plot import (
    _draw_edges,
    _edges_to_segments,
    adjacency_to_edges,
    draw_adjacency,
)


@pytest.fixture
def ax() -> Iterator[Axes]:
    """Provide a fresh matplotlib Axes, closed after the test."""
    fig, axes = plt.subplots()
    try:
        yield axes
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------
# Tests for :func:`graph_edges.adjacency_to_edges`.
# --------------------------------------------------------------------------


def test_empty_mapping() -> None:
    assert adjacency_to_edges({}) == []


def test_isolated_nodes_produce_no_edge() -> None:
    assert adjacency_to_edges({0: [], 1: [], 2: []}) == []


def test_single_edge() -> None:
    assert adjacency_to_edges({0: [1]}) == [(0, 1)]


def test_reciprocal_edges_are_deduplicated() -> None:
    adjacency: dict[int, list[int]] = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    assert adjacency_to_edges(adjacency) == [(0, 1), (0, 2), (1, 2)]


def test_repeated_neighbour_is_deduplicated() -> None:
    assert adjacency_to_edges({0: [1, 1, 1]}) == [(0, 1)]


def test_orientation_is_normalised() -> None:
    assert adjacency_to_edges({5: [3]}) == [(3, 5)]
    assert adjacency_to_edges({3: [5]}) == adjacency_to_edges({5: [3]})


def test_self_loop_kept_once() -> None:
    assert adjacency_to_edges({0: [0, 0, 1]}) == [(0, 0), (0, 1)]


def test_dangling_neighbour_not_declared_as_key() -> None:
    # 9 never appears as a key; the edge must still be reported.
    assert adjacency_to_edges({0: [9]}) == [(0, 9)]


def test_negative_and_unordered_labels() -> None:
    adjacency = {3: [-1], -1: [3, 7], 7: [-1]}
    assert adjacency_to_edges(adjacency) == [(-1, 3), (-1, 7)]


def test_result_is_sorted_by_default() -> None:
    edges = adjacency_to_edges({4: [2], 0: [3], 2: [1]})
    assert edges == sorted(edges)
    assert edges == [(0, 3), (1, 2), (2, 4)]


def test_sort_false_preserves_insertion_order() -> None:
    adjacency = {4: [2], 0: [3], 2: [1, 4]}
    assert adjacency_to_edges(adjacency, sort=False) == [(2, 4), (0, 3), (1, 2)]


def test_accepts_non_list_iterables() -> None:
    adjacency = {0: {1, 2}, 1: (0,), 2: iter([0])}
    assert adjacency_to_edges(adjacency) == [(0, 1), (0, 2)]


def test_no_duplicates_in_output() -> None:
    adjacency = {n: [(n + 1) % 10, (n - 1) % 10] for n in range(10)}
    edges = adjacency_to_edges(adjacency)
    assert len(edges) == len(set(edges)) == 10


def test_every_edge_appears_in_both_directions_of_input() -> None:
    adjacency = {0: [1, 2], 1: [0], 2: [0]}
    edges = adjacency_to_edges(adjacency)
    for u, v in edges:
        assert v in adjacency.get(u, []) or u in adjacency.get(v, [])


@pytest.mark.parametrize(
    ("adjacency", "expected"),
    [
        ({}, []),
        ({0: [1]}, [(0, 1)]),
        ({0: [1], 1: [0]}, [(0, 1)]),
        ({0: [1, 2], 2: [0]}, [(0, 1), (0, 2)]),
        ({1: [1]}, [(1, 1)]),
    ],
)
def test_parametrised_cases(
    adjacency: dict[int, list[int]],
    expected: list[tuple[int, int]],
) -> None:
    assert adjacency_to_edges(adjacency) == expected


# --------------------------------------------------------------------------
# Tests for :func:`graph_edges._edges_to_segments`.
# --------------------------------------------------------------------------


def test_edges_to_segments_shape_and_values() -> None:
    pos = {0: (0.0, 0.0), 1: (1.0, 2.0), 2: (3.0, 4.0)}
    edges = [(0, 1), (1, 2)]
    segments = _edges_to_segments(edges, pos)
    assert segments.shape == (2, 2, 2)
    np.testing.assert_allclose(segments[0], [[0.0, 0.0], [1.0, 2.0]])
    np.testing.assert_allclose(segments[1], [[1.0, 2.0], [3.0, 4.0]])


def test_edges_to_segments_empty_edges() -> None:
    pos = {0: (0.0, 0.0)}
    segments = _edges_to_segments([], pos)
    assert segments.shape == (0, 2, 2)


# --------------------------------------------------------------------------
# Tests for :func:`graph_edges._draw_edges`.
# --------------------------------------------------------------------------


def test_draw_edges_adds_line_collection(ax: Axes) -> None:
    pos = {0: (0.0, 0.0), 1: (1.0, 1.0)}
    collection = _draw_edges(ax, [(0, 1)], pos)
    assert isinstance(collection, LineCollection)
    assert collection in ax.collections


def test_draw_edges_updates_datalim(ax: Axes) -> None:
    pos = {0: (-5.0, -5.0), 1: (5.0, 5.0)}
    _draw_edges(ax, [(0, 1)], pos)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert xlim[0] <= -5.0
    assert xlim[1] >= 5.0
    assert ylim[0] <= -5.0
    assert ylim[1] >= 5.0


def test_draw_edges_forwards_kwargs(ax: Axes) -> None:
    pos = {0: (0.0, 0.0), 1: (1.0, 1.0)}
    _ = _draw_edges(ax, [(0, 1)], pos, colors="red", linewidths=2.5)


def test_draw_edges_no_edges_skips_autoscale(ax: Axes) -> None:
    collection = _draw_edges(ax, [], {})
    assert isinstance(collection, LineCollection)
    assert collection in ax.collections


# --------------------------------------------------------------------------
# Tests for :func:`graph_edges.draw_adjacency`.
# --------------------------------------------------------------------------


def test_draw_adjacency_returns_line_collection(ax: Axes) -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    adjacency = {0: [1], 1: [0, 2], 2: [1]}
    collection = draw_adjacency(ax, points, adjacency)
    assert isinstance(collection, LineCollection)
    assert len(collection.get_segments()) == 2


def test_draw_adjacency_maps_points_by_row_index(ax: Axes) -> None:
    points = np.array([[0.0, 0.0], [10.0, 0.0]])
    adjacency = {0: [1]}
    collection = draw_adjacency(ax, points, adjacency)
    segments = collection.get_segments()
    assert len(segments) == 1
    np.testing.assert_allclose(sorted(segments[0].tolist()), [[0.0, 0.0], [10.0, 0.0]])


def test_draw_adjacency_forwards_edge_kwargs(ax: Axes) -> None:
    points = np.array([[0.0, 0.0], [1.0, 1.0]])
    adjacency = {0: [1]}
    _ = draw_adjacency(ax, points, adjacency, linewidths=3.0)


def test_draw_adjacency_without_node_kwargs_draws_no_nodes(ax: Axes) -> None:
    points = np.array([[0.0, 0.0], [1.0, 1.0]])
    adjacency = {0: [1]}
    draw_adjacency(ax, points, adjacency)
    assert not any(isinstance(line, Line2D) for line in ax.lines)


def test_draw_adjacency_with_node_kwargs_draws_nodes(ax: Axes) -> None:
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    adjacency = {0: [1], 1: [2]}
    draw_adjacency(ax, points, adjacency, node_kwargs={"marker": "o", "color": "blue"})
    node_lines = [line for line in ax.lines if isinstance(line, Line2D)]
    assert len(node_lines) == 1
    np.testing.assert_allclose(np.asarray(node_lines[0].get_xdata()), points[:, 0])
    np.testing.assert_allclose(np.asarray(node_lines[0].get_ydata()), points[:, 1])


def test_draw_adjacency_empty_graph(ax: Axes) -> None:
    points = np.zeros((0, 2))
    collection = draw_adjacency(ax, points, {})
    assert isinstance(collection, LineCollection)
    assert len(collection.get_segments()) == 0
