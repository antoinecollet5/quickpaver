# SPDX-License-Identifier: BSD-3-Clause
"""Tests of the variable-radius Poisson disk sampling."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from scipy.spatial import KDTree

import quickpaver as qp


def _radius_at(radius: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = radius.shape
    i = np.minimum(h - 1, xy[:, 1].astype(int))
    j = np.minimum(w - 1, xy[:, 0].astype(int))
    return radius.astype(np.float32)[i, j].astype(float)


def _min_spacing_ratio(radius: np.ndarray, xy: np.ndarray) -> float:
    """``min |p_i - p_j| / (r_i + r_j)`` over all pairs."""
    xy = xy.astype(float)
    r = _radius_at(radius, xy)
    pairs = KDTree(xy).query_pairs(2 * r.max(), output_type="ndarray")
    if not len(pairs):
        return np.inf
    d = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    return float((d / (r[pairs[:, 0]] + r[pairs[:, 1]])).min())


def _uncovered_fraction(radius: np.ndarray, xy: np.ndarray, n: int = 20000) -> float:
    """Fraction of random probes that could still be added to the set."""
    xy = xy.astype(float)
    r = _radius_at(radius, xy)
    h, w = radius.shape
    q = np.random.default_rng(0).uniform([0, 0], [w, h], (n, 2))
    rq = _radius_at(radius, q)
    tree = KDTree(xy)
    free = 0
    for i, js in enumerate(tree.query_ball_point(q, rq + r.max())):
        js = np.asarray(js, dtype=int)
        d = np.linalg.norm(xy[js] - q[i], axis=1)
        free += not np.any(d < rq[i] + r[js])
    return free / n


def _fields() -> dict:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:120, 0:180]
    return {
        "uniform": np.full((120, 180), 3.0),
        "gradient": np.tile(np.linspace(1.5, 12.0, 180), (120, 1)),
        "radial": 1.5 + 0.08 * np.hypot(xx - 60, yy - 70),
        "step": np.where(xx < 90, 1.2, 15.0),
        "noise": rng.uniform(1.0, 6.0, (120, 180)),
    }


@pytest.mark.parametrize("name", list(_fields()))
def test_spacing_and_coverage(name: str) -> None:
    radius = _fields()[name]
    n, xy = qp.poisson_disk_sampling(radius, seed=3)
    assert xy.shape == (n, 2) and xy.dtype == np.float32 and n > 10
    h, w = radius.shape
    assert np.all((xy >= 0) & (xy <= [w, h]))
    # |p_i - p_j| >= r_i + r_j, up to the float32 rounding of the coordinates
    assert _min_spacing_ratio(radius, xy) >= 1.0 - 1e-5
    # maximal set: no region is left where a point would still fit (also
    # across radius jumps, where growth alone cannot go)
    assert _uncovered_fraction(radius, xy) < 0.005


def test_golden_outputs() -> None:
    """Same seed -> same points on every platform (pure integer RNG and
    basic IEEE arithmetic only)."""
    expected = {
        "uniform": (185, "dcc649a2fbad2872"),
        "gradient": (130, "a51faad0fa51112b"),
    }
    fields = {
        "uniform": np.full((60, 90), 2.5),
        "gradient": np.tile(np.linspace(1.5, 6.0, 90), (60, 1)),
    }
    for name, radius in fields.items():
        n, xy = qp.poisson_disk_sampling(radius, seed=42)
        digest = hashlib.sha256(xy.tobytes()).hexdigest()[:16]
        assert (n, digest) == expected[name], name


def test_seeds() -> None:
    radius = np.full((50, 70), 2.0)
    a = qp.poisson_disk_sampling(radius, seed=7)[1]
    np.testing.assert_array_equal(a, qp.poisson_disk_sampling(radius, seed=7)[1])
    np.testing.assert_array_equal(
        a, qp.poisson_disk_sampling(radius, seed=np.uint64(7))[1]
    )
    assert not np.array_equal(a, qp.poisson_disk_sampling(radius, seed=8)[1])
    for seed in (0, 2**64 - 1):
        assert qp.poisson_disk_sampling(radius, seed=seed)[0] > 0
    n, xy = qp.poisson_disk_sampling(radius)  # random seed
    assert n == len(xy) > 0


def test_input_conversion() -> None:
    radius = np.full((40, 60), 2.0)
    ref = qp.poisson_disk_sampling(radius, seed=1)[1]
    for r in (radius.astype(np.float32), radius.astype(int), radius.tolist()):
        np.testing.assert_array_equal(qp.poisson_disk_sampling(r, seed=1)[1], ref)
    # Fortran-ordered / strided input is made contiguous
    big = np.full((80, 120), 2.0)
    np.testing.assert_array_equal(
        qp.poisson_disk_sampling(big[::2, ::2], seed=1)[1], ref
    )
    np.testing.assert_array_equal(
        qp.poisson_disk_sampling(np.asfortranarray(radius), seed=1)[1], ref
    )


@pytest.mark.parametrize("k", [1, 5, 100])
def test_any_k_gives_a_maximal_set(k: int) -> None:
    """Whatever ``k``, the fill pass leaves no room for another point."""
    radius = _fields()["radial"]
    n, xy = qp.poisson_disk_sampling(radius, k=k, seed=0)
    assert _min_spacing_ratio(radius, xy) >= 1.0 - 1e-5
    assert _uncovered_fraction(radius, xy) < 0.005


def test_small_domains() -> None:
    assert qp.poisson_disk_sampling(np.ones((1, 1)), seed=0)[0] == 1
    # a radius larger than the domain leaves room for one point only
    assert qp.poisson_disk_sampling(np.full((20, 30), 100.0), seed=0)[0] == 1
    n, xy = qp.poisson_disk_sampling(np.full((1, 50), 1.0), seed=0)
    assert n > 1 and np.all(xy[:, 1] < 1)


@pytest.mark.parametrize(
    "radius, match",
    [
        (np.ones(5), "2D"),
        (np.ones((2, 2, 2)), "2D"),
        (np.zeros((0, 3)), "non-empty"),
        (np.array([[1.0, 0.0]]), "> 0"),
        (np.array([[1.0, -2.0]]), "> 0"),
        (np.array([[1.0, np.nan]]), "> 0"),
        (np.array([[1.0, np.inf]]), "> 0"),
    ],
)
def test_bad_radius(radius, match) -> None:
    with pytest.raises(ValueError, match=match):
        qp.poisson_disk_sampling(radius, seed=0)


def test_bad_arguments() -> None:
    radius = np.ones((5, 5))
    with pytest.raises(ValueError, match="k must be"):
        qp.poisson_disk_sampling(radius, k=0)
    with pytest.raises(ValueError, match="seed"):
        qp.poisson_disk_sampling(radius, seed=-1)
    with pytest.raises(ValueError, match="seed"):
        qp.poisson_disk_sampling(radius, seed=2**64)
    with pytest.raises(TypeError):
        qp.poisson_disk_sampling(radius, seed=1.5)  # ty: ignore[invalid-argument-type]
