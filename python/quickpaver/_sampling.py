# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Variable-radius Poisson disk sampling.

Based on Bridson, R. (2007), *Fast Poisson Disk Sampling in Arbitrary
Dimensions*, with spatially varying radii (original adaptation: Adrian
Bittner, 2021). The sampler runs in Rust with the GIL released.
"""

from __future__ import annotations

import operator
from typing import Optional, SupportsIndex

import numpy as np

from quickpaver import _quickpaver as _rs
from quickpaver._types import ArrayLike, NDArrayFloat


def poisson_disk_sampling(
    radius: ArrayLike,
    k: SupportsIndex = 30,
    seed: Optional[SupportsIndex] = None,
) -> tuple[int, NDArrayFloat]:
    """Variable-density Poisson disk samples over a 2D pixel domain.

    The domain is ``[0, width) x [0, height)`` in pixel units, with
    ``(height, width) = radius.shape``. Every pair of returned points ``i``,
    ``j`` satisfies ``|p_i - p_j| >= radius[i] + radius[j]``, where the radius
    of a point is read at its own pixel ``radius[int(y), int(x)]``. This gives
    a blue-noise distribution whose local density follows the radius field:
    small radii pack points densely, large radii push them apart.

    The sampler follows Bridson's algorithm: candidates are drawn on the
    circle of radius ``2 r_p`` around a random active point ``p`` (``r_p +
    r_c`` where the radius grows away from it, ``r_c`` being the radius at the
    candidate), and a point is retired after ``k`` rejected candidates. When
    no active point is left, the first pixel centre (in raster order) where a
    point still fits seeds a new growth. The result is maximal: no point can be
    added anywhere, including across sharp jumps of the radius that growth
    alone cannot cross.

    Parameters
    ----------
    radius : ArrayLike
        2D radius map, shape ``(height, width)``, in pixels. Every value must
        be finite and strictly positive. Converted to a C-contiguous float32
        array.
    k : int, default 30
        Candidates tried around an active point before it is retired. Lower
        values are faster but leave more of the domain to the raster fill
        pass, whose points are less isotropic (more regular rows) than the
        blue noise of Bridson's growth.
    seed : int, optional
        Seed in ``[0, 2**64)`` of the xoshiro256++ generator (expanded with
        SplitMix64). The same seed and radius map give the same points on
        every platform (integer random numbers and basic IEEE arithmetic
        only: no libm call). By default, a random seed is drawn and the result is
        not reproducible.

    Returns
    -------
    n_particles : int
        Number of points.
    coordinates : NDArrayFloat
        Array of shape ``(n_particles, 2)``, dtype float32, with
        ``coordinates[:, 0] = x`` (column axis) and ``coordinates[:, 1] = y``
        (row axis), in generation order. The spacing holds for the float64
        positions; the float32 storage can shift it by ~1e-7 relative.

    Raises
    ------
    ValueError
        If ``radius`` is not a non-empty 2D array of finite positive values,
        if ``k < 1`` or if ``seed`` is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> radius = np.full((50, 80), 3.0)
    >>> n, xy = poisson_disk_sampling(radius, seed=0)
    >>> xy.shape == (n, 2) and xy.dtype == np.float32
    True
    >>> d = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    >>> bool(d[np.triu_indices(n, 1)].min() >= 6.0 - 1e-4)
    True
    """
    r = np.asarray(radius)
    if r.ndim != 2:
        raise ValueError("radius must be 2D")
    r = np.ascontiguousarray(r, dtype=np.float32)
    n_try = operator.index(k)
    if n_try < 1:
        raise ValueError(f"k must be >= 1, got {n_try}.")
    if seed is None:
        s = int(np.random.default_rng().integers(0, 2**63 - 1))
    else:
        s = operator.index(seed)
    if not 0 <= s < 2**64:
        raise ValueError(f"seed must be in [0, 2**64), got {s}.")
    xy = _rs.poisson_disk_sampling(r, n_try, s)
    return len(xy), xy
