==============
Changelog
==============

0.7.0 (2026-11-26)
------------------

Performance (no API change; results are bit-identical to 0.6.0 except where noted):

* ENH: the geometry kernels are now written in Rust (PyO3 + rayon, GIL released) and
  replace the numba kernels and the numpy fallbacks, which are removed. numba is no longer
  used: no JIT compilation at first call (5.5 s before, 20 ms now) and no ``numba`` extra.
* ENH: every transfer path runs its candidate search, masks, pre-filter, clip, area filter
  and weights in a single parallel Rust pass. ``compute_transfer_matrix`` is 2x to 36x
  faster depending on the grids, and the intersection geometry 2x to 9x faster with
  ~40% less memory.
* ENH: cells lying strictly inside a convex polygon (or rotated cell) get their area
  analytically, without clipping (large gains when the resolutions differ a lot).
* ENH: concave and holed polygons are clipped in Rust against convex windows (grid cells,
  triangles): GEOS is used only for their intersection geometry, and for pairs of two
  non-convex polygons. Polygon / polygon weights may differ from 0.6.0 by ~1e-13
  (Sutherland-Hodgman instead of GEOS).
* ENH: grid-hash bounding-box join instead of STRtree; ``TriMesh.to_shapely`` vectorized;
  sparse matrices built directly in CSC form.
* ENH: tiling: tiles clearly inside / outside the surface are classified in Rust, so the
  GEOS ``intersects`` predicate (and the polygon construction) only runs near the
  boundary; rotated tilings are rotated once, before building any geometry
  (``gen_polygonal_tiling`` 2x to 3x faster, ``intersects_mask`` faster too).

New (additive) API:

* ENH: ``compute_transfer_matrix_with_intersection_vertices`` and ``IntersectionVertices``:
  the intersection geometry as ragged NumPy arrays, without creating Shapely objects
  (``IntersectionVertices.to_shapely()`` converts them).
* ENH: ``gen_polygonal_tiling_arrays``: tile vertices and CSR adjacency as NumPy arrays.
* ENH: polygon grids may also be given as a sequence / array of ``shapely.Polygon``.

Fixes:

* FIX: the rectilinear / polygon transfer raised ``ValueError`` for convex polygons with
  more than 28 vertices (e.g. disk-clipped tiles); any polygon size is now supported.
* FIX: ``import quickpaver`` failed when matplotlib was not installed although it was not
  a declared dependency: matplotlib is now imported lazily and declared as a dependency.

Packaging:

* DOCS: the example notebooks are marimo notebooks only (the Jupyter copy is removed).
  They are executed and exported to HTML with ``marimo export html`` while building the
  documentation, which no longer needs nbsphinx, Jupyter or pandoc. Fixed the notebooks
  (``mo`` was used but never imported).
* STYLE: upgrade to ``ruff==0.17.6`` and ``ty==0.0.111``.
* BUILD: maturin mixed Rust / Python project (``python/`` + ``src/``); the version lives in
  ``Cargo.toml``. Wheels for CPython 3.10 to 3.15, including the free-threaded 3.14t and
  3.15t builds (the extension does not need the GIL), on Linux (glibc, musl), macOS and
  Windows; release workflow with wheel tests on every interpreter; ``uv`` based dev setup.

0.6.0 (2026-10-06)
------------------

* ENH: add ``extract_tiling_edge_centers`` to extract the centres of the polygon
  edges, deduplicating the edges shared between adjacent polygons. It also returns
  the adjacency between edges and polygons, and between polygons and edges (in ring
  order, so that opposite edges of a rectangle are easy to retrieve).
* FIX: ``extract_tiling_vertices`` (and therefore ``adjacency_by_shared_vertices``)
  could fail to merge coincident vertices when a coordinate fell on a rounding tie
  (e.g. ``1.035`` scaled to ``103.5``), because float noise sent copies of the same
  corner to different buckets. Vertices are now merged by gap clustering, which has
  no rounding boundary.
* FIX: the vertex-to-polygon adjacency returned by ``extract_tiling_vertices`` was
  wrong (some vertices had no polygon) on tilings with more polygons than vertices,
  such as triangular tilings, due to colliding keys when grouping the pairs.
* ENH: ``n_decimals`` in ``extract_tiling_vertices`` is now the merge tolerance
  (``10**-n_decimals``); the returned coordinates are still rounded to ``n_decimals``.
* TEST: add tests for ``extract_tiling_edge_centers`` and for the vertex clustering
  (rounding ties, float noise, large coordinates), reaching 100% coverage of the
  tiling module.

0.5.0 (2026-09-27)
------------------

* ENH: generalize ``get_array_borders_selection`` to any dim.
* ENH: add methods to RectilinearGrid: ``gc_face_area_m2``, ``pipj_m``, ``get_slicer_forward`` and ``get_slicer_backward``.
* STYLE: upgrade to `ty==0.0.84`.


0.4.0 (2026-08-25)
------------------

* ENH: implement ``TriMesh`` for triangular mesh support, with a conservative
  transfer matrix implementation.
* ENH: add ``compute_transfer_matrix_with_intersections`` to compute transfer
  matrices while also returning the underlying geometric intersections, with
  support for masks.
* ENH: implement a conservative transfer matrix for rectilinear grids, backed
  by numpy/numba for improved performance.
* ENH: add ``Disk`` polygon support in ``gen_polygonal_tiling``, and optimize
  the multipolygon creation.
* ENH: add ``intersects_mask`` utility.
* ENH: add adjacency and plotting utilities:

  * ``adjacency_by_shared_vertices``
  * ``adjacency_to_edges``
  * ``draw_adjacency``

* ENH: ``extract_tiling_vertices`` now also returns the adjacency between
  polygons and vertices.
* ENH: add a ``contour`` property to ``RectilinearGrid``.
* ENH: significant performance improvements for ``RectilinearGrid`` and
  ``tilling.py``, including "F"-order transfer matrix computation on
  rectilinear grids.
* FIX: fix the adjacency computation in the tiling module.
* TEST: add extensive test coverage for the transfer matrix, ``TriMesh``,
  and plotting utilities.
* Update CI/tooling to Python 3.14 and ``ty==0.0.74``.

0.3.1 (2026-06-16)
------------------

* DOCS: improved ``gen_polygon`` docstring.
* FIX: when no ``alignment_point`` is provided, the grid positioning is consistent with
  v0.2.x for retro-compatibility.
* TEST: updated github actions in CI.


0.3.0 (2026-06-03)
------------------

* ENH: add ``compute_transfer_matrix`` to build conservative sparse transfer
* ENH: add add regular-grid indexing helpers:

  * ``rlg_idx_to_nn``
  * ``rlg_nn_to_idx``
  * ``span_to_node_numbers_2d``
  * ``span_to_node_numbers_3d``

* ENH: add sparse spatial operators for rectilinear grids:

  * ``get_rlg_spatial_grad_mat``
  * ``make_rlg_spatial_gradient_matrices``
  * ``get_rlg_perm_mat``
  * ``make_rlg_spatial_permutation_matrices``

* ENH: add polygon-to-grid selection utilities:

  * ``create_selections_array_2d``
  * ``get_polygon_selection_with_dilation_2d``
  * ``get_polygon_selection_with_dilation_3d``
  * ``binary_dilation``

* ENH: add grid resampling and array upsampling utilities:

  * ``resample_grid``
  * ``duplicative_upsample``
  * ``conservative_upsample``

* ENH: add export helpers for ``RectilinearGrid``:

  * ``to_shapely`` for 2D Shapely polygon export.
  * ``to_pyvista`` for optional PyVista export using ``ImageData``,
    ``RectilinearGrid`` or ``StructuredGrid`` representations.

* ENH: add grid geometry helpers such as ``get_array_borders_selection`` and
  owner/neighbour index pairing utilities.
* ENH: add input validation for rectilinear grid dimensions, cell sizes, sparse
  operator axes, dilation masks, and upsampling factors.
* TEST: add extensive pytest coverage for rectilinear grids, sparse operators,
  polygon selections, dilation, resampling, upsampling, Shapely export, and
  optional PyVista export.
* DOC: improve tiling documentation.
* DOC: add documentation for rectilinear grid utilities.
* DOC: update notebooks for tiling, transfer matrices, and grid utilities.

0.2.0 (2026-03-18)
------------------

* ENH: ``extract_tiling_vertices`` now returns cluster indices for merged vertices.
* DOC: improve documentation.

0.1.0 (2026-03-14)
------------------

* First release on PyPI.
