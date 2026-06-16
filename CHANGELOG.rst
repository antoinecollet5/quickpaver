==============
Changelog
==============

0.3.1 (2026-06-16)
------------------

* DOCS: improved ``gen_polygon`` docstring.
* FIX: when no ``alignment_point``is provided, the grid positioning is consistent with
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
