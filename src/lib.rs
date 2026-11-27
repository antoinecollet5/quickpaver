//! Rust core of quickpaver: conservative transfer matrices between polygonal
//! grids (rectilinear grids, triangular meshes, arbitrary polygons), cell
//! quality metrics, variable-radius Poisson disk sampling and VTK meshes.
//!
//! The Python-facing API lives at the bottom of this file; the kernels in
//! [`transfer`] are plain Rust so that they can be unit-tested without an
//! interpreter. All heavy work runs with the GIL released (free-threading
//! ready), parallelised with rayon.

pub mod clip;
pub mod geom;
pub mod mesh;
pub mod quality;
pub mod sampling;
pub mod surface;
pub mod transfer;

use geom::{Acc, Polys, ToLocal, ToWorld};
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use transfer::{Grid, NonSep, RectOpts};

type F1<'py> = Bound<'py, PyArray1<f64>>;
type F2<'py> = Bound<'py, PyArray2<f64>>;
type I1<'py> = Bound<'py, PyArray1<isize>>;
type U1<'py> = Bound<'py, PyArray1<u8>>;
/// `(a, b, value, kind, ring coords (n, 2), ring length per entry)`
type Out<'py> = (I1<'py>, I1<'py>, F1<'py>, U1<'py>, F2<'py>, I1<'py>);
type Frame = (f64, f64, f64, f64);
/// `(legacy cells, cell types, 2D cell areas)`
type Extruded3d<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<u8>>, F1<'py>);
type Csc<'py> = (
    F1<'py>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
);

fn s1<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'_, T>) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("array must be contiguous"))
}
fn opt<'a, T: numpy::Element>(a: &'a Option<PyReadonlyArray1<'_, T>>) -> PyResult<Option<&'a [T]>> {
    a.as_ref().map(s1).transpose()
}
fn check_mask(m: Option<&[bool]>, n: usize, what: &str) -> PyResult<()> {
    match m {
        Some(m) if m.len() != n => Err(PyValueError::new_err(format!(
            "{what} has length {}, expected {n}",
            m.len()
        ))),
        _ => Ok(()),
    }
}

fn polys(
    x: &PyReadonlyArray1<'_, f64>,
    y: &PyReadonlyArray1<'_, f64>,
    ring_off: &PyReadonlyArray1<'_, isize>,
    poly_ring: &PyReadonlyArray1<'_, isize>,
) -> PyResult<Polys> {
    Polys::from_slices(s1(x)?, s1(y)?, s1(ring_off)?, s1(poly_ring)?).map_err(PyValueError::new_err)
}

fn out(py: Python<'_>, acc: Acc) -> Out<'_> {
    let nc = acc.coords.len() / 2;
    (
        acc.a.into_pyarray(py),
        acc.b.into_pyarray(py),
        acc.w.into_pyarray(py),
        acc.kind.into_pyarray(py),
        Array2::from_shape_vec((nc, 2), acc.coords)
            .expect("interleaved coordinates")
            .into_pyarray(py),
        acc.ring_len.into_pyarray(py),
    )
}

fn to_world(f: Option<Frame>) -> Option<ToWorld> {
    f.map(|(cx, cy, ca, sa)| ToWorld { cx, cy, ca, sa })
}

/// Rectilinear grid vs a polygon set (ragged, world frame), clipped in the
/// grid's local frame. Returns item-major entries `(item, cell, area, ...)`.
#[pyfunction]
#[pyo3(signature = (x, y, ring_off, poly_ring, convex, to_local, x_edges, y_edges,
                    cell_mask=None, item_mask=None, with_geom=false, to_world=None, interior=true))]
#[allow(clippy::too_many_arguments)]
fn rect_items_transfer<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    ring_off: PyReadonlyArray1<'py, isize>,
    poly_ring: PyReadonlyArray1<'py, isize>,
    convex: Option<PyReadonlyArray1<'py, bool>>,
    to_local: Frame,
    x_edges: PyReadonlyArray1<'py, f64>,
    y_edges: PyReadonlyArray1<'py, f64>,
    cell_mask: Option<PyReadonlyArray1<'py, bool>>,
    item_mask: Option<PyReadonlyArray1<'py, bool>>,
    with_geom: bool,
    to_world: Option<Frame>,
    interior: bool,
) -> PyResult<Out<'py>> {
    let p = polys(&x, &y, &ring_off, &poly_ring)?;
    let cv = opt(&convex)?;
    let (xe, ye) = (s1(&x_edges)?, s1(&y_edges)?);
    if xe.len() < 2 || ye.len() < 2 {
        return Err(PyValueError::new_err("a grid needs at least one cell"));
    }
    let (cm, im) = (opt(&cell_mask)?, opt(&item_mask)?);
    check_mask(cm, (xe.len() - 1) * (ye.len() - 1), "cell_mask")?;
    check_mask(im, p.n(), "item_mask")?;
    check_mask(cv, p.n(), "convex")?;
    let acc = py.detach(|| {
        let cvx: Vec<bool> = match cv {
            Some(c) => c.to_vec(),
            None => p.convexity(), // world frame, like the reference
        };
        let (cx, cy, ca, sa) = to_local;
        let local = p.to_local(&ToLocal { cx, cy, ca, sa });
        let o = RectOpts {
            cell_mask: cm,
            with_geom,
            to_world: self::to_world(to_world),
            interior,
        };
        transfer::rect_items(&local, &cvx, &Grid { xe, ye }, im, &o)
    });
    Ok(out(py, acc))
}

/// Non-separable rectilinear/rectilinear transfer (rotated target cells).
/// Returns target-major `(source cell, target cell, weight, ...)`.
#[pyfunction]
#[pyo3(signature = (snx, sny, sdx, sdy, tnx, tny, tdx, tdy, origin_x, origin_y, cr, sr,
                    template_x, template_y, half_bx, half_by, src_mask=None, tgt_mask=None,
                    with_geom=false, to_world=None, interior=true))]
#[allow(clippy::too_many_arguments)]
fn nonseparable_transfer<'py>(
    py: Python<'py>,
    snx: usize,
    sny: usize,
    sdx: f64,
    sdy: f64,
    tnx: usize,
    tny: usize,
    tdx: f64,
    tdy: f64,
    origin_x: f64,
    origin_y: f64,
    cr: f64,
    sr: f64,
    template_x: [f64; 4],
    template_y: [f64; 4],
    half_bx: f64,
    half_by: f64,
    src_mask: Option<PyReadonlyArray1<'py, bool>>,
    tgt_mask: Option<PyReadonlyArray1<'py, bool>>,
    with_geom: bool,
    to_world: Option<Frame>,
    interior: bool,
) -> PyResult<Out<'py>> {
    let (sm, tm) = (opt(&src_mask)?, opt(&tgt_mask)?);
    check_mask(sm, snx * sny, "src_mask")?;
    check_mask(tm, tnx * tny, "tgt_mask")?;
    let q = NonSep {
        snx,
        sny,
        sdx,
        sdy,
        tnx,
        tny,
        tdx,
        tdy,
        origin_x,
        origin_y,
        cr,
        sr,
        template_x,
        template_y,
        half_bx,
        half_by,
    };
    let acc = py.detach(|| {
        let o = RectOpts {
            cell_mask: sm,
            with_geom,
            to_world: self::to_world(to_world),
            interior,
        };
        transfer::nonseparable(&q, tm, &o)
    });
    Ok(out(py, acc))
}

/// Separable rectilinear/rectilinear transfer from the two 1-D overlap lists.
#[pyfunction]
#[pyo3(signature = (is_x, it_x, ox, xlo, xhi, is_y, it_y, oy, ylo, yhi, sx_is_tgt_y,
                    snx, sny, tnx, tny, src_area, src_mask=None, tgt_mask=None,
                    with_geom=false, to_world=None))]
#[allow(clippy::too_many_arguments)]
fn separable_transfer<'py>(
    py: Python<'py>,
    is_x: PyReadonlyArray1<'py, isize>,
    it_x: PyReadonlyArray1<'py, isize>,
    ox: PyReadonlyArray1<'py, f64>,
    xlo: PyReadonlyArray1<'py, f64>,
    xhi: PyReadonlyArray1<'py, f64>,
    is_y: PyReadonlyArray1<'py, isize>,
    it_y: PyReadonlyArray1<'py, isize>,
    oy: PyReadonlyArray1<'py, f64>,
    ylo: PyReadonlyArray1<'py, f64>,
    yhi: PyReadonlyArray1<'py, f64>,
    sx_is_tgt_y: bool,
    snx: usize,
    sny: usize,
    tnx: usize,
    tny: usize,
    src_area: f64,
    src_mask: Option<PyReadonlyArray1<'py, bool>>,
    tgt_mask: Option<PyReadonlyArray1<'py, bool>>,
    with_geom: bool,
    to_world: Option<Frame>,
) -> PyResult<Out<'py>> {
    let (sm, tm) = (opt(&src_mask)?, opt(&tgt_mask)?);
    check_mask(sm, snx * sny, "src_mask")?;
    check_mask(tm, tnx * tny, "tgt_mask")?;
    let (a1, a2, a3, a4, a5) = (s1(&is_x)?, s1(&it_x)?, s1(&ox)?, s1(&xlo)?, s1(&xhi)?);
    let (b1, b2, b3, b4, b5) = (s1(&is_y)?, s1(&it_y)?, s1(&oy)?, s1(&ylo)?, s1(&yhi)?);
    let n_ok = [a2.len(), a3.len(), a4.len(), a5.len()]
        .iter()
        .all(|&l| l == a1.len())
        && [b2.len(), b3.len(), b4.len(), b5.len()]
            .iter()
            .all(|&l| l == b1.len());
    if !n_ok {
        return Err(PyValueError::new_err(
            "overlap arrays have inconsistent lengths",
        ));
    }
    let acc = py.detach(|| {
        transfer::separable(
            a1,
            a2,
            a3,
            a4,
            a5,
            b1,
            b2,
            b3,
            b4,
            b5,
            sx_is_tgt_y,
            snx,
            tnx,
            src_area,
            sm,
            tm,
            with_geom,
            self::to_world(to_world),
        )
    });
    Ok(out(py, acc))
}

/// Two irregular polygon sets (ragged, world frame): grid-hash bbox join and
/// convex-window clips. Returns b-major `(a, b, area, kind, ...)`; entries of
/// kind 2 (no convex side) carry `NaN` areas for the GEOS fallback.
#[pyfunction]
#[pyo3(signature = (ax, ay, a_ring_off, a_poly_ring, a_convex, bx, by, b_ring_off, b_poly_ring,
                    b_convex, a_mask=None, b_mask=None, prefer_a_window=true, with_geom=false))]
#[allow(clippy::too_many_arguments)]
fn join_clip_transfer<'py>(
    py: Python<'py>,
    ax: PyReadonlyArray1<'py, f64>,
    ay: PyReadonlyArray1<'py, f64>,
    a_ring_off: PyReadonlyArray1<'py, isize>,
    a_poly_ring: PyReadonlyArray1<'py, isize>,
    a_convex: Option<PyReadonlyArray1<'py, bool>>,
    bx: PyReadonlyArray1<'py, f64>,
    by: PyReadonlyArray1<'py, f64>,
    b_ring_off: PyReadonlyArray1<'py, isize>,
    b_poly_ring: PyReadonlyArray1<'py, isize>,
    b_convex: Option<PyReadonlyArray1<'py, bool>>,
    a_mask: Option<PyReadonlyArray1<'py, bool>>,
    b_mask: Option<PyReadonlyArray1<'py, bool>>,
    prefer_a_window: bool,
    with_geom: bool,
) -> PyResult<Out<'py>> {
    let pa = polys(&ax, &ay, &a_ring_off, &a_poly_ring)?;
    let pb = polys(&bx, &by, &b_ring_off, &b_poly_ring)?;
    let (ca, cb) = (opt(&a_convex)?, opt(&b_convex)?);
    let (am, bm) = (opt(&a_mask)?, opt(&b_mask)?);
    check_mask(ca, pa.n(), "a_convex")?;
    check_mask(cb, pb.n(), "b_convex")?;
    check_mask(am, pa.n(), "a_mask")?;
    check_mask(bm, pb.n(), "b_mask")?;
    let acc = py.detach(|| {
        let ca: Vec<bool> = ca.map_or_else(|| pa.convexity(), |c| c.to_vec());
        let cb: Vec<bool> = cb.map_or_else(|| pb.convexity(), |c| c.to_vec());
        transfer::join_clip(&pa, &ca, &pb, &cb, am, bm, prefer_a_window, with_geom)
    });
    Ok(out(py, acc))
}

/// Convexity of every polygon of a ragged set (exterior ring, no holes).
#[pyfunction]
fn polygon_convexity<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    ring_off: PyReadonlyArray1<'py, isize>,
    poly_ring: PyReadonlyArray1<'py, isize>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let p = polys(&x, &y, &ring_off, &poly_ring)?;
    Ok(py.detach(|| p.convexity()).into_pyarray(py))
}

/// Classify boxes `(n, 4)` `[xmin, ymin, xmax, ymax]` against a polygonal
/// surface (ragged rings, all rings used with the even-odd rule): 1 inside,
/// 0 outside, -1 ambiguous (touches the boundary neighbourhood: use GEOS).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn classify_boxes<'py>(
    py: Python<'py>,
    xmin: PyReadonlyArray1<'py, f64>,
    ymin: PyReadonlyArray1<'py, f64>,
    xmax: PyReadonlyArray1<'py, f64>,
    ymax: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    ring_off: PyReadonlyArray1<'py, isize>,
    poly_ring: PyReadonlyArray1<'py, isize>,
    cell: f64,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let s = polys(&x, &y, &ring_off, &poly_ring)?;
    let (a, b, c, d) = (s1(&xmin)?, s1(&ymin)?, s1(&xmax)?, s1(&ymax)?);
    if b.len() != a.len() || c.len() != a.len() || d.len() != a.len() {
        return Err(PyValueError::new_err("box arrays differ in length"));
    }
    if !(cell > 0.0 && cell.is_finite()) {
        return Err(PyValueError::new_err("cell must be positive"));
    }
    let out: Vec<i8> = py.detach(|| {
        if s.n() == 0 {
            return vec![surface::OUTSIDE; a.len()];
        }
        let g = surface::SurfaceGrid::build(&s, cell);
        (0..a.len())
            .into_par_iter()
            .with_min_len(4096)
            .map(|k| g.classify([a[k], b[k], c[k], d[k]]))
            .collect()
    });
    Ok(out.into_pyarray(py))
}

/// Point-in-polygon test, same results as matplotlib's
/// `Path(vertices).contains_points(points)` (see `surface::points_in_polygon`).
#[pyfunction]
fn points_in_polygon<'py>(
    py: Python<'py>,
    px: PyReadonlyArray1<'py, f64>,
    py_: PyReadonlyArray1<'py, f64>,
    vx: PyReadonlyArray1<'py, f64>,
    vy: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let (a, b, c, d) = (s1(&px)?, s1(&py_)?, s1(&vx)?, s1(&vy)?);
    if a.len() != b.len() || c.len() != d.len() {
        return Err(PyValueError::new_err("x / y arrays differ in length"));
    }
    let out = py.detach(|| surface::points_in_polygon(a, b, c, d));
    Ok(out.into_pyarray(py))
}

/// COO -> CSC arrays `(data, indices, indptr)` for *unique* (row, col) pairs.
#[pyfunction]
fn coo_to_csc<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<'py, isize>,
    cols: PyReadonlyArray1<'py, isize>,
    data: PyReadonlyArray1<'py, f64>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<Csc<'py>> {
    let (r, c, d) = (s1(&rows)?, s1(&cols)?, s1(&data)?);
    if r.len() != c.len() || r.len() != d.len() {
        return Err(PyValueError::new_err(
            "rows, cols and data differ in length",
        ));
    }
    if r.len() >= i32::MAX as usize || n_rows >= i32::MAX as usize {
        return Err(PyValueError::new_err("too large for int32 indices"));
    }
    if r.iter().any(|&x| x < 0 || x as usize >= n_rows)
        || c.iter().any(|&x| x < 0 || x as usize >= n_cols)
    {
        return Err(PyValueError::new_err("index out of bounds"));
    }
    let (od, oi, op) = py.detach(|| transfer::coo_to_csc(r, c, d, n_cols));
    Ok((
        od.into_pyarray(py),
        oi.into_pyarray(py),
        op.into_pyarray(py),
    ))
}

/// Roundness quality of every polygon of an interleaved vertex buffer
/// `[x0, y0, x1, y1, ...]` split by `offsets` (in vertices). With `closed`,
/// the last vertex of every ring is its closing duplicate and is skipped.
#[pyfunction]
#[pyo3(signature = (xy, offsets, closed, centers=None, parallel=true))]
fn cell_quality<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray1<'py, f64>,
    offsets: PyReadonlyArray1<'py, isize>,
    closed: bool,
    centers: Option<PyReadonlyArray1<'py, f64>>,
    parallel: bool,
) -> PyResult<F1<'py>> {
    let r = quality::Rings {
        xy: s1(&xy)?,
        off: s1(&offsets)?,
        closed,
    };
    r.validate().map_err(PyValueError::new_err)?;
    let c = opt(&centers)?;
    if let Some(c) = c {
        if c.len() != 2 * r.n() {
            return Err(PyValueError::new_err(format!(
                "centers must hold {} (x, y) pairs, got {} values",
                r.n(),
                c.len()
            )));
        }
    }
    let q = py.detach(|| quality::rings_quality(&r, c, parallel));
    Ok(q.into_pyarray(py))
}

/// Variable-radius Poisson disk sampling of a C-contiguous float32 radius
/// map `(h, w)`; returns the float32 `(n, 2)` points `(x, y)`.
#[pyfunction]
fn poisson_disk_sampling<'py>(
    py: Python<'py>,
    radius: PyReadonlyArray2<'py, f32>,
    k: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (h, w) = (radius.shape()[0], radius.shape()[1]);
    let r = radius
        .as_slice()
        .map_err(|_| PyValueError::new_err("radius must be C-contiguous"))?;
    let pts = py
        .detach(|| sampling::poisson_disk(r, h, w, k, seed))
        .map_err(PyValueError::new_err)?;
    let n = pts.len() / 2;
    Ok(Array2::from_shape_vec((n, 2), pts)
        .expect("interleaved points")
        .into_pyarray(py))
}

/// Polygons (interleaved vertices split by `offsets`, closing vertex
/// included if `closed`) to a VTK polygon mesh: vertices within `tol` are
/// merged (single linkage, cluster means), rings are made counter-clockwise.
/// Returns the merged points `(m, 2)` and the legacy cell array.
#[pyfunction]
fn polygon_mesh<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray1<'py, f64>,
    offsets: PyReadonlyArray1<'py, isize>,
    closed: bool,
    tol: f64,
) -> PyResult<(F2<'py>, Bound<'py, PyArray1<i64>>)> {
    let r = quality::Rings {
        xy: s1(&xy)?,
        off: s1(&offsets)?,
        closed,
    };
    r.validate().map_err(PyValueError::new_err)?;
    if !(tol >= 0.0 && tol.is_finite()) {
        return Err(PyValueError::new_err(
            "merge tolerance must be finite and >= 0",
        ));
    }
    let res = py.detach(|| {
        let mut open = Vec::with_capacity(r.xy.len());
        let mut off = Vec::with_capacity(r.n() + 1);
        off.push(0usize);
        for i in 0..r.n() {
            open.extend_from_slice(r.ring(i));
            off.push(open.len() / 2);
        }
        let (labels, merged) = mesh::merge_points(&open, tol);
        mesh::polygon_cells(&off, &labels, &merged).map(|c| (merged, c))
    });
    let (merged, cells) = res.map_err(|p| {
        PyValueError::new_err(format!(
            "polygon {p} has fewer than 3 distinct vertices after merging \
             (empty, degenerate, or merge tolerance too large)"
        ))
    })?;
    let m = merged.len() / 2;
    Ok((
        Array2::from_shape_vec((m, 2), merged)
            .expect("interleaved points")
            .into_pyarray(py),
        cells.into_pyarray(py),
    ))
}

/// Extrude the 2D cells of a surface (legacy `cells`, `celltypes`, points
/// `(n, 3)`) into `n_layers` layers. Returns the 3D legacy cell array, the
/// cell types and the horizontal area of every 2D cell.
#[pyfunction]
fn extrude_cells<'py>(
    py: Python<'py>,
    cells: PyReadonlyArray1<'py, i64>,
    celltypes: PyReadonlyArray1<'py, u8>,
    points: PyReadonlyArray2<'py, f64>,
    n_layers: usize,
    downward: bool,
) -> PyResult<Extruded3d<'py>> {
    let (c, t) = (s1(&cells)?, s1(&celltypes)?);
    if points.shape()[1] != 3 {
        return Err(PyValueError::new_err("points must have shape (n, 3)"));
    }
    let p = points
        .as_slice()
        .map_err(|_| PyValueError::new_err("points must be C-contiguous"))?;
    let e = py
        .detach(|| mesh::extrude(c, t, p, n_layers, downward))
        .map_err(PyValueError::new_err)?;
    Ok((
        e.cells.into_pyarray(py),
        e.types.into_pyarray(py),
        e.area.into_pyarray(py),
    ))
}

#[pymodule(gil_used = false)]
fn _quickpaver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rect_items_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(nonseparable_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(separable_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(join_clip_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(polygon_convexity, m)?)?;
    m.add_function(wrap_pyfunction!(coo_to_csc, m)?)?;
    m.add_function(wrap_pyfunction!(classify_boxes, m)?)?;
    m.add_function(wrap_pyfunction!(points_in_polygon, m)?)?;
    m.add_function(wrap_pyfunction!(cell_quality, m)?)?;
    m.add_function(wrap_pyfunction!(poisson_disk_sampling, m)?)?;
    m.add_function(wrap_pyfunction!(polygon_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(extrude_cells, m)?)?;
    Ok(())
}
