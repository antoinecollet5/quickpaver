//! Transfer-matrix kernels: every candidate search, mask, pre-filter, clip and
//! area filter of one transfer runs here, in parallel, without the GIL.

use crate::clip::{area, clip_convex, clip_rect, Buf, Poly, VecPoly};
use crate::geom::{Acc, Polys, ToWorld, KIND_AREA, KIND_GEOS};
use rayon::prelude::*;

pub const EPS_AREA: f64 = 1e-15;
/// Stack buffer size; convex subjects with more than `STACK - 4` vertices
/// use the growable buffer instead.
pub const STACK: usize = 40;

#[inline(always)]
pub fn clampi(v: f64, hi: isize) -> isize {
    // numpy: np.clip(np.floor(v).astype(intp), 0, hi)
    let f = v.floor();
    let i = if f.is_nan() { isize::MIN } else { f as isize };
    i.clamp(0, hi)
}

/// Strictly positive overlap (`NaN` -> false, like the numpy pre-filter).
#[inline(always)]
fn positive(v: f64) -> bool {
    v > 1e-15
}

#[inline(always)]
pub fn mask_ok(m: Option<&[bool]>, i: usize) -> bool {
    m.is_none_or(|m| m[i])
}

/// Per-thread scratch buffers (allocated once per worker, never re-zeroed).
struct Scratch {
    p: VecPoly,
    q: VecPoly,
    sp: Box<Poly<STACK>>,
    sq: Box<Poly<STACK>>,
}

impl Default for Scratch {
    fn default() -> Self {
        Scratch {
            p: VecPoly::default(),
            q: VecPoly::default(),
            sp: Box::new(Poly::new()),
            sq: Box::new(Poly::new()),
        }
    }
}

/// Minimum number of candidate cells per axis for the interior shortcut to
/// pay for its per-row span computation.
const INTERIOR_MIN_SPAN: usize = 4;

/// Grid description: local-frame edges.
pub struct Grid<'a> {
    pub xe: &'a [f64],
    pub ye: &'a [f64],
}

impl Grid<'_> {
    #[inline(always)]
    fn nx(&self) -> usize {
        self.xe.len() - 1
    }
    #[inline(always)]
    fn ny(&self) -> usize {
        self.ye.len() - 1
    }
}

/// x-extent `[L, R]` of a convex ring on the horizontal line `y`
/// (`None` if the line misses the ring).
#[inline(always)]
fn span_at(x: &[f64], y: &[f64], yl: f64) -> Option<(f64, f64)> {
    let n = x.len();
    let (mut l, mut r) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut j = n - 1;
    for i in 0..n {
        let (ya, yb) = (y[j], y[i]);
        if (ya <= yl && yl <= yb) || (yb <= yl && yl <= ya) {
            if ya == yb {
                l = l.min(x[j]).min(x[i]);
                r = r.max(x[j]).max(x[i]);
            } else {
                let xi = x[j] + (yl - ya) * (x[i] - x[j]) / (yb - ya);
                l = l.min(xi);
                r = r.max(xi);
            }
        }
        j = i;
    }
    if l <= r {
        Some((l, r))
    } else {
        None
    }
}

/// Cells of row strip `[y0, y1]` lying *strictly* inside a convex ring
/// (inclusive cell-index range), or `None`. For a convex ring the left border
/// is convex and the right one concave in `y`, so their extremes over the
/// strip are attained at the strip edges.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn interior_cells(
    x: &[f64],
    y: &[f64],
    bb: [f64; 4],
    y0: f64,
    y1: f64,
    xe: &[f64],
    i_lo: usize,
    i_hi: usize,
) -> Option<(usize, usize)> {
    let tol = 1e-11
        * (1.0
            + bb[0]
                .abs()
                .max(bb[2].abs())
                .max(bb[1].abs())
                .max(bb[3].abs()));
    if !(y0 > bb[1] + tol && y1 < bb[3] - tol) {
        return None;
    }
    let (l0, r0) = span_at(x, y, y0)?;
    let (l1, r1) = span_at(x, y, y1)?;
    let (lm, rm) = (l0.max(l1) + tol, r0.min(r1) - tol);
    // first cell with xe[i] >= lm, last cell with xe[i + 1] <= rm
    let mut a = i_lo;
    while a <= i_hi && xe[a] < lm {
        a += 1;
    }
    let mut b = i_hi as isize;
    while b >= a as isize && xe[b as usize + 1] > rm {
        b -= 1;
    }
    if b >= a as isize {
        Some((a, b as usize))
    } else {
        None
    }
}

/// Interior cell range of every row `j_lo..=j_hi` (out of line: only used
/// for polygons spanning many cells, keeps the hot loop small).
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn interior_rows(
    x: &[f64],
    y: &[f64],
    bb: [f64; 4],
    g: &Grid,
    i_lo: usize,
    i_hi: usize,
    j_lo: usize,
    j_hi: usize,
) -> Vec<Option<(usize, usize)>> {
    (j_lo..=j_hi)
        .map(|j| interior_cells(x, y, bb, g.ye[j], g.ye[j + 1], g.xe, i_lo, i_hi))
        .collect()
}

/// Settings shared by the rectilinear kernels.
pub struct RectOpts<'a> {
    pub cell_mask: Option<&'a [bool]>,
    pub with_geom: bool,
    pub to_world: Option<ToWorld>,
    /// Use the analytic interior-cell shortcut for convex subjects.
    pub interior: bool,
}

/// Clip one subject polygon against the cells `i_lo..=i_hi x j_lo..=j_hi` of
/// an axis-aligned grid and push every positive-area entry.
/// `convex`: exterior is convex without holes (the clip is then exact geometry).
/// `prefilter`: drop cells whose bbox overlap is <= 1e-15 in x or y.
/// `scale`: divisor applied to the area (cell area for normalised weights).
#[allow(clippy::too_many_arguments)]
fn clip_item_cells(
    acc: &mut Acc,
    sc: &mut Scratch,
    item: usize,
    (ex, ey): (&[f64], &[f64]),
    holes: Option<(&Polys, usize)>,
    convex: bool,
    bb: [f64; 4],
    (i_lo, i_hi, j_lo, j_hi): (usize, usize, usize, usize),
    g: &Grid,
    o: &RectOpts,
    prefilter: bool,
    scale: f64,
    item_is_a: bool,
) {
    let nx = g.nx();
    let small = convex && ex.len() + 4 <= STACK;
    let use_interior = o.interior
        && convex
        && i_hi + 1 >= i_lo + INTERIOR_MIN_SPAN
        && j_hi + 1 >= j_lo + INTERIOR_MIN_SPAN;
    let tw = o.to_world.as_ref();
    let push = |_: &mut Acc, cell: usize, w: f64| -> (usize, usize, f64) {
        if item_is_a {
            (item, cell, w)
        } else {
            (cell, item, w)
        }
    };
    let rows = if use_interior {
        Some(interior_rows(ex, ey, bb, g, i_lo, i_hi, j_lo, j_hi))
    } else {
        None
    };
    for j in j_lo..=j_hi {
        let (cy0, cy1) = (g.ye[j], g.ye[j + 1]);
        if prefilter && !positive(cy1.min(bb[3]) - cy0.max(bb[1])) {
            continue;
        }
        let inner = rows.as_ref().and_then(|r| r[j - j_lo]);
        for i in i_lo..=i_hi {
            let r = j * nx + i;
            if !mask_ok(o.cell_mask, r) {
                continue;
            }
            let (cx0, cx1) = (g.xe[i], g.xe[i + 1]);
            if prefilter && !positive(cx1.min(bb[2]) - cx0.max(bb[0])) {
                continue;
            }
            if let Some((a, b)) = inner {
                if i >= a && i <= b {
                    let ar = (cx1 - cx0) * (cy1 - cy0);
                    let (u, v, w) = push(acc, r, ar / scale);
                    acc.push_rect(u, v, w, [cx0, cy0, cx1, cy1], o.with_geom, tw);
                    continue;
                }
            }
            if small {
                let (pp, qq) = (&mut *sc.sp, &mut *sc.sq);
                pp.load(ex, ey);
                clip_rect(pp, qq, cx0, cy0, cx1, cy1);
                if pp.len() < 3 {
                    continue;
                }
                let ar = area(pp);
                if ar > EPS_AREA {
                    let (u, v, w) = push(acc, r, ar / scale);
                    acc.push_buf(u, v, w, &*pp, o.with_geom, tw);
                }
                continue;
            }
            // general subject: exterior minus holes, growable buffers
            sc.p.load(ex, ey);
            clip_rect(&mut sc.p, &mut sc.q, cx0, cy0, cx1, cy1);
            let mut ar = if sc.p.len() < 3 { 0.0 } else { area(&sc.p) };
            if convex {
                if ar > EPS_AREA {
                    let (u, v, w) = push(acc, r, ar / scale);
                    acc.push_buf(u, v, w, &sc.p, o.with_geom, tw);
                }
                continue;
            }
            if ar <= 0.0 {
                continue;
            }
            if let Some((polys, p)) = holes {
                for h in polys.holes(p) {
                    let (hx, hy) = polys.ring(h);
                    sc.p.load(hx, hy);
                    clip_rect(&mut sc.p, &mut sc.q, cx0, cy0, cx1, cy1);
                    if sc.p.len() >= 3 {
                        ar -= area(&sc.p);
                    }
                }
            }
            if ar > EPS_AREA {
                let (u, v, w) = push(acc, r, ar / scale);
                acc.push_bare(u, v, w, KIND_AREA, o.with_geom);
            }
        }
    }
}

/// Rectilinear grid (local frame) vs a polygon set already in that frame.
/// Output is item-major (`a` = item, `b` = cell), cells row-major.
pub fn rect_items(
    polys: &Polys,
    convex: &[bool],
    g: &Grid,
    item_mask: Option<&[bool]>,
    o: &RectOpts,
) -> Acc {
    let (nx, ny) = (g.nx(), g.ny());
    let (x0, y0) = (g.xe[0], g.ye[0]);
    let dx = if nx > 0 { g.xe[1] - g.xe[0] } else { 1.0 };
    let dy = if ny > 0 { g.ye[1] - g.ye[0] } else { 1.0 };
    let (inv_dx, inv_dy) = (1.0 / dx, 1.0 / dy);
    let bounds = polys.bounds();
    let n = polys.n();
    let parts: Vec<Acc> = (0..n.div_ceil(128))
        .into_par_iter()
        .map_init(Scratch::default, |sc, bi| {
            let mut acc = Acc::default();
            for it in bi * 128..((bi + 1) * 128).min(n) {
                if !mask_ok(item_mask, it) || polys.exterior(it).0.len() < 3 {
                    continue;
                }
                let b = bounds[it];
                let i_lo = clampi((b[0] - x0) * inv_dx, nx as isize - 1);
                let i_hi = clampi((b[2] - x0) * inv_dx - 1e-12, nx as isize - 1).max(i_lo);
                let j_lo = clampi((b[1] - y0) * inv_dy, ny as isize - 1);
                let j_hi = clampi((b[3] - y0) * inv_dy - 1e-12, ny as isize - 1).max(j_lo);
                let rng = (i_lo as usize, i_hi as usize, j_lo as usize, j_hi as usize);
                clip_item_cells(
                    &mut acc,
                    sc,
                    it,
                    polys.exterior(it),
                    Some((polys, it)),
                    convex[it],
                    b,
                    rng,
                    g,
                    o,
                    true,
                    1.0,
                    true,
                );
            }
            acc
        })
        .collect();
    Acc::concat(parts)
}

/// Parameters of the non-separable rectilinear/rectilinear transfer (scalar
/// set-up done in numpy by the caller so the inputs are bit-identical).
pub struct NonSep {
    pub snx: usize,
    pub sny: usize,
    pub sdx: f64,
    pub sdy: f64,
    pub tnx: usize,
    pub tny: usize,
    pub tdx: f64,
    pub tdy: f64,
    pub origin_x: f64,
    pub origin_y: f64,
    pub cr: f64,
    pub sr: f64,
    pub template_x: [f64; 4],
    pub template_y: [f64; 4],
    pub half_bx: f64,
    pub half_by: f64,
}

/// Non-separable transfer: target cells (rotated rectangles) vs source cells.
/// Output is target-major (`a` = source cell, `b` = target cell), weights
/// already divided by the source cell area.
pub fn nonseparable(q: &NonSep, tgt_mask: Option<&[bool]>, o: &RectOpts) -> Acc {
    let sxe: Vec<f64> = (0..=q.snx)
        .map(|i| (i as f64 - q.snx as f64 / 2.0) * q.sdx)
        .collect();
    let sye: Vec<f64> = (0..=q.sny)
        .map(|j| (j as f64 - q.sny as f64 / 2.0) * q.sdy)
        .collect();
    let kx: Vec<f64> = (0..q.tnx)
        .map(|k| (k as f64 - (q.tnx as f64 - 1.0) / 2.0) * q.tdx)
        .collect();
    let ly: Vec<f64> = (0..q.tny)
        .map(|l| (l as f64 - (q.tny as f64 - 1.0) / 2.0) * q.tdy)
        .collect();
    let g = Grid { xe: &sxe, ye: &sye };
    let (sx0, sy0) = (sxe[0], sye[0]);
    let (inv_sdx, inv_sdy) = (1.0 / q.sdx, 1.0 / q.sdy);
    let src_area = q.sdx * q.sdy;
    let n_target = q.tnx * q.tny;
    let block = 256usize;
    let parts: Vec<Acc> = (0..n_target.div_ceil(block))
        .into_par_iter()
        .map_init(Scratch::default, |sc, bi| {
            let mut acc = Acc::default();
            for t in bi * block..((bi + 1) * block).min(n_target) {
                if !mask_ok(tgt_mask, t) {
                    continue;
                }
                let (k, l) = (t % q.tnx, t / q.tnx);
                let tcx = (q.origin_x + q.cr * kx[k]) - q.sr * ly[l];
                let tcy = (q.origin_y + q.sr * kx[k]) + q.cr * ly[l];
                let i_lo = clampi((tcx - q.half_bx - sx0) * inv_sdx, q.snx as isize - 1);
                let i_hi = clampi((tcx + q.half_bx - sx0) * inv_sdx, q.snx as isize - 1);
                let j_lo = clampi((tcy - q.half_by - sy0) * inv_sdy, q.sny as isize - 1);
                let j_hi = clampi((tcy + q.half_by - sy0) * inv_sdy, q.sny as isize - 1);
                let mut vx = [0.0f64; 4];
                let mut vy = [0.0f64; 4];
                let mut bb = [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ];
                for p in 0..4 {
                    vx[p] = q.template_x[p] + tcx;
                    vy[p] = q.template_y[p] + tcy;
                    bb = [
                        bb[0].min(vx[p]),
                        bb[1].min(vy[p]),
                        bb[2].max(vx[p]),
                        bb[3].max(vy[p]),
                    ];
                }
                let rng = (i_lo as usize, i_hi as usize, j_lo as usize, j_hi as usize);
                clip_item_cells(
                    &mut acc,
                    sc,
                    t,
                    (&vx, &vy),
                    None,
                    true,
                    bb,
                    rng,
                    &g,
                    o,
                    false,
                    src_area,
                    false,
                );
            }
            acc
        })
        .collect();
    Acc::concat(parts)
}

/// Separable transfer from the two 1-D overlap lists. Output order is the
/// reference one (x-pairs outer, y-pairs inner). The output size is known, so
/// it is written in place in parallel, then compacted only if masks apply.
#[allow(clippy::too_many_arguments)]
pub fn separable(
    is_x: &[isize],
    it_x: &[isize],
    ox: &[f64],
    xlo: &[f64],
    xhi: &[f64],
    is_y: &[isize],
    it_y: &[isize],
    oy: &[f64],
    ylo: &[f64],
    yhi: &[f64],
    sx_is_tgt_y: bool,
    snx: usize,
    tnx: usize,
    src_area: f64,
    src_mask: Option<&[bool]>,
    tgt_mask: Option<&[bool]>,
    with_geom: bool,
    tf: Option<ToWorld>,
) -> Acc {
    let (nxp, nyp) = (is_x.len(), is_y.len());
    let n = nxp * nyp;
    let mut acc = Acc {
        a: vec![0; n],
        b: vec![0; n],
        w: vec![0.0; n],
        kind: vec![0; n],
        coords: vec![],
        ring_len: if with_geom { vec![5; n] } else { vec![] },
    };
    if n == 0 {
        return acc;
    }
    let masked = src_mask.is_some() || tgt_mask.is_some();
    // unused buffers get one dummy slot per x-pair so the zip lengths agree
    let (gw, kw) = (
        if with_geom { nyp * 10 } else { 1 },
        if masked { nyp } else { 1 },
    );
    let mut coords = vec![0.0; nxp * gw];
    let mut keep = vec![true; nxp * kw];
    acc.a
        .par_chunks_mut(nyp)
        .zip(acc.b.par_chunks_mut(nyp))
        .zip(acc.w.par_chunks_mut(nyp))
        .zip(coords.par_chunks_mut(gw))
        .zip(keep.par_chunks_mut(kw))
        .enumerate()
        .for_each(|(ia, ((((ra, rb), rw), rc), rk))| {
            for jb in 0..nyp {
                let s = (is_y[jb] as usize) * snx + is_x[ia] as usize;
                let (tix, tjy) = if sx_is_tgt_y {
                    (it_y[jb], it_x[ia])
                } else {
                    (it_x[ia], it_y[jb])
                };
                let t = tjy as usize * tnx + tix as usize;
                ra[jb] = s as isize;
                rb[jb] = t as isize;
                rw[jb] = ox[ia] * oy[jb] / src_area;
                if masked {
                    rk[jb] = mask_ok(src_mask, s) && mask_ok(tgt_mask, t);
                }
                if with_geom {
                    let (x0, x1, y0, y1) = (xlo[ia], xhi[ia], ylo[jb], yhi[jb]);
                    let pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)];
                    for (q, (x, y)) in pts.into_iter().enumerate() {
                        let (x, y) = match tf.as_ref() {
                            Some(t) => t.apply(x, y),
                            None => (x, y),
                        };
                        rc[jb * 10 + 2 * q] = x;
                        rc[jb * 10 + 2 * q + 1] = y;
                    }
                }
            }
        });
    if masked {
        let mut out = Acc::default();
        for k in (0..n).filter(|&k| keep[k]) {
            out.a.push(acc.a[k]);
            out.b.push(acc.b[k]);
            out.w.push(acc.w[k]);
            out.kind.push(0);
            if with_geom {
                out.coords.extend_from_slice(&coords[k * 10..k * 10 + 10]);
                out.ring_len.push(5);
            }
        }
        return out;
    }
    if with_geom {
        acc.coords = coords;
    }
    acc
}

// ---------------------------------------------------------------------------
// Irregular sets: grid-hash bbox join + clip
// ---------------------------------------------------------------------------

struct BinGrid {
    gx0: f64,
    gy0: f64,
    inv: f64,
    gnx: usize,
    gny: usize,
    start: Vec<usize>,
    items: Vec<u32>,
}

impl BinGrid {
    fn build(b: &[[f64; 4]], keep: &[usize]) -> BinGrid {
        let (mut x0, mut y0, mut x1, mut y1) = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        let mut wsum = 0.0;
        for &i in keep {
            let r = b[i];
            x0 = x0.min(r[0]);
            y0 = y0.min(r[1]);
            x1 = x1.max(r[2]);
            y1 = y1.max(r[3]);
            wsum += (r[2] - r[0]).max(r[3] - r[1]);
        }
        let n = keep.len().max(1);
        let mut cell = (wsum / n as f64).max(1e-300);
        let span_x = (x1 - x0).max(1e-300);
        let span_y = (y1 - y0).max(1e-300);
        while (span_x / cell).ceil() * (span_y / cell).ceil() > 2.0 * n as f64 + 16.0 {
            cell *= 1.5;
        }
        let gnx = ((span_x / cell).ceil() as usize).max(1);
        let gny = ((span_y / cell).ceil() as usize).max(1);
        let mut g = BinGrid {
            gx0: x0,
            gy0: y0,
            inv: 1.0 / cell,
            gnx,
            gny,
            start: vec![0; gnx * gny + 1],
            items: vec![],
        };
        for &i in keep {
            let (a, b2, c, d) = g.range(&b[i]);
            for cy in c..=d {
                for cx in a..=b2 {
                    g.start[cy * gnx + cx + 1] += 1;
                }
            }
        }
        for k in 0..gnx * gny {
            g.start[k + 1] += g.start[k];
        }
        let mut fill = g.start.clone();
        g.items = vec![0; g.start[gnx * gny]];
        for &i in keep {
            let (a, b2, c, d) = g.range(&b[i]);
            for cy in c..=d {
                for cx in a..=b2 {
                    let k = cy * gnx + cx;
                    g.items[fill[k]] = i as u32;
                    fill[k] += 1;
                }
            }
        }
        g
    }
    #[inline(always)]
    fn cx(&self, x: f64) -> usize {
        clampi((x - self.gx0) * self.inv, self.gnx as isize - 1) as usize
    }
    #[inline(always)]
    fn cy(&self, y: f64) -> usize {
        clampi((y - self.gy0) * self.inv, self.gny as isize - 1) as usize
    }
    #[inline(always)]
    fn range(&self, r: &[f64; 4]) -> (usize, usize, usize, usize) {
        (self.cx(r[0]), self.cx(r[2]), self.cy(r[1]), self.cy(r[3]))
    }
}

/// For every query box `b` (index order) the tree boxes `a` whose bbox overlap
/// is strictly positive (> 1e-15) in both axes, sorted by `a`.
pub fn bbox_join(
    a: &[[f64; 4]],
    keep_a: &[usize],
    b: &[[f64; 4]],
    keep_b: &[usize],
) -> Vec<Vec<(u32, u32)>> {
    if keep_a.is_empty() || keep_b.is_empty() {
        return vec![];
    }
    let g = BinGrid::build(a, keep_a);
    keep_b
        .par_chunks(256)
        .map(|chunk| {
            let mut out = Vec::new();
            let mut buf: Vec<u32> = Vec::new();
            for &bi in chunk {
                let rb = &b[bi];
                let (c0, c1, d0, d1) = g.range(rb);
                buf.clear();
                for cy in d0..=d1 {
                    for cx in c0..=c1 {
                        let k = cy * g.gnx + cx;
                        for &ai in &g.items[g.start[k]..g.start[k + 1]] {
                            let ra = &a[ai as usize];
                            // dedupe: report only from the cell holding the
                            // lower-left corner of the overlap
                            if g.cx(ra[0].max(rb[0])) != cx || g.cy(ra[1].max(rb[1])) != cy {
                                continue;
                            }
                            let ox = ra[2].min(rb[2]) - ra[0].max(rb[0]);
                            let oy = ra[3].min(rb[3]) - ra[1].max(rb[1]);
                            if ox > 1e-15 && oy > 1e-15 {
                                buf.push(ai);
                            }
                        }
                    }
                }
                buf.sort_unstable();
                out.extend(buf.iter().map(|&ai| (bi as u32, ai)));
            }
            out
        })
        .collect()
}

pub fn keep_idx(n: usize, m: Option<&[bool]>) -> Vec<usize> {
    (0..n).filter(|&i| mask_ok(m, i)).collect()
}

/// Two irregular polygon sets: bbox join, then per pair a clip with a convex
/// side as the window (`prefer_a_window` decides which side is tried first).
/// A non-convex subject is clipped exterior-minus-holes (exact area, geometry
/// from GEOS); pairs with no convex side are returned as `KIND_GEOS`.
/// Output is b-major, a ascending.
#[allow(clippy::too_many_arguments)]
pub fn join_clip(
    pa: &Polys,
    ca: &[bool],
    pb: &Polys,
    cb: &[bool],
    a_mask: Option<&[bool]>,
    b_mask: Option<&[bool]>,
    prefer_a_window: bool,
    with_geom: bool,
) -> Acc {
    let (ba, bb) = (pa.bounds(), pb.bounds());
    // degenerate polygons (< 3 vertices) can never overlap anything
    let ka: Vec<usize> = keep_idx(pa.n(), a_mask)
        .into_iter()
        .filter(|&i| pa.exterior(i).0.len() >= 3)
        .collect();
    let kb: Vec<usize> = keep_idx(pb.n(), b_mask)
        .into_iter()
        .filter(|&i| pb.exterior(i).0.len() >= 3)
        .collect();
    let (wax, way, waoff) = pa.ccw_exteriors(ca);
    let (wbx, wby, wboff) = pb.ccw_exteriors(cb);
    let parts = bbox_join(&ba, &ka, &bb, &kb);
    let accs: Vec<Acc> = parts
        .into_par_iter()
        .map_init(Scratch::default, |sc, pairs| {
            let mut acc = Acc::default();
            for (b, a) in pairs {
                let (b, a) = (b as usize, a as usize);
                // (subject set, subject index, subject convex, window coords)
                let a_win = || (&wax[waoff[a]..waoff[a + 1]], &way[waoff[a]..waoff[a + 1]]);
                let b_win = || (&wbx[wboff[b]..wboff[b + 1]], &wby[wboff[b]..wboff[b + 1]]);
                let choice = if prefer_a_window {
                    if ca[a] {
                        Some((pb, b, cb[b], a_win()))
                    } else if cb[b] {
                        Some((pa, a, ca[a], b_win()))
                    } else {
                        None
                    }
                } else if cb[b] {
                    Some((pa, a, ca[a], b_win()))
                } else if ca[a] {
                    Some((pb, b, cb[b], a_win()))
                } else {
                    None
                };
                let Some((sp, s, s_convex, (wx, wy))) = choice else {
                    acc.push_bare(a, b, f64::NAN, KIND_GEOS, with_geom);
                    continue;
                };
                let (ex, ey) = sp.exterior(s);
                if ex.len() < 3 {
                    continue;
                }
                if s_convex && ex.len() + wx.len() <= STACK {
                    let (pp, qq) = (&mut *sc.sp, &mut *sc.sq);
                    pp.load(ex, ey);
                    clip_convex(pp, qq, wx, wy);
                    if pp.len() < 3 {
                        continue;
                    }
                    let ar = area(pp);
                    if ar > EPS_AREA {
                        acc.push_buf(a, b, ar, &*pp, with_geom, None);
                    }
                    continue;
                }
                sc.p.load(ex, ey);
                clip_convex(&mut sc.p, &mut sc.q, wx, wy);
                let mut ar = if sc.p.len() < 3 { 0.0 } else { area(&sc.p) };
                if s_convex {
                    if ar > EPS_AREA {
                        acc.push_buf(a, b, ar, &sc.p, with_geom, None);
                    }
                    continue;
                }
                if ar <= 0.0 {
                    continue;
                }
                for h in sp.holes(s) {
                    let (hx, hy) = sp.ring(h);
                    sc.p.load(hx, hy);
                    clip_convex(&mut sc.p, &mut sc.q, wx, wy);
                    if sc.p.len() >= 3 {
                        ar -= area(&sc.p);
                    }
                }
                if ar > EPS_AREA {
                    acc.push_bare(a, b, ar, KIND_AREA, with_geom);
                }
            }
            acc
        })
        .collect();
    Acc::concat(accs)
}

/// COO -> CSC for unique (row, col) pairs. Returns `(data, indices, indptr)`.
pub fn coo_to_csc(
    r: &[isize],
    c: &[isize],
    d: &[f64],
    n_cols: usize,
) -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let nnz = r.len();
    let mut cp = vec![0i32; n_cols + 1];
    for &x in c {
        cp[x as usize + 1] += 1;
    }
    for i in 0..n_cols {
        cp[i + 1] += cp[i];
    }
    let sorted_cols = c.par_windows(2).all(|w| w[0] <= w[1]);
    let mut oi: Vec<i32>;
    let mut od: Vec<f64> = if sorted_cols {
        oi = r.par_iter().map(|&x| x as i32).collect();
        d.to_vec()
    } else {
        let mut fill: Vec<i32> = cp[..n_cols].to_vec();
        let mut od = vec![0.0f64; nnz];
        oi = vec![0i32; nnz];
        for k in 0..nnz {
            let col = c[k] as usize;
            let dst = fill[col] as usize;
            fill[col] += 1;
            od[dst] = d[k];
            oi[dst] = r[k] as i32;
        }
        od
    };
    // per-column insertion sort on the row index (columns hold few entries)
    const B: usize = 4096;
    let nb = n_cols.div_ceil(B);
    let mut i_chunks: Vec<&mut [i32]> = Vec::with_capacity(nb);
    let mut d_chunks: Vec<&mut [f64]> = Vec::with_capacity(nb);
    {
        let (mut ri, mut rd) = (&mut oi[..], &mut od[..]);
        for b in 0..nb {
            let len = (cp[((b + 1) * B).min(n_cols)] - cp[b * B]) as usize;
            let (hi, ti) = std::mem::take(&mut ri).split_at_mut(len);
            let (hd, td) = std::mem::take(&mut rd).split_at_mut(len);
            i_chunks.push(hi);
            d_chunks.push(hd);
            ri = ti;
            rd = td;
        }
    }
    i_chunks
        .into_par_iter()
        .zip(d_chunks.into_par_iter())
        .enumerate()
        .for_each(|(b, (ci, cd))| {
            let base = cp[b * B] as usize;
            for col in b * B..((b + 1) * B).min(n_cols) {
                let (s0, s1) = (cp[col] as usize - base, cp[col + 1] as usize - base);
                for x in s0 + 1..s1 {
                    let (ki, kd) = (ci[x], cd[x]);
                    let mut y = x;
                    while y > s0 && ci[y - 1] > ki {
                        ci[y] = ci[y - 1];
                        cd[y] = cd[y - 1];
                        y -= 1;
                    }
                    ci[y] = ki;
                    cd[y] = kd;
                }
            }
        });
    (od, oi, cp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_square(x0: f64, y0: f64, s: f64) -> Polys {
        Polys {
            x: vec![x0, x0 + s, x0 + s, x0, x0],
            y: vec![y0, y0, y0 + s, y0 + s, y0],
            ring_off: vec![0, 5],
            poly_ring: vec![0, 1],
        }
    }

    fn opts() -> RectOpts<'static> {
        RectOpts {
            cell_mask: None,
            with_geom: true,
            to_world: None,
            interior: true,
        }
    }

    #[test]
    fn square_over_grid_conserves_area_with_interior_shortcut() {
        let xe: Vec<f64> = (0..=10).map(|i| i as f64).collect();
        let g = Grid { xe: &xe, ye: &xe };
        let p = one_square(1.5, 2.25, 6.0);
        let acc = rect_items(&p, &[true], &g, None, &opts());
        let tot: f64 = acc.w.iter().sum();
        assert!((tot - 36.0).abs() < 1e-12, "{tot}");
        // 5x5 fully interior cells have area exactly 1
        assert_eq!(acc.w.iter().filter(|&&w| w == 1.0).count(), 25);
        let mut o = opts();
        o.interior = false;
        let acc2 = rect_items(&p, &[true], &g, None, &o);
        assert_eq!(acc.a.len(), acc2.a.len());
        for (x, y) in acc.w.iter().zip(&acc2.w) {
            assert!((x - y).abs() < 1e-14);
        }
    }

    #[test]
    fn holed_polygon_area_subtracts_holes() {
        // 4x4 square with a 2x2 hole, over a 1x1 grid
        let p = Polys {
            x: vec![0., 4., 4., 0., 0., 1., 1., 3., 3., 1.],
            y: vec![0., 0., 4., 4., 0., 1., 3., 3., 1., 1.],
            ring_off: vec![0, 5, 10],
            poly_ring: vec![0, 2],
        };
        let xe: Vec<f64> = (0..=4).map(|i| i as f64).collect();
        let g = Grid { xe: &xe, ye: &xe };
        let acc = rect_items(&p, &[false], &g, None, &opts());
        let tot: f64 = acc.w.iter().sum();
        assert!((tot - 12.0).abs() < 1e-12);
        assert_eq!(acc.a.len(), 12); // the 4 hole cells are empty
    }

    #[test]
    fn bbox_join_dedupes_and_sorts() {
        let a = vec![
            [0.0, 0.0, 2.0, 2.0],
            [1.0, 1.0, 3.0, 3.0],
            [5.0, 5.0, 6.0, 6.0],
        ];
        let b = vec![[1.5, 1.5, 2.5, 2.5]];
        let r = bbox_join(&a, &[0, 1, 2], &b, &[0]);
        let flat: Vec<_> = r.into_iter().flatten().collect();
        assert_eq!(flat, vec![(0, 0), (0, 1)]);
    }

    #[test]
    fn csc_matches_dense() {
        let r = [2isize, 0, 1, 0];
        let c = [1isize, 1, 0, 0];
        let d = [1.0, 2.0, 3.0, 4.0];
        let (od, oi, op) = coo_to_csc(&r, &c, &d, 2);
        assert_eq!(op, vec![0, 2, 4]);
        assert_eq!(oi, vec![0, 1, 0, 2]);
        assert_eq!(od, vec![4.0, 3.0, 2.0, 1.0]);
    }
}
