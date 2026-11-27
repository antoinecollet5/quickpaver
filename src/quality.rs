//! Roundness quality of polygonal cells (triangles, Voronoi cells, tiles).
//!
//! The score is `1` for a regular polygon and decreases towards `0` as the
//! cell gets distorted:
//!
//! * triangles use the exact `2 r / R` ratio (inradius over circumradius);
//! * other polygons use `(r_in / r_circ) / cos(pi / n)`, where `r_circ` is the
//!   largest vertex distance and `r_in` the smallest edge distance from a
//!   reference point (the area-weighted centroid unless one is given).
//!
//! Polygons are read from an interleaved `[x0, y0, x1, y1, ...]` vertex buffer
//! and CSR offsets (in vertices), so both `shapely.get_coordinates` output and
//! `(n, k, 2)` vertex arrays are used without any copy. Every computation runs
//! relative to the reference point (or the first vertex), which keeps full
//! precision for cells far from the origin (e.g. projected coordinates).

use rayon::prelude::*;

/// Below this area a triangle is degenerate (score 0), as in the reference.
const AREA_EPS: f64 = 1e-15;
/// Squared length below which an edge is treated as a point.
const SEG_EPS: f64 = 1e-15;

/// A set of polygons: vertex `v` is `(xy[2 v], xy[2 v + 1])`, polygon `i`
/// spans vertices `off[i]..off[i + 1]` (minus the last one if `closed`).
pub struct Rings<'a> {
    pub xy: &'a [f64],
    pub off: &'a [isize],
    pub closed: bool,
}

impl Rings<'_> {
    #[inline(always)]
    pub fn n(&self) -> usize {
        self.off.len().saturating_sub(1)
    }

    /// Interleaved open ring of polygon `i`.
    #[inline(always)]
    pub fn ring(&self, i: usize) -> &[f64] {
        let s = self.off[i] as usize;
        let mut e = self.off[i + 1] as usize;
        if self.closed && e > s {
            e -= 1;
        }
        &self.xy[2 * s..2 * e]
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.xy.len().is_multiple_of(2) {
            return Err("vertex buffer must hold (x, y) pairs".into());
        }
        let nv = (self.xy.len() / 2) as isize;
        if self.off.is_empty() || self.off[0] < 0 || *self.off.last().unwrap() > nv {
            return Err("offsets out of bounds".into());
        }
        if self.off.windows(2).any(|w| w[0] > w[1]) {
            return Err("offsets must be non-decreasing".into());
        }
        Ok(())
    }
}

/// `2 r / R` of a triangle, from the cross product (exact up to rounding,
/// unlike Heron's formula which loses precision on needle triangles):
/// `2 r / R = 8 A^2 / (s a b c)` with `A = |cross| / 2`.
#[inline]
pub fn triangle_quality(p: &[f64]) -> f64 {
    let (ux, uy) = (p[2] - p[0], p[3] - p[1]);
    let (vx, vy) = (p[4] - p[0], p[5] - p[1]);
    let cross = ux * vy - uy * vx;
    if 0.5 * cross.abs() < AREA_EPS {
        return 0.0;
    }
    let c = (ux * ux + uy * uy).sqrt();
    let b = (vx * vx + vy * vy).sqrt();
    let (wx, wy) = (vx - ux, vy - uy);
    let a = (wx * wx + wy * wy).sqrt();
    let s = 0.5 * (a + b + c);
    2.0 * cross * cross / (s * a * b * c)
}

/// Area-weighted centroid of an open interleaved ring, relative to its
/// first vertex; vertex mean for a zero-area ring.
#[inline]
fn centroid_rel(p: &[f64]) -> (f64, f64) {
    let n = p.len() / 2;
    let (x0, y0) = (p[0], p[1]);
    let (mut a2, mut cx, mut cy) = (0.0, 0.0, 0.0);
    // the first vertex is the origin: edges (n-1 -> 0) and (0 -> 1) add nothing
    let (mut xa, mut ya) = (p[2] - x0, p[3] - y0);
    for v in 2..n {
        let (xb, yb) = (p[2 * v] - x0, p[2 * v + 1] - y0);
        let cr = xa * yb - xb * ya;
        a2 += cr;
        cx += (xa + xb) * cr;
        cy += (ya + yb) * cr;
        (xa, ya) = (xb, yb);
    }
    if (0.5 * a2).abs() < AREA_EPS {
        let (mut mx, mut my) = (0.0, 0.0);
        for v in 0..n {
            mx += p[2 * v] - x0;
            my += p[2 * v + 1] - y0;
        }
        return (mx / n as f64, my / n as f64);
    }
    (cx / (3.0 * a2), cy / (3.0 * a2))
}

/// Squared distance from the origin to segment `a -> b`.
#[inline(always)]
fn seg_dist2_origin(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let (ex, ey) = (bx - ax, by - ay);
    let den = ex * ex + ey * ey;
    if den < SEG_EPS {
        return ax * ax + ay * ay;
    }
    let t = (-(ax * ex + ay * ey) / den).clamp(0.0, 1.0);
    let (dx, dy) = (ax + t * ex, ay + t * ey);
    dx * dx + dy * dy
}

/// Generic n-gon roundness around `(cx, cy)` (absolute coordinates if
/// `rel` is false, relative to the first vertex if true).
#[inline]
fn polygon_roundness_at(p: &[f64], cx: f64, cy: f64, rel: bool) -> f64 {
    let n = p.len() / 2;
    let (ox, oy) = if rel {
        (p[0] + cx, p[1] + cy)
    } else {
        (cx, cy)
    };
    let mut rc2 = 0.0f64;
    let mut ri2 = f64::INFINITY;
    let (mut ax, mut ay) = (p[2 * n - 2] - ox, p[2 * n - 1] - oy);
    for v in 0..n {
        let (bx, by) = (p[2 * v] - ox, p[2 * v + 1] - oy);
        rc2 = rc2.max(bx * bx + by * by);
        ri2 = ri2.min(seg_dist2_origin(ax, ay, bx, by));
        (ax, ay) = (bx, by);
    }
    if rc2 == 0.0 {
        return 0.0;
    }
    (ri2 / rc2).sqrt() / (std::f64::consts::PI / n as f64).cos()
}

/// Quality of one open interleaved ring; `center` overrides the centroid of
/// non-triangular cells. Empty rings give `NaN`; 1- and 2-vertex rings `0`.
#[inline]
pub fn cell_quality(p: &[f64], center: Option<(f64, f64)>) -> f64 {
    match p.len() / 2 {
        0 => f64::NAN,
        1 | 2 => 0.0,
        3 => triangle_quality(p),
        _ => match center {
            Some((cx, cy)) => polygon_roundness_at(p, cx, cy, false),
            None => {
                let (cx, cy) = centroid_rel(p);
                polygon_roundness_at(p, cx, cy, true)
            }
        },
    }
}

/// Quality of every polygon; `centers` is interleaved `(n, 2)` if given.
pub fn rings_quality(r: &Rings<'_>, centers: Option<&[f64]>, parallel: bool) -> Vec<f64> {
    let f = |i: usize| cell_quality(r.ring(i), centers.map(|c| (c[2 * i], c[2 * i + 1])));
    if parallel {
        (0..r.n())
            .into_par_iter()
            .with_min_len(2048)
            .map(f)
            .collect()
    } else {
        (0..r.n()).map(f).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regular(n: usize, r: f64, x0: f64, y0: f64) -> Vec<f64> {
        (0..n)
            .flat_map(|k| {
                let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                [x0 + r * t.cos(), y0 + r * t.sin()]
            })
            .collect()
    }

    #[test]
    fn regular_polygons_score_one() {
        for n in 3..12 {
            let p = regular(n, 2.5, 1e6, -3e6);
            assert!((cell_quality(&p, None) - 1.0).abs() < 1e-9, "n = {n}");
        }
    }

    #[test]
    fn degenerate_cells() {
        assert_eq!(cell_quality(&[0.0, 0.0, 1.0, 0.0, 2.0, 0.0], None), 0.0);
        assert_eq!(cell_quality(&[1.0; 8], None), 0.0);
        assert!(cell_quality(&[], None).is_nan());
    }

    #[test]
    fn closed_rings_drop_last_vertex() {
        let mut p = regular(6, 1.0, 0.0, 0.0);
        p.extend_from_slice(&[p[0], p[1]]);
        let off = [0isize, 7];
        let r = Rings {
            xy: &p,
            off: &off,
            closed: true,
        };
        assert!(r.validate().is_ok());
        assert!((rings_quality(&r, None, false)[0] - 1.0).abs() < 1e-12);
    }
}
