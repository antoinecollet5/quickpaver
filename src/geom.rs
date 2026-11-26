//! Ragged polygon sets, frames, convexity and the output accumulator.

use crate::clip::{signed_area2, Buf};
use rayon::prelude::*;

/// World -> local frame of a rotated grid (`(dx*ca + dy*sa, -dx*sa + dy*ca)`).
#[derive(Clone, Copy)]
pub struct ToLocal {
    pub cx: f64,
    pub cy: f64,
    pub ca: f64,
    pub sa: f64,
}

impl ToLocal {
    #[inline(always)]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = (x - self.cx, y - self.cy);
        (dx * self.ca + dy * self.sa, -dx * self.sa + dy * self.ca)
    }
}

/// Local -> world (`(c + x*ca) - y*sa`, `(c + x*sa) + y*ca`).
#[derive(Clone, Copy)]
pub struct ToWorld {
    pub cx: f64,
    pub cy: f64,
    pub ca: f64,
    pub sa: f64,
}

impl ToWorld {
    #[inline(always)]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            (self.cx + x * self.ca) - y * self.sa,
            (self.cy + x * self.sa) + y * self.ca,
        )
    }
}

/// A set of polygons in the layout of `shapely.get_coordinates(get_rings(...))`:
/// closed rings (`x`, `y`), `ring_off` (n_rings + 1) and `poly_ring` (n_polys + 1),
/// exterior ring first for every polygon.
#[derive(Clone)]
pub struct Polys {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub ring_off: Vec<usize>,
    pub poly_ring: Vec<usize>,
}

impl Polys {
    pub fn from_slices(
        x: &[f64],
        y: &[f64],
        ring_off: &[isize],
        poly_ring: &[isize],
    ) -> Result<Polys, String> {
        let ring_off: Vec<usize> = ring_off.iter().map(|&v| v as usize).collect();
        let poly_ring: Vec<usize> = poly_ring.iter().map(|&v| v as usize).collect();
        if x.len() != y.len()
            || ring_off.is_empty()
            || poly_ring.is_empty()
            || *ring_off.last().unwrap() != x.len()
            || *poly_ring.last().unwrap() != ring_off.len() - 1
            || ring_off.windows(2).any(|w| w[0] > w[1])
            || poly_ring.windows(2).any(|w| w[0] > w[1])
        {
            return Err("inconsistent ragged polygon arrays".into());
        }
        Ok(Polys {
            x: x.to_vec(),
            y: y.to_vec(),
            ring_off,
            poly_ring,
        })
    }

    /// Triangles given as `(n, 3)` coordinate arrays (closing vertex added).
    pub fn from_triangles(tx: &[f64], ty: &[f64]) -> Polys {
        let n = tx.len() / 3;
        let mut x = Vec::with_capacity(n * 4);
        let mut y = Vec::with_capacity(n * 4);
        for t in 0..n {
            x.extend_from_slice(&tx[t * 3..t * 3 + 3]);
            x.push(tx[t * 3]);
            y.extend_from_slice(&ty[t * 3..t * 3 + 3]);
            y.push(ty[t * 3]);
        }
        Polys {
            x,
            y,
            ring_off: (0..=n).map(|t| t * 4).collect(),
            poly_ring: (0..=n).collect(),
        }
    }

    #[inline(always)]
    pub fn n(&self) -> usize {
        self.poly_ring.len() - 1
    }

    /// Open ring `r` (closing vertex dropped).
    #[inline(always)]
    pub fn ring(&self, r: usize) -> (&[f64], &[f64]) {
        let (s, e) = (self.ring_off[r], self.ring_off[r + 1]);
        let e = if e > s { e - 1 } else { e };
        (&self.x[s..e], &self.y[s..e])
    }

    /// Exterior ring (empty for an empty polygon).
    #[inline(always)]
    pub fn exterior(&self, p: usize) -> (&[f64], &[f64]) {
        if self.poly_ring[p] == self.poly_ring[p + 1] {
            return (&[], &[]);
        }
        self.ring(self.poly_ring[p])
    }

    #[inline(always)]
    pub fn holes(&self, p: usize) -> std::ops::Range<usize> {
        (self.poly_ring[p] + 1).min(self.poly_ring[p + 1])..self.poly_ring[p + 1]
    }

    #[inline(always)]
    pub fn has_holes(&self, p: usize) -> bool {
        self.poly_ring[p + 1] > self.poly_ring[p] + 1
    }

    /// Copy expressed in a grid's local frame.
    pub fn to_local(&self, tf: &ToLocal) -> Polys {
        let mut x = vec![0.0; self.x.len()];
        let mut y = vec![0.0; self.y.len()];
        x.par_iter_mut()
            .zip(y.par_iter_mut())
            .enumerate()
            .with_min_len(4096)
            .for_each(|(k, (ox, oy))| {
                let (a, b) = tf.apply(self.x[k], self.y[k]);
                *ox = a;
                *oy = b;
            });
        Polys {
            x,
            y,
            ring_off: self.ring_off.clone(),
            poly_ring: self.poly_ring.clone(),
        }
    }

    /// Exterior-ring bounding boxes `[xmin, ymin, xmax, ymax]` (closing vertex
    /// included: it equals the first one).
    pub fn bounds(&self) -> Vec<[f64; 4]> {
        (0..self.n())
            .into_par_iter()
            .with_min_len(1024)
            .map(|p| {
                let r = self.poly_ring[p];
                let (s, e) = if r == self.poly_ring[p + 1] {
                    (0, 0)
                } else {
                    (self.ring_off[r], self.ring_off[r + 1])
                };
                let mut b = [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ];
                for k in s..e {
                    b[0] = b[0].min(self.x[k]);
                    b[1] = b[1].min(self.y[k]);
                    b[2] = b[2].max(self.x[k]);
                    b[3] = b[3].max(self.y[k]);
                }
                b
            })
            .collect()
    }

    /// Convexity of the exterior ring (holed polygons are never convex). A turn
    /// counts only when |cross| > 1e-15; fewer than 3 vertices -> not convex.
    pub fn convexity(&self) -> Vec<bool> {
        (0..self.n())
            .into_par_iter()
            .with_min_len(1024)
            .map(|p| {
                if self.has_holes(p) {
                    return false;
                }
                let (x, y) = self.exterior(p);
                is_convex_ring(x, y)
            })
            .collect()
    }

    /// CCW copies of the exterior rings of the polygons flagged in `which`
    /// (reversed when clockwise), as `(x, y, offsets)` with open rings.
    pub fn ccw_exteriors(&self, which: &[bool]) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let mut off = Vec::with_capacity(self.n() + 1);
        off.push(0);
        for p in 0..self.n() {
            let m = if which[p] {
                self.exterior(p).0.len()
            } else {
                0
            };
            off.push(off[p] + m);
        }
        let tot = off[self.n()];
        let (mut x, mut y) = (vec![0.0; tot], vec![0.0; tot]);
        for p in 0..self.n() {
            if !which[p] {
                continue;
            }
            let (ex, ey) = self.exterior(p);
            let m = ex.len();
            let (dx, dy) = (&mut x[off[p]..off[p] + m], &mut y[off[p]..off[p] + m]);
            if signed_area2(ex, ey) >= 0.0 {
                dx.copy_from_slice(ex);
                dy.copy_from_slice(ey);
            } else {
                for k in 0..m {
                    dx[k] = ex[m - 1 - k];
                    dy[k] = ey[m - 1 - k];
                }
            }
        }
        (x, y, off)
    }
}

pub fn is_convex_ring(x: &[f64], y: &[f64]) -> bool {
    let n = x.len();
    if n < 3 {
        return false;
    }
    let (mut pos, mut neg) = (false, false);
    for k in 0..n {
        let (p, q) = ((k + n - 1) % n, (k + 1) % n);
        let (e1x, e1y) = (x[k] - x[p], y[k] - y[p]);
        let (e2x, e2y) = (x[q] - x[k], y[q] - y[k]);
        let cr = e1x * e2y - e1y * e2x;
        if cr.abs() > 1e-15 {
            if cr > 0.0 {
                pos = true
            } else {
                neg = true
            }
        }
    }
    !(pos && neg)
}

/// Kind of a result entry.
pub const KIND_GEOM: u8 = 0; // area and exact geometry computed in Rust
pub const KIND_AREA: u8 = 1; // area computed in Rust; geometry must come from GEOS
pub const KIND_GEOS: u8 = 2; // neither side convex: area and geometry from GEOS

/// Per-block result buffers, concatenated in block order (deterministic).
#[derive(Default)]
pub struct Acc {
    pub a: Vec<isize>,
    pub b: Vec<isize>,
    pub w: Vec<f64>,
    pub kind: Vec<u8>,
    /// Closed rings, interleaved `x, y`, one ring per entry with `ring_len > 0`.
    pub coords: Vec<f64>,
    pub ring_len: Vec<isize>,
}

impl Acc {
    pub fn concat(parts: Vec<Acc>) -> Acc {
        let mut out = Acc::default();
        let tot: usize = parts.iter().map(|p| p.a.len()).sum();
        let totc: usize = parts.iter().map(|p| p.coords.len()).sum();
        out.a.reserve_exact(tot);
        out.b.reserve_exact(tot);
        out.w.reserve_exact(tot);
        out.kind.reserve_exact(tot);
        out.ring_len.reserve_exact(tot);
        out.coords.reserve_exact(totc);
        for p in parts {
            out.a.extend_from_slice(&p.a);
            out.b.extend_from_slice(&p.b);
            out.w.extend_from_slice(&p.w);
            out.kind.extend_from_slice(&p.kind);
            out.ring_len.extend_from_slice(&p.ring_len);
            out.coords.extend_from_slice(&p.coords);
        }
        out
    }

    #[inline(always)]
    fn head(&mut self, a: usize, b: usize, w: f64, kind: u8) {
        self.a.push(a as isize);
        self.b.push(b as isize);
        self.w.push(w);
        self.kind.push(kind);
    }

    /// Entry without Rust geometry (area only, or GEOS fallback).
    #[inline(always)]
    pub fn push_bare(&mut self, a: usize, b: usize, w: f64, kind: u8, with_geom: bool) {
        self.head(a, b, w, kind);
        if with_geom {
            self.ring_len.push(0);
        }
    }

    /// Entry with the clipped polygon as geometry.
    #[inline(always)]
    pub fn push_buf<B: Buf>(
        &mut self,
        a: usize,
        b: usize,
        w: f64,
        p: &B,
        with_geom: bool,
        tf: Option<&ToWorld>,
    ) {
        self.head(a, b, w, KIND_GEOM);
        if with_geom {
            let (xs, ys) = (p.xs(), p.ys());
            let n = xs.len();
            let base = self.coords.len();
            self.coords.resize(base + 2 * (n + 1), 0.0);
            let dst = &mut self.coords[base..];
            for k in 0..=n {
                let kk = if k == n { 0 } else { k };
                let (x, y) = match tf {
                    Some(t) => t.apply(xs[kk], ys[kk]),
                    None => (xs[kk], ys[kk]),
                };
                dst[2 * k] = x;
                dst[2 * k + 1] = y;
            }
            self.ring_len.push(n as isize + 1);
        }
    }

    /// Entry whose geometry is an axis-aligned (local-frame) rectangle,
    /// written in the vertex order `(x0,y0) (x1,y0) (x1,y1) (x0,y1)`.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn push_rect(
        &mut self,
        a: usize,
        b: usize,
        w: f64,
        r: [f64; 4],
        with_geom: bool,
        tf: Option<&ToWorld>,
    ) {
        self.head(a, b, w, KIND_GEOM);
        if with_geom {
            let [x0, y0, x1, y1] = r;
            let pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)];
            let mut buf = [0.0f64; 10];
            for (q, (x, y)) in pts.into_iter().enumerate() {
                let (x, y) = match tf {
                    Some(t) => t.apply(x, y),
                    None => (x, y),
                };
                buf[2 * q] = x;
                buf[2 * q + 1] = y;
            }
            self.coords.extend_from_slice(&buf);
            self.ring_len.push(5);
        }
    }
}
