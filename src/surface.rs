//! Fast inside / outside / ambiguous classification of boxes against a
//! polygonal surface, used to skip the GEOS `intersects` predicate for every
//! tile that is clearly inside or clearly outside the surface.
//!
//! The boundary edges are rasterised (dilated by one cell, so rounding can
//! never hide an edge) on a uniform grid. A box whose cells hold no edge lies
//! entirely on one side of the boundary; that side is the even-odd status of
//! any of its cells, computed once per row with a sorted crossing sweep. Boxes
//! touching an edge cell are reported as ambiguous and left to GEOS, so the
//! result is exactly the GEOS one (touching included).

use crate::geom::Polys;
use rayon::prelude::*;

pub const OUTSIDE: i8 = 0;
pub const INSIDE: i8 = 1;
pub const AMBIGUOUS: i8 = -1;

pub struct SurfaceGrid {
    x0: f64,
    y0: f64,
    inv: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    /// 0 = outside, 1 = inside, 2 = contains (or neighbours) a boundary edge
    status: Vec<u8>,
}

impl SurfaceGrid {
    pub fn build(s: &Polys, cell: f64) -> SurfaceGrid {
        let b = s.bounds();
        let (mut x0, mut y0, mut x1, mut y1) = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        for r in &b {
            x0 = x0.min(r[0]);
            y0 = y0.min(r[1]);
            x1 = x1.max(r[2]);
            y1 = y1.max(r[3]);
        }
        let mut cell = cell.max(1e-300);
        // keep the grid size reasonable (~16M cells max)
        while ((x1 - x0) / cell + 3.0) * ((y1 - y0) / cell + 3.0) > 16.0e6 {
            cell *= 2.0;
        }
        // one spare cell of margin on every side
        let (x0, y0) = (x0 - cell, y0 - cell);
        let nx = (((x1 - x0) / cell).floor() as usize + 2).max(1);
        let ny = (((y1 - y0) / cell).floor() as usize + 2).max(1);
        let mut g = SurfaceGrid {
            x0,
            y0,
            inv: 1.0 / cell,
            cell,
            nx,
            ny,
            status: vec![0; nx * ny],
        };
        // edges, rasterised column by column and dilated by one cell
        let mut edges: Vec<(f64, f64, f64, f64)> = Vec::new();
        for p in 0..s.n() {
            for r in s.poly_ring[p]..s.poly_ring[p + 1] {
                let (x, y) = s.ring(r);
                let m = x.len();
                for k in 0..m {
                    let k1 = if k + 1 == m { 0 } else { k + 1 };
                    edges.push((x[k], y[k], x[k1], y[k1]));
                }
            }
        }
        for &(ax, ay, bx, by) in &edges {
            let (lx, hx) = (ax.min(bx), ax.max(bx));
            let c0 = g.ci(lx);
            let c1 = g.ci(hx);
            for c in c0..=c1 {
                // y-range of the segment inside column c (clamped to the segment)
                let (cx0, cx1) = (g.x0 + c as f64 * cell, g.x0 + (c + 1) as f64 * cell);
                let (ya, yb) = if bx != ax {
                    let t0 = ((cx0.max(lx) - ax) / (bx - ax)).clamp(0.0, 1.0);
                    let t1 = ((cx1.min(hx) - ax) / (bx - ax)).clamp(0.0, 1.0);
                    (ay + t0 * (by - ay), ay + t1 * (by - ay))
                } else {
                    (ay, by)
                };
                let (r0, r1) = (g.ri(ya.min(yb)), g.ri(ya.max(yb)));
                for cc in c.saturating_sub(1)..=(c + 1).min(nx - 1) {
                    for rr in r0.saturating_sub(1)..=(r1 + 1).min(ny - 1) {
                        g.status[rr * nx + cc] = 2;
                    }
                }
            }
        }
        // edges bucketed by the rows whose centre line they may cross
        let mut by_row: Vec<Vec<u32>> = vec![Vec::new(); ny];
        for (k, &(_, ay, _, by)) in edges.iter().enumerate() {
            let (r0, r1) = (g.ri(ay.min(by)), g.ri(ay.max(by)));
            for row in &mut by_row[r0..=r1] {
                row.push(k as u32);
            }
        }
        // even-odd status of the edge-free cells, row by row
        let rows: Vec<Vec<u8>> = (0..ny)
            .into_par_iter()
            .map(|r| {
                let yc = g.y0 + (r as f64 + 0.5) * cell;
                let mut xs: Vec<f64> = by_row[r]
                    .iter()
                    .map(|&k| edges[k as usize])
                    .filter_map(|(ax, ay, bx, by)| {
                        // half-open rule: counts each vertex crossing once
                        if (ay <= yc) != (by <= yc) {
                            Some(ax + (yc - ay) * (bx - ax) / (by - ay))
                        } else {
                            None
                        }
                    })
                    .collect();
                xs.sort_unstable_by(|a, b| a.total_cmp(b));
                let mut row = g.status[r * nx..(r + 1) * nx].to_vec();
                let mut k = 0;
                for (c, v) in row.iter_mut().enumerate() {
                    let xc = g.x0 + (c as f64 + 0.5) * cell;
                    while k < xs.len() && xs[k] < xc {
                        k += 1;
                    }
                    if *v != 2 {
                        *v = (k % 2) as u8; // crossings on the left: odd -> inside
                    }
                }
                row
            })
            .collect();
        for (r, row) in rows.into_iter().enumerate() {
            g.status[r * nx..(r + 1) * nx].copy_from_slice(&row);
        }
        g
    }

    #[inline(always)]
    fn ci(&self, x: f64) -> usize {
        let v = ((x - self.x0) * self.inv).floor();
        if v.is_nan() || v < 0.0 {
            0
        } else {
            (v as usize).min(self.nx - 1)
        }
    }
    #[inline(always)]
    fn ri(&self, y: f64) -> usize {
        let v = ((y - self.y0) * self.inv).floor();
        if v.is_nan() || v < 0.0 {
            0
        } else {
            (v as usize).min(self.ny - 1)
        }
    }

    /// Classify one box `[x0, y0, x1, y1]`.
    pub fn classify(&self, b: [f64; 4]) -> i8 {
        if !(b.iter().all(|v| v.is_finite())) {
            return AMBIGUOUS;
        }
        let (gx1, gy1) = (
            self.x0 + self.nx as f64 * self.cell,
            self.y0 + self.ny as f64 * self.cell,
        );
        // entirely beyond the margin cells: cannot touch the surface
        if b[2] < self.x0 || b[0] > gx1 || b[3] < self.y0 || b[1] > gy1 {
            return OUTSIDE;
        }
        let (c0, c1, r0, r1) = (self.ci(b[0]), self.ci(b[2]), self.ri(b[1]), self.ri(b[3]));
        let first = self.status[r0 * self.nx + c0];
        if first == 2 {
            return AMBIGUOUS;
        }
        for r in r0..=r1 {
            for c in c0..=c1 {
                if self.status[r * self.nx + c] != first {
                    return AMBIGUOUS;
                }
            }
        }
        if first == 1 {
            INSIDE
        } else {
            OUTSIDE
        }
    }
}

/// Point-in-polygon test with exactly the semantics of matplotlib's
/// `Path(vertices).contains_points(points)` (radius 0, no transform) for a
/// single polygon of finite vertices: the Haines crossing test of
/// `matplotlib/src/_path.h::point_in_path_impl`, same comparisons and the same
/// floating-point expressions, so points on edges / vertices are classified
/// identically. Non-finite points are outside; fewer than 3 vertices -> none.
pub fn points_in_polygon(px: &[f64], py: &[f64], vx: &[f64], vy: &[f64]) -> Vec<bool> {
    let n = vx.len();
    if n < 3 {
        return vec![false; px.len()];
    }
    px.par_iter()
        .zip(py.par_iter())
        .with_min_len(1024)
        .map(|(&tx, &ty)| {
            if !(tx.is_finite() && ty.is_finite()) {
                return false;
            }
            let mut inside = false;
            // edges v[k-1] -> v[k] for k = 1..n, then the closing edge v[n-1] -> v[0]
            let mut yflag0 = vy[0] >= ty;
            for k in 1..=n {
                let (x0, y0) = (vx[k - 1], vy[k - 1]);
                let (x1, y1) = if k == n {
                    (vx[0], vy[0])
                } else {
                    (vx[k], vy[k])
                };
                let yflag1 = y1 >= ty;
                if yflag0 != yflag1 && (((y1 - ty) * (x0 - x1) >= (x1 - tx) * (y0 - y1)) == yflag1)
                {
                    inside = !inside;
                }
                yflag0 = yflag1;
            }
            inside
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_with_hole() {
        // 10x10 square with a 4x4 hole
        let s = Polys {
            x: vec![0., 10., 10., 0., 0., 3., 3., 7., 7., 3.],
            y: vec![0., 0., 10., 10., 0., 3., 7., 7., 3., 3.],
            ring_off: vec![0, 5, 10],
            poly_ring: vec![0, 2],
        };
        let g = SurfaceGrid::build(&s, 0.5);
        assert_eq!(g.classify([1.0, 8.0, 1.5, 8.5]), INSIDE);
        assert_eq!(g.classify([4.5, 4.5, 5.5, 5.5]), OUTSIDE); // in the hole
        assert_eq!(g.classify([20.0, 20.0, 21.0, 21.0]), OUTSIDE);
        assert_eq!(g.classify([9.5, 5.0, 10.5, 5.5]), AMBIGUOUS); // crosses the border
        assert_eq!(g.classify([10.0, 5.0, 11.0, 5.5]), AMBIGUOUS); // touches it
    }

    #[test]
    fn point_in_polygon_square() {
        let (vx, vy) = ([0.0, 2.0, 2.0, 0.0], [0.0, 0.0, 2.0, 2.0]);
        let r = points_in_polygon(&[1.0, 3.0, f64::NAN], &[1.0, 1.0, 1.0], &vx, &vy);
        assert_eq!(r, vec![true, false, false]);
        assert_eq!(
            points_in_polygon(&[1.0], &[1.0], &vx[..2], &vy[..2]),
            vec![false]
        );
    }
}
