//! Polygon meshes for VTK: vertex merging, surface cells and layered
//! extrusion to 3D cells.
//!
//! Cells are written in the legacy VTK cell-array layout
//! (`[n, id_0, ..., id_(n-1), n, ...]`, and for polyhedra
//! `[n_items, n_faces, n_0, ids..., n_1, ids...]`), which
//! `pyvista.UnstructuredGrid(cells, celltypes, points)` reads directly.

use rayon::prelude::*;

pub const VTK_TRIANGLE: u8 = 5;
pub const VTK_POLYGON: u8 = 7;
pub const VTK_PIXEL: u8 = 8;
pub const VTK_QUAD: u8 = 9;
pub const VTK_HEXAHEDRON: u8 = 12;
pub const VTK_WEDGE: u8 = 13;
pub const VTK_PENTAGONAL_PRISM: u8 = 15;
pub const VTK_HEXAGONAL_PRISM: u8 = 16;
pub const VTK_POLYHEDRON: u8 = 42;

/// Fill a ragged output in parallel: item `i` writes `sizes[i]` values.
fn fill_ragged<T, F>(sizes: &[usize], f: F) -> Vec<T>
where
    T: Copy + Default + Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let total: usize = sizes.iter().sum();
    let mut out = vec![T::default(); total];
    let mut parts: Vec<&mut [T]> = Vec::with_capacity(sizes.len());
    let mut rest = out.as_mut_slice();
    for &s in sizes {
        let (a, b) = rest.split_at_mut(s);
        parts.push(a);
        rest = b;
    }
    parts
        .into_par_iter()
        .enumerate()
        .with_min_len(1024)
        .for_each(|(i, p)| f(i, p));
    out
}

struct Dsu(Vec<u32>);

impl Dsu {
    fn find(&mut self, mut a: u32) -> u32 {
        while self.0[a as usize] != a {
            let g = self.0[self.0[a as usize] as usize];
            self.0[a as usize] = g;
            a = g;
        }
        a
    }
    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.0[hi as usize] = lo;
        }
    }
}

/// Single-linkage merge of 2D points: two points belong to the same cluster
/// when a chain of points at most `tol` apart joins them (the flat clusters of
/// `scipy.cluster.hierarchy.fcluster(linkage(pdist(xy)), tol, "distance")`).
/// Returns the cluster of every point (numbered by first occurrence) and the
/// mean position of every cluster (interleaved).
pub fn merge_points(xy: &[f64], tol: f64) -> (Vec<i64>, Vec<f64>) {
    let n = xy.len() / 2;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::MIN, f64::MIN);
    for &[x, y] in xy.as_chunks::<2>().0 {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    // Any cell size >= tol finds every pair (they lie in adjacent cells); the
    // floor keeps the keys within i64 whatever tol.
    let extent = (x1 - x0).max(y1 - y0);
    let cell = tol.max(extent * 1e-12).max(f64::MIN_POSITIVE);
    let inv = 1.0 / cell;
    let key = |i: usize| -> (i64, i64) {
        (
            ((xy[2 * i] - x0) * inv).floor() as i64,
            ((xy[2 * i + 1] - y0) * inv).floor() as i64,
        )
    };
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.par_sort_unstable_by_key(|&i| (key(i as usize), i));
    // runs of equal keys in `order`
    let mut runs: Vec<((i64, i64), usize, usize)> = Vec::new();
    let mut s = 0;
    while s < n {
        let k = key(order[s] as usize);
        let mut e = s + 1;
        while e < n && key(order[e] as usize) == k {
            e += 1;
        }
        runs.push((k, s, e));
        s = e;
    }
    let close = |a: u32, b: u32| {
        let (a, b) = (a as usize, b as usize);
        let (dx, dy) = (xy[2 * a] - xy[2 * b], xy[2 * a + 1] - xy[2 * b + 1]);
        (dx * dx + dy * dy).sqrt() <= tol
    };
    // forward half of the 3 x 3 neighbourhood (each pair of cells seen once)
    const NB: [(i64, i64); 4] = [(0, 1), (1, -1), (1, 0), (1, 1)];
    let pairs: Vec<(u32, u32)> = runs
        .par_iter()
        .with_min_len(1024)
        .flat_map_iter(|&((kx, ky), s, e)| {
            let mut v = Vec::new();
            let own = &order[s..e];
            for (i, &a) in own.iter().enumerate() {
                for &b in &own[i + 1..] {
                    if close(a, b) {
                        v.push((a, b));
                    }
                }
            }
            for (dx, dy) in NB {
                let k = (kx + dx, ky + dy);
                if let Ok(r) = runs.binary_search_by_key(&k, |r| r.0) {
                    let (_, s2, e2) = runs[r];
                    for &a in own {
                        for &b in &order[s2..e2] {
                            if close(a, b) {
                                v.push((a, b));
                            }
                        }
                    }
                }
            }
            v
        })
        .collect();
    let mut dsu = Dsu((0..n as u32).collect());
    for (a, b) in pairs {
        dsu.union(a, b);
    }
    let mut label_of_root = vec![-1i64; n];
    let mut labels = vec![0i64; n];
    let mut n_lab = 0i64;
    for (i, l) in labels.iter_mut().enumerate() {
        let r = dsu.find(i as u32) as usize;
        if label_of_root[r] < 0 {
            label_of_root[r] = n_lab;
            n_lab += 1;
        }
        *l = label_of_root[r];
    }
    let mut sums = vec![0.0f64; 2 * n_lab as usize];
    let mut cnt = vec![0u32; n_lab as usize];
    for (i, &l) in labels.iter().enumerate() {
        let l = l as usize;
        sums[2 * l] += xy[2 * i];
        sums[2 * l + 1] += xy[2 * i + 1];
        cnt[l] += 1;
    }
    for (l, &c) in cnt.iter().enumerate() {
        sums[2 * l] /= c as f64;
        sums[2 * l + 1] /= c as f64;
    }
    (labels, sums)
}

/// Twice the signed area of the ring `ids` (indices into interleaved `pts`
/// with stride `dim`), relative to its first vertex.
#[inline]
fn signed_area2(ids: &[i64], pts: &[f64], dim: usize) -> f64 {
    let p = |k: usize| {
        let i = ids[k] as usize * dim;
        (pts[i], pts[i + 1])
    };
    let (x0, y0) = p(0);
    let mut a = 0.0;
    for k in 1..ids.len().saturating_sub(1) {
        let (xa, ya) = p(k);
        let (xb, yb) = p(k + 1);
        a += (xa - x0) * (yb - y0) - (xb - x0) * (ya - y0);
    }
    a
}

/// Ring of merged ids without consecutive repeats (wrap-around included).
fn dedup_ring(ids: impl Iterator<Item = i64>, out: &mut Vec<i64>) {
    out.clear();
    for i in ids {
        if out.last() != Some(&i) {
            out.push(i);
        }
    }
    while out.len() > 1 && out.first() == out.last() {
        out.pop();
    }
}

/// Polygon cells (legacy layout) from polygons given by vertex `labels`:
/// polygon `p` uses labels `off[p]..off[p + 1]` (open rings). Repeated
/// consecutive labels (merged vertices) are dropped and every ring is made
/// counter-clockwise. Fails if a ring keeps fewer than 3 vertices.
pub fn polygon_cells(off: &[usize], labels: &[i64], pts: &[f64]) -> Result<Vec<i64>, usize> {
    let np = off.len() - 1;
    let ring = |p: usize, buf: &mut Vec<i64>| {
        dedup_ring(labels[off[p]..off[p + 1]].iter().copied(), buf);
    };
    let sizes: Vec<usize> = (0..np)
        .into_par_iter()
        .with_min_len(1024)
        .map_init(Vec::new, |buf, p| {
            ring(p, buf);
            buf.len()
        })
        .collect();
    if let Some(p) = sizes.iter().position(|&s| s < 3) {
        return Err(p);
    }
    let sizes1: Vec<usize> = sizes.iter().map(|s| s + 1).collect();
    Ok(fill_ragged(&sizes1, |p, out| {
        let mut buf = Vec::with_capacity(out.len());
        ring(p, &mut buf);
        if signed_area2(&buf, pts, 2) < 0.0 {
            buf.reverse();
        }
        out[0] = buf.len() as i64;
        out[1..].copy_from_slice(&buf);
    }))
}

/// Number of legacy cell-array values of a 3D cell extruded from an
/// `n`-vertex polygon.
#[inline]
pub fn extruded_size(n: usize) -> usize {
    if n <= 6 {
        1 + 2 * n
    } else {
        2 + (n + 1) * 2 + n * 5
    }
}

#[inline]
fn extruded_type(n: usize) -> u8 {
    match n {
        3 => VTK_WEDGE,
        4 => VTK_HEXAHEDRON,
        5 => VTK_PENTAGONAL_PRISM,
        6 => VTK_HEXAGONAL_PRISM,
        _ => VTK_POLYHEDRON,
    }
}

pub struct Extruded {
    pub cells: Vec<i64>,
    pub types: Vec<u8>,
    /// horizontal area of every 2D cell
    pub area: Vec<f64>,
}

/// Extrude the 2D cells of a surface (legacy `cells`, `types`, points
/// `(n_pts, 3)` interleaved) into `n_layers` layers of 3D cells. Layer `k`
/// of the points is `n_pts * k ..`; `downward` means that layer `k + 1` lies
/// below layer `k`. Output cell `i * n_layers + j` is surface cell `i` in
/// layer `j`. Rings are oriented so that every cell has a positive volume
/// and outward-facing polyhedron faces.
pub fn extrude(
    cells: &[i64],
    types: &[u8],
    pts: &[f64],
    n_layers: usize,
    downward: bool,
) -> Result<Extruded, String> {
    let n_pts = pts.len() / 3;
    // locate the cells
    let mut start = Vec::with_capacity(types.len());
    let mut k = 0usize;
    for (c, &t) in types.iter().enumerate() {
        if k >= cells.len() {
            return Err("cell array shorter than the cell types".into());
        }
        let n = cells[k];
        if n < 0 || k + 1 + n as usize > cells.len() {
            return Err("invalid cell array".into());
        }
        let n = n as usize;
        match t {
            VTK_TRIANGLE | VTK_POLYGON | VTK_QUAD => {}
            VTK_PIXEL if n == 4 => {}
            _ => return Err(format!("cell {c} has VTK type {t}: only 2D cells (triangle, quad, pixel, polygon) can be extruded")),
        }
        if n < 3 {
            return Err(format!("cell {c} has {n} points: at least 3 are needed"));
        }
        if cells[k + 1..k + 1 + n]
            .iter()
            .any(|&i| i < 0 || i as usize >= n_pts)
        {
            return Err(format!("cell {c} references a missing point"));
        }
        start.push(k + 1);
        k += 1 + n;
    }
    if k != cells.len() {
        return Err("cell array longer than the cell types".into());
    }
    let nc = types.len();
    let ring_of = |c: usize| -> Vec<i64> {
        let s = start[c];
        let n = cells[s - 1] as usize;
        let ids = &cells[s..s + n];
        let mut r: Vec<i64> = if types[c] == VTK_PIXEL {
            vec![ids[0], ids[1], ids[3], ids[2]] // pixel order -> ring order
        } else {
            ids.to_vec()
        };
        if signed_area2(&r, pts, 3) < 0.0 {
            r.reverse();
        }
        r
    };
    let n_of = |c: usize| cells[start[c] - 1] as usize;
    let area: Vec<f64> = (0..nc)
        .into_par_iter()
        .with_min_len(1024)
        .map(|c| 0.5 * signed_area2(&ring_of(c), pts, 3).abs())
        .collect();
    let sizes: Vec<usize> = (0..nc).map(|c| extruded_size(n_of(c)) * n_layers).collect();
    let np = n_pts as i64;
    let out = fill_ragged(&sizes, |c, out| {
        let r = ring_of(c);
        let n = r.len();
        let sz = extruded_size(n);
        for j in 0..n_layers {
            let (lo_l, up_l) = if downward { (j + 1, j) } else { (j, j + 1) };
            let (lo, up) = (lo_l as i64 * np, up_l as i64 * np);
            let o = &mut out[j * sz..(j + 1) * sz];
            if n <= 6 {
                // prism: lower ring (counter-clockwise seen from above, i.e.
                // normal towards the upper ring), then the upper ring
                o[0] = 2 * n as i64;
                for (q, &v) in r.iter().enumerate() {
                    o[1 + q] = v + lo;
                    o[1 + n + q] = v + up;
                }
            } else {
                o[0] = (sz - 1) as i64;
                o[1] = n as i64 + 2;
                let mut w = 2;
                o[w] = n as i64; // upper face, normal up
                for (q, &v) in r.iter().enumerate() {
                    o[w + 1 + q] = v + up;
                }
                w += 1 + n;
                for q in 0..n {
                    let (a, b) = (r[q], r[(q + 1) % n]);
                    o[w..w + 5].copy_from_slice(&[4, a + lo, b + lo, b + up, a + up]);
                    w += 5;
                }
                o[w] = n as i64; // lower face, normal down
                for (q, &v) in r.iter().rev().enumerate() {
                    o[w + 1 + q] = v + lo;
                }
            }
        }
    });
    let mut types3 = Vec::with_capacity(nc * n_layers);
    for c in 0..nc {
        let t = extruded_type(n_of(c));
        types3.extend(std::iter::repeat_n(t, n_layers));
    }
    Ok(Extruded {
        cells: out,
        types: types3,
        area,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_chains_and_tolerance() {
        // 0-1 close, 1-2 close (0-2 not): single linkage joins all three
        let xy = [0.0, 0.0, 0.9, 0.0, 1.8, 0.0, 5.0, 5.0, 5.0, 5.0];
        let (l, m) = merge_points(&xy, 1.0);
        assert_eq!(l, vec![0, 0, 0, 1, 1]);
        assert!((m[0] - 0.9).abs() < 1e-15 && m[2] == 5.0);
        let (l, _) = merge_points(&xy, 0.0);
        assert_eq!(l, vec![0, 1, 2, 3, 3]);
    }

    #[test]
    fn polygon_cells_orient_and_dedup() {
        // clockwise square with a repeated vertex
        let pts = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let labels = [0, 1, 1, 2, 3, 0];
        let c = polygon_cells(&[0, 6], &labels, &pts).unwrap();
        assert_eq!(c, vec![4, 3, 2, 1, 0]);
        assert_eq!(polygon_cells(&[0, 3], &[0, 0, 1], &pts), Err(0));
    }

    #[test]
    fn extrude_sizes() {
        let pts: Vec<f64> = (0..7)
            .flat_map(|k| {
                let t = std::f64::consts::TAU * k as f64 / 7.0;
                [t.cos(), t.sin(), 0.0]
            })
            .collect();
        let cells = [7, 0, 1, 2, 3, 4, 5, 6];
        let e = extrude(&cells, &[VTK_POLYGON], &pts, 2, false).unwrap();
        assert_eq!(e.cells.len(), 2 * extruded_size(7));
        assert_eq!(e.types, vec![VTK_POLYHEDRON; 2]);
        assert_eq!(e.cells[0] as usize, extruded_size(7) - 1);
        assert!(extrude(&cells, &[3], &pts, 1, false).is_err());
    }
}
