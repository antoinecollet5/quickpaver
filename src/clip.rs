//! Sutherland-Hodgman clipping and shoelace areas.
//!
//! The arithmetic (comparisons, interpolation formula, shoelace summation order)
//! is the one of quickpaver's original kernels, so convex clips are bitwise
//! reproducible. The code is generic over a vertex buffer:
//!
//! * [`Poly`]: fixed-size stack buffer, used for convex subjects whose output
//!   size is bounded (`n_in + n_window_edges`);
//! * [`VecPoly`]: growable buffer, used for arbitrary (concave, large) subject
//!   rings. Clipping a concave ring against a *convex* window can produce
//!   degenerate zero-width bridges, but the shoelace area stays exact.

/// Vertex buffer used by the clip passes.
pub trait Buf {
    fn clear(&mut self);
    fn push(&mut self, x: f64, y: f64);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn xs(&self) -> &[f64];
    fn ys(&self) -> &[f64];
    fn copy_from(&mut self, other: &Self);
    /// Writable buffers able to hold at least `cap` vertices (contents
    /// unspecified); the caller then calls [`Buf::set_len`].
    fn write_bufs(&mut self, cap: usize) -> (&mut [f64], &mut [f64]);
    fn set_len(&mut self, n: usize);
}

/// Stack buffer of capacity `M`.
#[derive(Clone, Copy)]
pub struct Poly<const M: usize> {
    pub x: [f64; M],
    pub y: [f64; M],
    pub n: usize,
}

impl<const M: usize> Default for Poly<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const M: usize> Poly<M> {
    #[inline(always)]
    pub fn new() -> Self {
        Poly {
            x: [0.0; M],
            y: [0.0; M],
            n: 0,
        }
    }
    #[inline(always)]
    pub fn load(&mut self, x: &[f64], y: &[f64]) {
        let n = x.len();
        self.x[..n].copy_from_slice(x);
        self.y[..n].copy_from_slice(&y[..n]);
        self.n = n;
    }
}

impl<const M: usize> Buf for Poly<M> {
    #[inline(always)]
    fn clear(&mut self) {
        self.n = 0;
    }
    #[inline(always)]
    fn push(&mut self, x: f64, y: f64) {
        self.x[self.n] = x;
        self.y[self.n] = y;
        self.n += 1;
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }
    #[inline(always)]
    fn xs(&self) -> &[f64] {
        &self.x[..self.n]
    }
    #[inline(always)]
    fn ys(&self) -> &[f64] {
        &self.y[..self.n]
    }
    #[inline(always)]
    fn copy_from(&mut self, o: &Self) {
        let n = o.n;
        self.x[..n].copy_from_slice(&o.x[..n]);
        self.y[..n].copy_from_slice(&o.y[..n]);
        self.n = n;
    }
    #[inline(always)]
    fn write_bufs(&mut self, _cap: usize) -> (&mut [f64], &mut [f64]) {
        // convex subjects never exceed M (checked by the indexing anyway)
        (&mut self.x[..], &mut self.y[..])
    }
    #[inline(always)]
    fn set_len(&mut self, n: usize) {
        self.n = n;
    }
}

/// Growable buffer (reused across calls, so it allocates only while growing).
#[derive(Default, Clone)]
pub struct VecPoly {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

impl VecPoly {
    pub fn load(&mut self, x: &[f64], y: &[f64]) {
        self.x.clear();
        self.y.clear();
        self.x.extend_from_slice(x);
        self.y.extend_from_slice(y);
    }
}

impl Buf for VecPoly {
    #[inline(always)]
    fn clear(&mut self) {
        self.x.clear();
        self.y.clear();
    }
    #[inline(always)]
    fn push(&mut self, x: f64, y: f64) {
        self.x.push(x);
        self.y.push(y);
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.x.len()
    }
    #[inline(always)]
    fn xs(&self) -> &[f64] {
        &self.x
    }
    #[inline(always)]
    fn ys(&self) -> &[f64] {
        &self.y
    }
    fn copy_from(&mut self, o: &Self) {
        self.load(&o.x, &o.y);
    }
    #[inline(always)]
    fn write_bufs(&mut self, cap: usize) -> (&mut [f64], &mut [f64]) {
        if self.x.len() < cap {
            self.x.resize(cap, 0.0);
            self.y.resize(cap, 0.0);
        }
        (&mut self.x[..], &mut self.y[..])
    }
    #[inline(always)]
    fn set_len(&mut self, n: usize) {
        self.x.truncate(n);
        self.y.truncate(n);
    }
}

/// Shoelace area (absolute value), same summation order as the reference.
#[inline(always)]
pub fn area<B: Buf>(p: &B) -> f64 {
    let (x, y) = (p.xs(), p.ys());
    let n = x.len();
    let mut a = 0.0;
    for i in 0..n {
        let j = if i + 1 == n { 0 } else { i + 1 };
        a += x[i] * y[j] - x[j] * y[i];
    }
    a.abs() * 0.5
}

/// One pass against an axis-aligned half-plane.
/// AXIS: 0 = x, 1 = y. GE: true keeps `coord >= ev`, false keeps `coord <= ev`.
/// Every input vertex emits at most two output vertices, hence `cap = 2n`.
#[inline(always)]
fn axis_pass<B: Buf, const AXIS: usize, const GE: bool>(a: &B, b: &mut B, ev: f64) {
    let (xs, ys) = (a.xs(), a.ys());
    let n = xs.len();
    let (bx, by) = b.write_bufs(2 * n);
    let cs = if AXIS == 0 { xs } else { ys };
    let mut k = 0usize;
    let mut pi = n - 1;
    for i in 0..n {
        let cc = cs[i];
        let pc = cs[pi];
        let (c_in, p_in) = if GE {
            (cc >= ev, pc >= ev)
        } else {
            (cc <= ev, pc <= ev)
        };
        if p_in != c_in {
            let d = cc - pc;
            let t = if d != 0.0 { (ev - pc) / d } else { 0.0 };
            bx[k] = xs[pi] + t * (xs[i] - xs[pi]);
            by[k] = ys[pi] + t * (ys[i] - ys[pi]);
            k += 1;
        }
        if c_in {
            bx[k] = xs[i];
            by[k] = ys[i];
            k += 1;
        }
        pi = i;
    }
    b.set_len(k);
}

/// Clip `p` against the axis-aligned rectangle; the result ends in `p`
/// (empty when fewer than 3 vertices remain). `q` is scratch space.
#[inline(always)]
pub fn clip_rect<B: Buf>(p: &mut B, q: &mut B, xmin: f64, ymin: f64, xmax: f64, ymax: f64) {
    if p.len() < 3 {
        p.clear();
        return;
    }
    axis_pass::<B, 0, true>(p, q, xmin);
    if q.len() < 3 {
        p.clear();
        return;
    }
    axis_pass::<B, 0, false>(q, p, xmax);
    if p.len() < 3 {
        p.clear();
        return;
    }
    axis_pass::<B, 1, true>(p, q, ymin);
    if q.len() < 3 {
        p.clear();
        return;
    }
    axis_pass::<B, 1, false>(q, p, ymax);
    if p.len() < 3 {
        p.clear();
    }
}

/// One pass against the half-plane left of edge (ex0,ey0)->(ex0+edx,ey0+edy).
#[inline(always)]
fn halfplane_pass<B: Buf>(a: &B, b: &mut B, ex0: f64, ey0: f64, edx: f64, edy: f64) {
    let (xs, ys) = (a.xs(), a.ys());
    let n = xs.len();
    let (bx, by) = b.write_bufs(2 * n);
    let mut k = 0usize;
    let mut pi = n - 1;
    let mut p_side = edx * (ys[pi] - ey0) - edy * (xs[pi] - ex0);
    for i in 0..n {
        let c_side = edx * (ys[i] - ey0) - edy * (xs[i] - ex0);
        let c_in = c_side >= 0.0;
        let p_in = p_side >= 0.0;
        if p_in != c_in {
            let dxp = xs[i] - xs[pi];
            let dyp = ys[i] - ys[pi];
            let denom = edx * dyp - edy * dxp;
            let t = if denom == 0.0 { 0.0 } else { p_side / -denom };
            bx[k] = xs[pi] + t * dxp;
            by[k] = ys[pi] + t * dyp;
            k += 1;
        }
        if c_in {
            bx[k] = xs[i];
            by[k] = ys[i];
            k += 1;
        }
        pi = i;
        p_side = c_side;
    }
    b.set_len(k);
}

/// Clip `p` against a CCW convex window; the result ends in `p`.
#[inline(always)]
pub fn clip_convex<B: Buf>(p: &mut B, q: &mut B, wx: &[f64], wy: &[f64]) {
    let m = wx.len();
    let mut in_p = true;
    for e in 0..m {
        let cur = if in_p { p.len() } else { q.len() };
        if cur < 3 {
            p.clear();
            return;
        }
        let e1 = if e + 1 == m { 0 } else { e + 1 };
        let (ex0, ey0) = (wx[e], wy[e]);
        let (edx, edy) = (wx[e1] - ex0, wy[e1] - ey0);
        if in_p {
            halfplane_pass(p, q, ex0, ey0, edx, edy);
        } else {
            halfplane_pass(q, p, ex0, ey0, edx, edy);
        }
        in_p = !in_p;
    }
    if !in_p {
        p.copy_from(q);
    }
    if p.len() < 3 {
        p.clear();
    }
}

/// Signed (CCW > 0) doubled area of a ring.
#[inline(always)]
pub fn signed_area2(x: &[f64], y: &[f64]) -> f64 {
    let m = x.len();
    let mut s = 0.0;
    for k in 0..m {
        let k1 = if k + 1 == m { 0 } else { k + 1 };
        s += x[k] * y[k1] - x[k1] * y[k];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(x0: f64, y0: f64, s: f64) -> (Vec<f64>, Vec<f64>) {
        (vec![x0, x0 + s, x0 + s, x0], vec![y0, y0, y0 + s, y0 + s])
    }

    #[test]
    fn rect_clip_of_overlapping_squares() {
        let (x, y) = sq(0.5, 0.5, 1.0);
        let mut p = Poly::<12>::new();
        let mut q = Poly::<12>::new();
        p.load(&x, &y);
        clip_rect(&mut p, &mut q, 0.0, 0.0, 1.0, 1.0);
        assert_eq!(p.len(), 4);
        assert!((area(&p) - 0.25).abs() < 1e-15);
    }

    #[test]
    fn disjoint_gives_empty() {
        let (x, y) = sq(5.0, 5.0, 1.0);
        let mut p = Poly::<12>::new();
        let mut q = Poly::<12>::new();
        p.load(&x, &y);
        clip_rect(&mut p, &mut q, 0.0, 0.0, 1.0, 1.0);
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn concave_subject_against_rect_has_exact_area() {
        // L-shape of area 3 (unit squares), clipped by [0,2]x[0,1] -> area 2
        let x = vec![0.0, 2.0, 2.0, 1.0, 1.0, 0.0];
        let y = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let mut p = VecPoly::default();
        let mut q = VecPoly::default();
        p.load(&x, &y);
        assert!((area(&p) - 3.0).abs() < 1e-15);
        clip_rect(&mut p, &mut q, 0.0, 0.0, 2.0, 1.0);
        assert!((area(&p) - 2.0).abs() < 1e-15);
        // U-shape crossing the window twice: bridges have zero area
        let x = vec![0.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 0.0];
        let y = vec![0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];
        p.load(&x, &y);
        clip_rect(&mut p, &mut q, -1.0, 1.5, 4.0, 3.0);
        assert!((area(&p) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn triangle_window() {
        let (x, y) = sq(0.0, 0.0, 1.0);
        let mut p = Poly::<12>::new();
        let mut q = Poly::<12>::new();
        p.load(&x, &y);
        clip_convex(&mut p, &mut q, &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]);
        assert!((area(&p) - 0.5).abs() < 1e-15);
        assert!(signed_area2(&[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]) > 0.0);
    }
}
