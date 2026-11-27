//! Variable-radius Poisson disk sampling (Bridson 2007) on a pixel radius map.
//!
//! Two accepted points `i`, `j` always satisfy `|p_i - p_j| >= r_i + r_j`,
//! with `r` read from the radius map at each point's pixel. New candidates are
//! drawn on the circle of radius `2 r_p` around a random active point `p` (or
//! `r_p + r_c` where the radius grows, `r_c` being the candidate's radius),
//! `k` times before that point is retired. When no active point is left, a fill
//! pass seeds the next pixel centre (raster order) that can still host a point
//! and growth resumes, so the final set is maximal: regions that growth cannot
//! reach (across a sharp radius jump) are covered too.
//!
//! Random numbers come from xoshiro256++ seeded through SplitMix64; points and
//! radii are stored in float32. Candidate directions are 24-bit fractions of a
//! turn whose cosine / sine are evaluated with basic IEEE operations only (no
//! libm), so a seed gives bit-identical points on every platform.
//!
//! Neighbour search: points are stored in one uniform hash grid per radius
//! octave (`[r_min 2^L, r_min 2^(L+1))`), with a cell size proportional to the
//! octave, so a query visits a handful of cells whatever the radius contrast.
//! The search window `r_c + r_j` is bounded with a min / max pyramid of the
//! radius map around the candidate, which keeps it tight in smooth fields and
//! lets the query skip octaves that cannot occur in the neighbourhood.

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
/// Fixed-point steps lengthening a candidate step uphill (see `run`).
const MAX_STEP_ITER: usize = 4;
const CELL_FACTOR: f64 = 2.0 * std::f64::consts::SQRT_2;

#[inline(always)]
fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256++, seeded with SplitMix64 (standard practice for this family).
pub struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    pub fn new(seed: u64) -> Self {
        let mut s = [0u64; 4];
        let mut sm = seed;
        for w in s.iter_mut() {
            sm = sm.wrapping_add(GOLDEN);
            *w = splitmix64(sm);
        }
        Xoshiro256pp { s }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let [s0, s1, s2, s3] = self.s;
        let result = s0.wrapping_add(s3).rotate_left(23).wrapping_add(s0);
        let t = s1 << 17;
        let s2 = s2 ^ s0;
        let s3 = s3 ^ s1;
        let s1 = s1 ^ s2;
        let s0 = s0 ^ s3;
        let s2 = s2 ^ t;
        let s3 = s3.rotate_left(45);
        self.s = [s0, s1, s2, s3];
        result
    }

    /// Top 24 bits of the next draw.
    #[inline(always)]
    pub fn next_u24(&mut self) -> u32 {
        (self.next_u64() >> 40) as u32
    }

    /// Uniform float32 in `[0, 1)` from the top 24 bits.
    #[inline(always)]
    pub fn uniform(&mut self) -> f32 {
        self.next_u24() as f32 / (1u32 << 24) as f32
    }
}

/// `(cos, sin)` of `2 pi m / 2^24` for a 24-bit integer `m`, with basic IEEE
/// operations only (exact octant reduction, then Taylor polynomials on
/// `[0, pi/4]`, truncation error < 1e-19): the same bits on every platform,
/// unlike the libm `cos` / `sin`, and several times faster.
#[inline(always)]
pub fn turn_cos_sin(m: u32) -> (f64, f64) {
    const STEP: f64 = std::f64::consts::TAU / (1u64 << 24) as f64;
    const OCT: u32 = 1 << 21; // 1/8 turn
    let oct = (m >> 21) & 7;
    let rem = m & (OCT - 1);
    // fold to [0, pi/4]: odd octants are measured from the next octant boundary
    let r = if oct & 1 == 1 { OCT - rem } else { rem };
    let t = r as f64 * STEP;
    let t2 = t * t;
    let s = t
        * (1.0
            + t2 * (-1.0 / 6.0
                + t2 * (1.0 / 120.0
                    + t2 * (-1.0 / 5040.0
                        + t2 * (1.0 / 362880.0
                            + t2 * (-1.0 / 39916800.0
                                + t2 * (1.0 / 6227020800.0
                                    + t2 * (-1.0 / 1307674368000.0
                                        + t2 * (1.0 / 355687428096000.0)))))))));
    let c = 1.0
        + t2 * (-0.5
            + t2 * (1.0 / 24.0
                + t2 * (-1.0 / 720.0
                    + t2 * (1.0 / 40320.0
                        + t2 * (-1.0 / 3628800.0
                            + t2 * (1.0 / 479001600.0
                                + t2 * (-1.0 / 87178291200.0
                                    + t2 * (1.0 / 20922789888000.0
                                        + t2 * (-1.0 / 6402373705728000.0)))))))));
    // angle = oct * pi/4 + (rem or pi/4 - r'); map (c, s) of the folded angle
    match oct {
        0 => (c, s),
        1 => (s, c),
        2 => (-s, c),
        3 => (-c, s),
        4 => (-c, -s),
        5 => (-s, -c),
        6 => (s, -c),
        _ => (c, -s),
    }
}

/// `trunc(v)` as an index for `v >= 0` (`floor` for non-negative values);
/// negative or NaN values give 0. Much cheaper than `f64 as usize` (which is
/// a saturating unsigned conversion) and `f64::floor` (a libm call on
/// baseline x86-64).
#[inline(always)]
fn ix(v: f64) -> usize {
    // clamp to [0, 2^52] without NaN bookkeeping (NaN -> 0), then truncate
    let v = if v > 0.0 { v } else { 0.0 };
    let v = if v < 4.5e15 { v } else { 4.5e15 };
    // SAFETY: v is finite and in [0, 4.5e15], representable as an i64
    unsafe { v.to_int_unchecked::<i64>() as usize }
}

/// `min` / `max` without NaN handling (the radii are validated finite):
/// plain comparisons, which vectorise.
#[inline(always)]
fn fmin(a: f32, b: f32) -> f32 {
    if b < a {
        b
    } else {
        a
    }
}
#[inline(always)]
fn fmax(a: f32, b: f32) -> f32 {
    if b > a {
        b
    } else {
        a
    }
}

/// Radius map, row-major `(h, w)`.
struct Field<'a> {
    r: &'a [f32],
    w: usize,
    h: usize,
}

impl Field<'_> {
    /// Radius at the pixel of a (non-negative) position, clamped to the map.
    #[inline(always)]
    fn at(&self, x: f64, y: f64) -> f32 {
        let (i, j) = (ix(y).min(self.h - 1), ix(x).min(self.w - 1));
        self.r[i * self.w + j]
    }
}

/// Min / max pyramid of the radius map (blocks of `B0 << level` pixels).
struct Pyramid {
    levels: Vec<(usize, usize, Vec<f32>, Vec<f32>)>,
}

const B0: usize = 8;

impl Pyramid {
    fn build(f: &Field<'_>) -> Pyramid {
        let (nx, ny) = (f.w.div_ceil(B0), f.h.div_ceil(B0));
        let mut mn = vec![f32::INFINITY; nx * ny];
        let mut mx = vec![f32::NEG_INFINITY; nx * ny];
        for i in 0..f.h {
            let row = &f.r[i * f.w..(i + 1) * f.w];
            let base = (i / B0) * nx;
            for (bj, chunk) in row.chunks(B0).enumerate() {
                let (a, b) = chunk
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                        (fmin(a, v), fmax(b, v))
                    });
                mn[base + bj] = fmin(mn[base + bj], a);
                mx[base + bj] = fmax(mx[base + bj], b);
            }
        }
        let mut levels = vec![(nx, ny, mn, mx)];
        while {
            let l = levels.last().unwrap();
            l.0 > 1 || l.1 > 1
        } {
            let (pnx, pny, pmn, pmx) = levels.last().unwrap();
            let (nx, ny) = (pnx.div_ceil(2), pny.div_ceil(2));
            let mut mn = vec![f32::INFINITY; nx * ny];
            let mut mx = vec![f32::NEG_INFINITY; nx * ny];
            for i in 0..*pny {
                for j in 0..*pnx {
                    let (s, d) = (i * pnx + j, (i / 2) * nx + j / 2);
                    mn[d] = fmin(mn[d], pmn[s]);
                    mx[d] = fmax(mx[d], pmx[s]);
                }
            }
            levels.push((nx, ny, mn, mx));
        }
        Pyramid { levels }
    }

    /// Bounds `(min, max)` of the radius over the pixel box `[x0, x1] x [y0, y1]`
    /// (inclusive), read on the coarsest level needing at most 3 x 3 blocks.
    #[inline]
    fn bounds(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> (f32, f32) {
        let span = (x1 - x0).max(y1 - y0) + 1;
        let mut lv = 0;
        while lv + 1 < self.levels.len() && (B0 << lv) * 2 < span {
            lv += 1;
        }
        let (nx, _, mn, mx) = &self.levels[lv];
        let b = B0 << lv;
        let (mut a, mut z) = (f32::INFINITY, f32::NEG_INFINITY);
        for bi in y0 / b..=y1 / b {
            for bj in x0 / b..=x1 / b {
                a = fmin(a, mn[bi * nx + bj]);
                z = fmax(z, mx[bi * nx + bj]);
            }
        }
        (a, z)
    }
}

#[derive(Clone, Copy)]
struct Particle {
    x: f32,
    y: f32,
    r: f32,
    next: i32,
}

/// Uniform hash grid holding the points of one radius octave.
struct Octave {
    inv: f64,
    nx: usize,
    ny: usize,
    head: Vec<i32>,
    rmin: f32,
    rmax: f32,
}

impl Octave {
    fn new(cell: f64, w: usize, h: usize) -> Octave {
        let (nx, ny) = (
            (w as f64 / cell) as usize + 1,
            (h as f64 / cell) as usize + 1,
        );
        Octave {
            inv: 1.0 / cell,
            nx,
            ny,
            head: Vec::new(), // allocated on first insertion
            rmin: f32::INFINITY,
            rmax: f32::NEG_INFINITY,
        }
    }

    #[inline(always)]
    fn cell_range(&self, lo: f64, hi: f64, n: usize) -> (usize, usize) {
        (ix(lo * self.inv).min(n - 1), ix(hi * self.inv).min(n - 1))
    }
}

pub struct Sampler<'a> {
    f: Field<'a>,
    pyr: Option<Pyramid>,
    r_lo: f32,
    r_hi: f32,
    octaves: Vec<Octave>,
    parts: Vec<Particle>,
}

impl<'a> Sampler<'a> {
    pub fn new(r: &'a [f32], h: usize, w: usize) -> Result<Sampler<'a>, String> {
        if h == 0 || w == 0 || r.len() != h * w {
            return Err("radius must be a non-empty 2D array".into());
        }
        // one vectorisable pass: validity (NaN fails both tests) and range
        let (ok, lo, hi) = r.chunks(4096).fold(
            (true, f32::INFINITY, f32::NEG_INFINITY),
            |(ok, lo, hi), c| {
                let ok = ok & c.iter().fold(true, |a, &v| a & (v > 0.0) & (v <= f32::MAX));
                let (lo, hi) = c
                    .iter()
                    .fold((lo, hi), |(a, b), &v| (fmin(a, v), fmax(b, v)));
                (ok, lo, hi)
            },
        );
        if !ok {
            return Err("radius must be finite and > 0 everywhere".into());
        }
        let f = Field { r, w, h };
        let n_oct = ((hi as f64 / lo as f64).log2().floor() as usize + 1).clamp(1, 64);
        // One point per cell at most in theory (spacing >= 2 r_lo 2^L = cell
        // diagonal); cells are never smaller than half a pixel.
        let octaves = (0..n_oct)
            .map(|l| {
                Octave::new(
                    (lo as f64 * 2f64.powi(l as i32) * CELL_FACTOR).max(0.5),
                    w,
                    h,
                )
            })
            .collect();
        let pyr = (hi > lo).then(|| Pyramid::build(&f));
        Ok(Sampler {
            f,
            pyr,
            r_lo: lo,
            r_hi: hi,
            octaves,
            parts: Vec::new(),
        })
    }

    #[inline(always)]
    fn octave_of(&self, r: f32) -> usize {
        let n = self.octaves.len();
        let mut l = ((r as f64 / self.r_lo as f64).log2().floor().max(0.0) as usize).min(n - 1);
        // guard against log2 rounding at octave boundaries
        while l > 0 && (r as f64) < self.r_lo as f64 * 2f64.powi(l as i32) {
            l -= 1;
        }
        while l + 1 < n && (r as f64) >= self.r_lo as f64 * 2f64.powi(l as i32 + 1) {
            l += 1;
        }
        l
    }

    fn insert(&mut self, x: f32, y: f32) {
        let r = self.f.at(x as f64, y as f64);
        let l = self.octave_of(r);
        let id = self.parts.len() as i32;
        let o = &mut self.octaves[l];
        if o.head.is_empty() {
            o.head = vec![-1; o.nx * o.ny];
        }
        let cx = ix(x as f64 * o.inv).min(o.nx - 1);
        let cy = ix(y as f64 * o.inv).min(o.ny - 1);
        let c = cy * o.nx + cx;
        self.parts.push(Particle {
            x,
            y,
            r,
            next: o.head[c],
        });
        o.head[c] = id;
        o.rmin = fmin(o.rmin, r);
        o.rmax = fmax(o.rmax, r);
    }

    /// Pixel box `[x - w, x + w] x [y - w, y + w]`, clamped to the map.
    #[inline(always)]
    fn pixel_box(&self, x: f64, y: f64, w: f64) -> (usize, usize, usize, usize) {
        let c = |v: f64, n: usize| ix(v).min(n - 1);
        (
            c(x - w, self.f.w),
            c(y - w, self.f.h),
            c(x + w, self.f.w),
            c(y + w, self.f.h),
        )
    }

    #[inline(always)]
    fn hits(&self, j: usize, x: f64, y: f64, rc: f32) -> bool {
        let p = unsafe { self.parts.get_unchecked(j) };
        let (dx, dy) = (x - p.x as f64, y - p.y as f64);
        // float32 sum and square, as in the reference
        let req = rc + p.r;
        dx * dx + dy * dy < (req * req) as f64
    }

    /// Scan the linked list of one cell; the conflicting point goes to `hint`.
    #[inline(always)]
    fn scan_cell(&self, head: i32, x: f64, y: f64, rc: f32, hint: &mut i32) -> bool {
        let mut j = head;
        while j >= 0 {
            if self.hits(j as usize, x, y, rc) {
                *hint = j;
                return true;
            }
            j = unsafe { self.parts.get_unchecked(j as usize) }.next;
        }
        false
    }

    /// True if a point at `(x, y)` with radius `rc` is closer than `rc + r_j`
    /// to an accepted point `j`. The result does not depend on the scan
    /// order; `hint` (the last conflicting point, tested first because
    /// successive candidates around a parent are mostly rejected by the same
    /// neighbour) and a centre-cell-first order only make rejections faster.
    #[inline]
    fn conflicts(&self, x: f64, y: f64, rc: f32, hint: &mut i32) -> bool {
        if *hint >= 0 && self.hits(*hint as usize, x, y, rc) {
            return true;
        }
        // (lo, hi) bound the radius of every point that can conflict
        let (mut lo, mut hi) = (self.r_lo, self.r_hi);
        if let Some(p) = &self.pyr {
            // j conflicts => |p_j - p| < rc + r_j <= rc + hi, so r_j is bounded
            // by the radius range over that box; tighten twice.
            for _ in 0..2 {
                let (x0, y0, x1, y1) = self.pixel_box(x, y, rc as f64 + hi as f64);
                let (a, b) = p.bounds(x0, y0, x1, y1);
                lo = a;
                if b >= hi {
                    break;
                }
                hi = b;
            }
        }
        for o in &self.octaves {
            if o.head.is_empty() || o.rmin > hi || o.rmax < lo {
                continue;
            }
            let w = rc as f64 + fmin(o.rmax, hi) as f64;
            let (cx0, cx1) = o.cell_range(x - w, x + w, o.nx);
            let (cy0, cy1) = o.cell_range(y - w, y + w, o.ny);
            let ccx = ix(x * o.inv).clamp(cx0, cx1);
            let ccy = ix(y * o.inv).clamp(cy0, cy1);
            let centre = ccy * o.nx + ccx;
            if self.scan_cell(o.head[centre], x, y, rc, hint) {
                return true;
            }
            for cy in cy0..=cy1 {
                let row = &o.head[cy * o.nx..(cy + 1) * o.nx];
                for (cx, &h) in (cx0..=cx1).zip(&row[cx0..=cx1]) {
                    if cy * o.nx + cx != centre && self.scan_cell(h, x, y, rc, hint) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Run Bridson's algorithm; returns interleaved float32 `(x, y)` points.
    pub fn run(mut self, k: usize, seed: u64) -> Vec<f32> {
        let (w, h) = (self.f.w as f64, self.f.h as f64);
        let mut rng = Xoshiro256pp::new(seed);
        let y0 = rng.uniform() as f64 * h;
        let x0 = rng.uniform() as f64 * w;
        self.insert(x0 as f32, y0 as f32);
        let mut active: Vec<u32> = vec![0];
        let mut probe = 0usize; // next pixel of the fill pass
        let mut probe_hint = -1i32;

        loop {
            if active.is_empty() {
                // Fill pass: Bridson only grows from existing points, so regions
                // it cannot reach (across a sharp radius jump, or cut off by the
                // domain) would stay empty. Seed the first pixel centre (raster
                // order) that can still host a point, then resume growing.
                let n_px = self.f.w * self.f.h;
                while probe < n_px {
                    let (x, y) = (
                        (probe % self.f.w) as f64 + 0.5,
                        (probe / self.f.w) as f64 + 0.5,
                    );
                    let rc = self.f.r[probe];
                    probe += 1;
                    if !self.conflicts(x, y, rc, &mut probe_hint) {
                        active.push(self.parts.len() as u32);
                        self.insert(x as f32, y as f32);
                        break;
                    }
                }
                if active.is_empty() {
                    break;
                }
            }
            let n_act = active.len();
            let ai = ix(rng.uniform() as f64 * n_act as f64).min(n_act - 1);
            let pid = active[ai] as usize;
            let (px, py) = (self.parts[pid].x as f64, self.parts[pid].y as f64);
            let rp = self.f.at(px, py) as f64;
            let inside = |x: f64, y: f64| (0.0..w).contains(&x) && (0.0..h).contains(&y);

            let mut accepted = false;
            let mut hint = -1i32;
            for _ in 0..k {
                let (ca, sa) = turn_cos_sin(rng.next_u24());
                let mut dist = rp * 2.0;
                let mut x = px + dist * ca;
                let mut y = py + dist * sa;
                // Where the radius grows away from the parent, a step of 2 r_p is
                // shorter than the required r_p + r_c: lengthen it to r_p + r_c
                // (a few fixed-point steps). Flat or decreasing radius keeps 2 r_p.
                let mut rc = f32::NAN; // radius at (x, y), once inside
                for it in 0..=MAX_STEP_ITER {
                    if !inside(x, y) {
                        break;
                    }
                    rc = self.f.r[ix(y) * self.f.w + ix(x)];
                    let need = rp + rc as f64;
                    if need <= dist || it == MAX_STEP_ITER {
                        break;
                    }
                    dist = need;
                    x = px + dist * ca;
                    y = py + dist * sa;
                    rc = f32::NAN;
                }
                if rc.is_nan() {
                    continue; // left the domain
                }
                if self.conflicts(x, y, rc, &mut hint) {
                    continue;
                }
                active.push(self.parts.len() as u32);
                self.insert(x as f32, y as f32);
                accepted = true;
                break;
            }
            if !accepted {
                active.swap_remove(ai);
            }
        }
        let mut out = Vec::with_capacity(2 * self.parts.len());
        for p in &self.parts {
            out.push(p.x);
            out.push(p.y);
        }
        out
    }
}

/// Poisson disk samples of a `(h, w)` radius map (row-major), interleaved
/// float32 `(x, y)`.
pub fn poisson_disk(
    r: &[f32],
    h: usize,
    w: usize,
    k: usize,
    seed: u64,
) -> Result<Vec<f32>, String> {
    Ok(Sampler::new(r, h, w)?.run(k, seed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn min_ratio(pts: &[f32], r: &[f32], h: usize, w: usize) -> f64 {
        let f = Field { r, w, h };
        let n = pts.len() / 2;
        let mut best = f64::INFINITY;
        for i in 0..n {
            for j in i + 1..n {
                let (xi, yi, xj, yj) = (
                    pts[2 * i] as f64,
                    pts[2 * i + 1] as f64,
                    pts[2 * j] as f64,
                    pts[2 * j + 1] as f64,
                );
                let d = ((xi - xj).powi(2) + (yi - yj).powi(2)).sqrt();
                let need = (f.at(xi, yi) + f.at(xj, yj)) as f64;
                best = best.min(d / need);
            }
        }
        best
    }

    #[test]
    fn turn_cos_sin_matches_libm() {
        let mut worst = 0.0f64;
        let mut m = 0u32;
        while m < (1 << 24) {
            let a = std::f64::consts::TAU * (m as f64 / (1u64 << 24) as f64);
            let (c, s) = turn_cos_sin(m);
            worst = worst.max((c - a.cos()).abs()).max((s - a.sin()).abs());
            m += 97;
        }
        for m in [0, 1, (1 << 21) - 1, 1 << 21, (1 << 22) + 5, (1 << 24) - 1] {
            let a = std::f64::consts::TAU * (m as f64 / (1u64 << 24) as f64);
            let (c, s) = turn_cos_sin(m);
            worst = worst.max((c - a.cos()).abs()).max((s - a.sin()).abs());
        }
        assert!(worst < 1e-15, "max error {worst:e}");
    }

    #[test]
    fn xoshiro_reference_values() {
        // reference values of xoshiro256++ seeded through SplitMix64
        let mut g = Xoshiro256pp::new(1);
        assert_eq!(g.next_u64(), 0xfc06_d5f2_75a7_3fd8);
        assert_eq!(g.next_u64(), 0x95fa_bd80_3316_fcb6);
        assert_eq!(g.next_u64(), 0xed4c_75f0_ff4f_444a);
        assert_eq!(
            Xoshiro256pp::new(u64::MAX).next_u64(),
            0xb4b9_b3de_17c5_a2e8
        );
    }

    #[test]
    fn spacing_uniform_and_graded() {
        let (h, w) = (60, 90);
        let uni = vec![3.0f32; h * w];
        let pts = poisson_disk(&uni, h, w, 30, 7).unwrap();
        assert!(pts.len() > 20);
        assert!(min_ratio(&pts, &uni, h, w) >= 1.0 - 1e-6);

        let grad: Vec<f32> = (0..h * w).map(|p| 1.0 + 0.2 * (p % w) as f32).collect();
        let pts = poisson_disk(&grad, h, w, 30, 3).unwrap();
        assert!(min_ratio(&pts, &grad, h, w) >= 1.0 - 1e-6);
    }

    #[test]
    fn rejects_bad_radius() {
        assert!(poisson_disk(&[1.0, 0.0], 1, 2, 30, 0).is_err());
        assert!(poisson_disk(&[1.0, f32::NAN], 1, 2, 30, 0).is_err());
        assert!(poisson_disk(&[], 0, 0, 30, 0).is_err());
    }
}
