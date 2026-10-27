//! CPU mirror of the GI shader math (`engine/shaders/gi/*.glsl`).
//!
//! The self-test compares GPU state against these functions, and the unit tests
//! below check the estimators the shaders rely on.

pub const MAX_SURFELS: u32 = 65536;
pub const SURFEL_MIN_RADIUS: f32 = 2.0;
pub const SURFEL_MAX_RADIUS: f32 = 256.0;
pub const SURFEL_KERNEL_MIN_COS: f32 = 0.5;
pub const SURFEL_KERNEL_PLANE_FRACTION: f32 = 0.25;
pub const GRID_LEVELS: u32 = 8;
pub const GRID_BASE_CELL: f32 = 4.0;
pub const GRID_BUCKETS: u32 = 1 << 18;
pub const GRID_SCAN_BLOCK: u32 = 1024;
pub const GRID_MAX_ITEMS: u32 = MAX_SURFELS * 8;
pub const FLUX_SCAN_BLOCK: u32 = 1024;
pub const LIGHT_TILES: u32 = 128;
pub const LIGHT_TILE_SIZE: u32 = 256;

/// Words of the pool state block (`ST_*` in `gi/surfels.glsl`).
pub const STATE_WORDS: usize = 64;
pub const ST_ALIVE_IN: usize = 0;
pub const ST_ALIVE_OUT: usize = 1;
pub const ST_SURVIVORS: usize = 2;
pub const ST_FREE: usize = 3;
pub const ST_SPAWN_SCREEN: usize = 4;
pub const ST_SPAWN_RAY: usize = 5;
pub const ST_EVICTED: usize = 6;
pub const ST_LEVEL_MASK: usize = 7;
pub const ST_GRID_ITEMS: usize = 8;
pub const ST_TOTAL_FLUX: usize = 9;
pub const ST_LIGHT_COUNT: usize = 10;
pub const ST_FREE_AT_BEGIN: usize = 11;
pub const ST_SPAWN_FAILED: usize = 12;
pub const ST_DOOMED: usize = 13;
pub const ST_MAX_FLUX: usize = 14;
pub const ST_PREV_MAX_FLUX: usize = 15;
pub const ST_STABLE_FRAMES: usize = 16;

/// Byte size of `Surfel` (std430) and offsets of the fields the self-test reads.
pub const SURFEL_STRIDE: usize = 128;
pub const SURFEL_OFFSET_POSITION: usize = 0;
pub const SURFEL_OFFSET_RADIUS: usize = 12;
pub const SURFEL_OFFSET_GENERATION: usize = 44;
pub const SURFEL_OFFSET_AREA: usize = 60;
pub const SURFEL_OFFSET_FLAGS: usize = 108;
pub const SURFEL_FLAG_ALIVE: u32 = 1;

/// Byte size of `Reservoir` (std430).
pub const RESERVOIR_STRIDE: usize = 32;
pub const TILE_ENTRY_STRIDE: usize = 32;
pub const LIGHT_RECORD_STRIDE: usize = 32;
pub const FLUX_QUANT_MAX: u32 = 65535;

pub fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    x
}

pub fn grid_cell_size(level: u32) -> f32 {
    GRID_BASE_CELL * (1u32 << level) as f32
}

pub fn grid_level(radius: f32) -> u32 {
    let mut level = 0;
    while level + 1 < GRID_LEVELS && grid_cell_size(level) < 2.0 * radius {
        level += 1;
    }
    level
}

pub fn grid_bucket(cell: [i32; 3], level: u32) -> u32 {
    let mut h = hash_u32(level.wrapping_add(0x632be5ab));
    h = hash_u32(h ^ cell[0] as u32);
    h = hash_u32(h ^ cell[1] as u32);
    h = hash_u32(h ^ cell[2] as u32);
    h & (GRID_BUCKETS - 1)
}

fn floor_div(v: f32, cs: f32) -> i32 {
    (v / cs).floor() as i32
}

/// Distinct buckets a surfel is inserted into (mirror of `surfel_buckets`).
pub fn surfel_buckets(c: [f32; 3], r: f32) -> Vec<u32> {
    let level = grid_level(r);
    let cs = grid_cell_size(level);
    let lo: Vec<i32> = (0..3).map(|a| floor_div(c[a] - r, cs)).collect();
    let hi: Vec<i32> = (0..3)
        .map(|a| floor_div(c[a] + r, cs).min(lo[a] + 1))
        .collect();
    let mut buckets = Vec::with_capacity(8);
    for z in lo[2]..=hi[2] {
        for y in lo[1]..=hi[1] {
            for x in lo[0]..=hi[0] {
                let b = grid_bucket([x, y, z], level);
                if !buckets.contains(&b) {
                    buckets.push(b);
                }
            }
        }
    }
    buckets
}

/// Distinct buckets a point query reads (mirror of `point_buckets`).
pub fn point_buckets(p: [f32; 3], level_mask: u32) -> Vec<u32> {
    let mut buckets = Vec::with_capacity(GRID_LEVELS as usize);
    for level in 0..GRID_LEVELS {
        if level_mask & (1 << level) == 0 {
            continue;
        }
        let cs = grid_cell_size(level);
        let cell = [
            floor_div(p[0], cs),
            floor_div(p[1], cs),
            floor_div(p[2], cs),
        ];
        let b = grid_bucket(cell, level);
        if !buckets.contains(&b) {
            buckets.push(b);
        }
    }
    buckets
}

pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Partition-of-unity kernel (mirror of `surfel_kernel`).
pub fn surfel_kernel(p: [f32; 3], np: [f32; 3], c: [f32; 3], nc: [f32; 3], r: f32) -> f32 {
    let d = sub(p, c);
    let d2 = dot(d, d);
    let r2 = r * r;
    if d2 >= r2 || dot(np, nc) < SURFEL_KERNEL_MIN_COS {
        return 0.0;
    }
    if dot(d, nc).abs() > SURFEL_KERNEL_PLANE_FRACTION * r {
        return 0.0;
    }
    let t = 1.0 - d2 / r2;
    t * t
}

pub const DISK_POLY_N: usize = 16;

/// Orthonormal basis around a unit normal (mirror of `make_basis`).
pub fn make_basis(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let s = if n[2] >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n[2]);
    let k = n[0] * n[1] * a;
    (
        [1.0 + s * n[0] * n[0] * a, s * k, -s * n[0]],
        [k, s + n[1] * n[1] * a, -n[1]],
    )
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = dot(v, v).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Geometry term G of a uniform Lambertian disk (center `c`, normal `nc`,
/// radius `r`) seen from (x, nx): irradiance = L * A * G, the emitted power
/// L A being spread over the disk. Exact vector irradiance when the disk is
/// above the receiver's horizon; equal-area 16-gon clipped to the horizon and
/// integrated with Lambert's polygon formula when the horizon cuts it.
/// Mirror of `disk_geometry` in surfels.glsl.
/// Smallest geometry term inside the support. rust: DISK_SUPPORT_FLOOR
pub const DISK_SUPPORT_FLOOR: f32 = 1e-30;

/// Whether `disk_geometry` is positive. rust: disk_supports
pub fn disk_supports(c: [f32; 3], nc: [f32; 3], r: f32, x: [f32; 3], nx: [f32; 3]) -> bool {
    let dv = sub(x, c);
    if dot(dv, nc) <= 0.0 || r <= 0.0 {
        return false;
    }
    let extent = r * (1.0 - dot(nc, nx) * dot(nc, nx)).max(0.0).sqrt();
    -dot(dv, nx) > -extent
}

pub fn disk_geometry(c: [f32; 3], nc: [f32; 3], r: f32, x: [f32; 3], nx: [f32; 3]) -> f32 {
    if !disk_supports(c, nc, r, x, nx) {
        return 0.0;
    }
    let dv = sub(x, c);
    let h = dot(dv, nc);
    let above = -dot(dv, nx);
    let extent = r * (1.0 - dot(nc, nx) * dot(nc, nx)).max(0.0).sqrt();
    if above >= extent {
        let rv = [dv[0] - h * nc[0], dv[1] - h * nc[1], dv[2] - h * nc[2]];
        let rho2 = dot(rv, rv);
        let t = dot(dv, dv) + r * r;
        let s = (t * t - 4.0 * rho2 * r * r).max(0.0).sqrt();
        let denom = s * (s + t);
        if denom <= 0.0 {
            return DISK_SUPPORT_FLOOR;
        }
        let num = dot(nx, nc) * (s + t - 2.0 * rho2) + 2.0 * h * dot(nx, rv);
        return (-num / denom).max(DISK_SUPPORT_FLOOR);
    }

    let (tb, bb) = make_basis(nc);
    let n = DISK_POLY_N as f32;
    let tau = std::f32::consts::TAU;
    let rp = r * (tau / (n * (tau / n).sin())).sqrt();
    let vertex = |k: usize| {
        let phi = tau * k as f32 / n;
        let (cs, sn) = (phi.cos(), phi.sin());
        [
            c[0] + rp * (cs * tb[0] + sn * bb[0]),
            c[1] + rp * (cs * tb[1] + sn * bb[1]),
            c[2] + rp * (cs * tb[2] + sn * bb[2]),
        ]
    };
    let mut poly: Vec<[f32; 3]> = Vec::with_capacity(DISK_POLY_N + 1);
    for k in 0..DISK_POLY_N {
        let (a, b) = (vertex(k), vertex(k + 1));
        let (da, db) = (dot(sub(a, x), nx), dot(sub(b, x), nx));
        if da >= 0.0 {
            poly.push(a);
        }
        if (da >= 0.0) != (db >= 0.0) {
            let f = da / (da - db);
            poly.push([
                a[0] + (b[0] - a[0]) * f,
                a[1] + (b[1] - a[1]) * f,
                a[2] + (b[2] - a[2]) * f,
            ]);
        }
    }
    let mut sum = 0.0f32;
    for k in 0..poly.len() {
        let ui = normalize(sub(poly[k], x));
        let uj = normalize(sub(poly[(k + 1) % poly.len()], x));
        let g = cross(ui, uj);
        let gl = dot(g, g).sqrt();
        if gl > 1e-12 {
            sum += dot(ui, uj).clamp(-1.0, 1.0).acos() * dot(nx, g) / gl;
        }
    }
    (sum.abs() * 0.5 / (std::f32::consts::PI * r * r)).max(DISK_SUPPORT_FLOOR)
}

/// Irradiance from a Lambertian disk light of radiance `l` and area `a`.
pub fn disk_irradiance(
    l: f32,
    a: f32,
    r: f32,
    x: [f32; 3],
    nx: [f32; 3],
    y: [f32; 3],
    ny: [f32; 3],
) -> f32 {
    l * a * disk_geometry(y, ny, r, x, nx)
}

/// Smallest index with `cdf[i] > target` (mirror of the presample search).
pub fn cdf_search(cdf: &[u32], target: u32) -> usize {
    let (mut lo, mut hi) = (0usize, cdf.len() - 1);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cdf[mid] > target {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// Mirror of `quantize_flux` in publish.comp.
pub fn quantize_flux(flux: f32, scale: f32) -> u32 {
    if !(flux > 0.0) {
        return 0;
    }
    if !(scale > 0.0) {
        return 1;
    }
    (flux / scale * FLUX_QUANT_MAX as f32)
        .ceil()
        .clamp(1.0, FLUX_QUANT_MAX as f32) as u32
}

/// Mirror of the uniform draw in presample.comp: mask to the next power of
/// two and reject. `None` only after 64 rejections.
pub fn uniform_below(total: u32, mut next: impl FnMut() -> u32) -> Option<u32> {
    let msb = 31 - (total - 1).leading_zeros() as i32;
    let mask = if total == 1 {
        0
    } else if msb >= 31 {
        u32::MAX
    } else {
        (1u32 << (msb + 1)) - 1
    };
    for _ in 0..65 {
        let t = next() & mask;
        if t < total {
            return Some(t);
        }
    }
    None
}

/// Octahedral decoding of `compress_unit_vec` (compress.glsl), up to rounding.
pub fn decompress_unit_vec(packed: u32) -> [f32; 3] {
    let mut x = ((packed & 0xFFFF) as i32 - 32767) as f32 / 32767.0;
    let mut y = ((packed >> 16) as i32 - 32767) as f32 / 32767.0;
    let z = 1.0 - x.abs() - y.abs();
    if z < 0.0 {
        let ox = (1.0 - y.abs()) * if x >= 0.0 { 1.0 } else { -1.0 };
        let oy = (1.0 - x.abs()) * if y >= 0.0 { 1.0 } else { -1.0 };
        x = ox;
        y = oy;
    }
    let len = (x * x + y * y + z * z).sqrt();
    [x / len, y / len, z / len]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic generator for the Monte Carlo checks.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
        }
        fn next_u32(&mut self) -> u32 {
            self.next();
            (self.0 >> 32) as u32
        }
        fn disk(&mut self) -> (f32, f32) {
            let r = self.next().sqrt();
            let phi = 2.0 * std::f32::consts::PI * self.next();
            (r * phi.cos(), r * phi.sin())
        }
    }

    #[test]
    fn grid_level_cell_holds_diameter() {
        let mut r = SURFEL_MIN_RADIUS;
        while r <= SURFEL_MAX_RADIUS {
            let level = grid_level(r);
            assert!(grid_cell_size(level) >= 2.0 * r, "radius {r} level {level}");
            if level > 0 {
                assert!(grid_cell_size(level - 1) < 2.0 * r);
            }
            r *= 1.07;
        }
    }

    #[test]
    fn surfel_spans_at_most_eight_cells_and_queries_find_it() {
        let mut rng = Rng(7);
        for _ in 0..20000 {
            let c = [
                (rng.next() - 0.5) * 4000.0,
                (rng.next() - 0.5) * 4000.0,
                (rng.next() - 0.5) * 4000.0,
            ];
            let r = SURFEL_MIN_RADIUS + rng.next() * (SURFEL_MAX_RADIUS - SURFEL_MIN_RADIUS);
            let inserted = surfel_buckets(c, r);
            assert!(!inserted.is_empty() && inserted.len() <= 8);
            let mask = 1u32 << grid_level(r);
            // any point of the sphere must read one of the surfel's buckets
            for _ in 0..8 {
                let (dx, dy) = rng.disk();
                let dz = (rng.next() - 0.5) * 2.0 * (1.0 - dx * dx - dy * dy).max(0.0).sqrt();
                let p = [
                    c[0] + dx * r * 0.999,
                    c[1] + dy * r * 0.999,
                    c[2] + dz * r * 0.999,
                ];
                let read = point_buckets(p, mask);
                assert_eq!(read.len(), 1);
                assert!(inserted.contains(&read[0]));
            }
        }
    }

    #[test]
    fn kernel_partition_area_integrates_to_disk_area() {
        // A lone surfel: the partition-of-unity area is its whole disk.
        let mut rng = Rng(11);
        let r = 10.0f32;
        let n = [0.0, 0.0, 1.0];
        let mut sum = 0.0f64;
        let samples = 200000;
        for _ in 0..samples {
            let (dx, dy) = rng.disk();
            let u = [dx * r, dy * r, 0.0];
            let k = surfel_kernel(u, n, [0.0; 3], n, r);
            sum += (std::f32::consts::PI * r * r * k / k) as f64;
        }
        let area = sum / samples as f64;
        let exact = std::f64::consts::PI * 100.0;
        assert!((area - exact).abs() / exact < 1e-4);
    }

    #[test]
    fn partition_areas_sum_to_covered_area() {
        // Two overlapping coplanar surfels: the partition-of-unity areas must
        // add up to the area of the union of their disks.
        let mut rng = Rng(5);
        let n = [0.0, 0.0, 1.0];
        let surfels = [([0.0f32, 0.0, 0.0], 10.0f32), ([12.0, 0.0, 0.0], 8.0)];
        let samples = 400000;
        let mut areas = [0.0f64; 2];
        for (j, (c, r)) in surfels.iter().enumerate() {
            for _ in 0..samples {
                let (dx, dy) = rng.disk();
                let u = [c[0] + dx * r, c[1] + dy * r, 0.0];
                let kj = surfel_kernel(u, n, *c, n, *r);
                let total: f32 = surfels
                    .iter()
                    .map(|(ck, rk)| surfel_kernel(u, n, *ck, n, *rk))
                    .sum();
                if total > 0.0 {
                    areas[j] += (std::f32::consts::PI * r * r * kj / total) as f64;
                }
            }
            areas[j] /= samples as f64;
        }
        // union area by Monte Carlo over the bounding box
        let mut inside = 0u64;
        let box_samples = 2000000u64;
        for _ in 0..box_samples {
            let x = -10.0 + rng.next() * 30.0;
            let y = -10.0 + rng.next() * 20.0;
            if surfels
                .iter()
                .any(|(c, r)| (x - c[0]).powi(2) + (y - c[1]).powi(2) < r * r)
            {
                inside += 1;
            }
        }
        let union = 600.0 * inside as f64 / box_samples as f64;
        let total = areas[0] + areas[1];
        assert!(
            (total - union).abs() / union < 0.01,
            "partition areas {total} vs union {union}"
        );
    }

    /// Brute-force polar integration of the disk's irradiance.
    fn disk_irradiance_numeric(
        c: [f32; 3],
        nc: [f32; 3],
        r: f32,
        x: [f32; 3],
        nx: [f32; 3],
    ) -> f64 {
        let helper = if nc[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let cross = |a: [f32; 3], b: [f32; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let t = cross(nc, helper);
        let tl = dot(t, t).sqrt();
        let t = [t[0] / tl, t[1] / tl, t[2] / tl];
        let b = cross(nc, t);
        let n = 600;
        let mut sum = 0.0f64;
        for i in 0..n {
            let rho = (i as f64 + 0.5) / n as f64 * r as f64;
            for j in 0..n {
                let phi = (j as f64 + 0.5) / n as f64 * std::f64::consts::TAU;
                let (u, v) = (rho * phi.cos(), rho * phi.sin());
                let y: Vec<f64> = (0..3)
                    .map(|k| c[k] as f64 + u * t[k] as f64 + v * b[k] as f64)
                    .collect();
                let d: Vec<f64> = (0..3).map(|k| y[k] - x[k] as f64).collect();
                let d2: f64 = d.iter().map(|v| v * v).sum();
                let dl = d2.sqrt();
                let cx: f64 = (0..3).map(|k| nx[k] as f64 * d[k] / dl).sum();
                let cy: f64 = -(0..3).map(|k| nc[k] as f64 * d[k] / dl).sum::<f64>();
                if cx > 0.0 && cy > 0.0 {
                    sum += cx * cy / d2
                        * rho
                        * (r as f64 / n as f64)
                        * (std::f64::consts::TAU / n as f64);
                }
            }
        }
        sum
    }

    /// The closed form matches brute-force integration in the near field (where
    /// the old L A cos cos / (d^2 + r^2) was off by up to 4x) and far field.
    #[test]
    fn disk_irradiance_matches_numeric_integration() {
        let s = std::f32::consts::FRAC_1_SQRT_2;
        // (center, normal, radius, receiver, receiver normal)
        let cases: [([f32; 3], [f32; 3], f32, [f32; 3], [f32; 3]); 13] = [
            (
                [0.0; 3],
                [0.0, 0.0, 1.0],
                1.0,
                [0.0, 0.0, 0.5],
                [0.0, 0.0, -1.0],
            ),
            (
                [0.0; 3],
                [0.0, 0.0, 1.0],
                1.0,
                [1.5, 0.0, 0.5],
                [-1.0, 0.0, 0.0],
            ),
            (
                [0.0; 3],
                [0.0, 0.0, 1.0],
                1.0,
                [1.1, 0.0, 0.2],
                [-1.0, 0.0, 0.0],
            ),
            (
                [0.0; 3],
                [0.0, 0.0, 1.0],
                1.0,
                [1.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ),
            (
                [0.0; 3],
                [0.0, 0.0, 1.0],
                1.0,
                [2.0, 0.0, 2.0],
                [-s, 0.0, -s],
            ),
            (
                [0.0; 3],
                [0.0, 0.0, 1.0],
                1.0,
                [3.0, 1.0, 1.0],
                [-1.0, 0.0, 0.0],
            ),
            (
                [6.0, 0.0, 9.0],
                [-s, 0.0, -s],
                1.0,
                [0.0; 3],
                [0.0, 0.0, 1.0],
            ),
            (
                [0.0, 0.0, 40.0],
                [0.0, 0.0, -1.0],
                3.0,
                [5.0, -2.0, 0.0],
                [0.0, 0.0, 1.0],
            ),
            // horizon cuts the disk (concave corners)
            (
                [0.0; 3],
                [0.0, 0.0, 1.0],
                1.0,
                [0.5, 0.0, 0.2],
                [-1.0, 0.0, 0.0],
            ),
            (
                [0.0; 3],
                [0.0, 0.0, 1.0],
                1.0,
                [0.0, 0.0, 0.5],
                [-1.0, 0.0, 0.0],
            ),
            (
                [0.0; 3],
                [0.0, 0.0, 1.0],
                1.0,
                [0.9, 0.0, 0.05],
                [-1.0, 0.0, 0.0],
            ),
            (
                [0.0; 3],
                [0.0, 0.0, 1.0],
                1.0,
                [0.3, 0.0, 0.4],
                [-0.957826, 0.0, -0.287348],
            ),
            (
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                2.0,
                [2.5, 1.6, 0.4],
                [-0.6, -0.48, 0.64],
            ),
        ];
        for (c, nc, r, x, nx) in cases {
            let numeric = disk_irradiance_numeric(c, nc, r, x, nx);
            let closed = (std::f32::consts::PI * r * r * disk_geometry(c, nc, r, x, nx)) as f64;
            assert!(
                (closed - numeric).abs() <= 2e-3 * numeric,
                "x {x:?}: closed {closed} numeric {numeric}"
            );
        }
        // Behind the disk or facing away: nothing.
        assert_eq!(
            disk_geometry(
                [0.0; 3],
                [0.0, 0.0, 1.0],
                1.0,
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0]
            ),
            0.0
        );
        assert_eq!(
            disk_geometry(
                [0.0; 3],
                [0.0, 0.0, 1.0],
                1.0,
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0]
            ),
            0.0
        );
    }

    /// The support test used for the Z of ReSTIR agrees with the geometry term,
    /// including receivers whose horizon cuts the disk.
    #[test]
    fn disk_support_matches_geometry() {
        let mut rng = Rng(7);
        let mut unit = |rng: &mut Rng| loop {
            let v = [
                rng.next() * 2.0 - 1.0,
                rng.next() * 2.0 - 1.0,
                rng.next() * 2.0 - 1.0,
            ];
            let l = dot(v, v);
            if l > 1e-4 && l <= 1.0 {
                let l = l.sqrt();
                break [v[0] / l, v[1] / l, v[2] / l];
            }
        };
        let (mut supported, mut cut) = (0, 0);
        for _ in 0..200_000 {
            let nc = unit(&mut rng);
            let nx = unit(&mut rng);
            let r = 0.5 + 2.0 * rng.next();
            let d = unit(&mut rng);
            let dist = 3.0 * r * rng.next();
            let x = [d[0] * dist, d[1] * dist, d[2] * dist];
            let g = disk_geometry([0.0; 3], nc, r, x, nx);
            let s = disk_supports([0.0; 3], nc, r, x, nx);
            assert_eq!(
                s,
                g > 0.0,
                "x {x:?} nx {nx:?} nc {nc:?} r {r}: geometry {g}"
            );
            if s {
                supported += 1;
                let extent = r * (1.0 - dot(nc, nx) * dot(nc, nx)).max(0.0).sqrt();
                if -dot(x, nx) < extent {
                    cut += 1;
                }
            }
        }
        assert!(
            supported > 10_000 && cut > 1_000,
            "supported {supported} cut {cut}"
        );
    }

    /// Every integer target maps to exactly q_i values for light i: the
    /// selection probability equals the stored pmf q_i / Q with no rounding.
    #[test]
    fn integer_cdf_selection_is_exact() {
        let fluxes = [0.0f32, 2.0, 0.0, 5.0e-6, 1.0, 0.0, 2.0, 1.0e3, 7.5];
        let scale = 1.0e3;
        let q: Vec<u32> = fluxes.iter().map(|f| quantize_flux(*f, scale)).collect();
        assert!(
            q.iter()
                .zip(fluxes.iter())
                .all(|(q, f)| (*q > 0) == (*f > 0.0))
        );
        let mut cdf = Vec::new();
        let mut acc = 0u32;
        for v in q.iter() {
            acc += v;
            cdf.push(acc);
        }
        let total = acc;
        let mut counts = vec![0u32; q.len()];
        for t in 0..total {
            counts[cdf_search(&cdf, t)] += 1;
        }
        assert_eq!(counts, q);
        // The largest possible sum still fits 32 bits.
        assert!((MAX_SURFELS as u64) * (FLUX_QUANT_MAX as u64) < (1u64 << 32));
    }

    #[test]
    fn masked_rejection_draw_is_uniform() {
        for total in [1u32, 2, 3, 5, 1000, 65537] {
            let mut rng = Rng(total as u64 + 3);
            let mut counts = vec![0u32; total.min(16) as usize];
            let n = 200000;
            for _ in 0..n {
                let t = uniform_below(total, || rng.next_u32()).unwrap();
                assert!(t < total);
                if (t as usize) < counts.len() {
                    counts[t as usize] += 1;
                }
            }
            let expected = n as f64 / total as f64;
            for c in counts {
                assert!(((c as f64) - expected).abs() < 6.0 * expected.sqrt() + 1.0);
            }
        }
    }

    /// One pixel, a handful of lights: RIS with the irradiance target must be
    /// unbiased, and so must the Z-normalized merge of reservoirs built for
    /// different receivers (some lights invisible to some domains).
    #[test]
    fn ris_and_z_normalized_merge_are_unbiased() {
        // contribution of each light at receiver d (0 = shaded pixel)
        let f = |d: usize, j: usize| -> f32 {
            let table = [
                [3.0f32, 0.5, 0.0, 2.0, 1.0],
                [0.0f32, 1.5, 2.0, 2.0, 0.0],
                [1.0f32, 0.0, 0.0, 4.0, 2.0],
            ];
            table[d][j]
        };
        let pmf = [0.1f32, 0.3, 0.2, 0.25, 0.15];
        let exact: f32 = (0..5).map(|j| f(0, j)).sum();

        let mut rng = Rng(42);
        let pick = |rng: &mut Rng| -> usize {
            let u = rng.next();
            let mut acc = 0.0;
            for (j, p) in pmf.iter().enumerate() {
                acc += p;
                if u < acc {
                    return j;
                }
            }
            4
        };
        // reservoir = (y, W, M) built with candidates for domain d
        let build = |rng: &mut Rng, d: usize, m: usize| -> (Option<usize>, f32, f32) {
            let mut wsum = 0.0;
            let mut y = None;
            for _ in 0..m {
                let j = pick(rng);
                let w = f(d, j) / pmf[j];
                wsum += w;
                if w > 0.0 && rng.next() * wsum < w {
                    y = Some(j);
                }
            }
            match y {
                Some(j) => (Some(j), wsum / (m as f32 * f(d, j)), m as f32),
                None => (None, 0.0, m as f32),
            }
        };

        let trials = 400000;
        let mut ris = 0.0f64;
        let mut merged = 0.0f64;
        for _ in 0..trials {
            let r0 = build(&mut rng, 0, 4);
            if let Some(j) = r0.0 {
                ris += (f(0, j) * r0.1) as f64;
            }
            // merge the pixel's reservoir with two neighbours' reservoirs
            let rs = [r0, build(&mut rng, 1, 4), build(&mut rng, 2, 4)];
            let mut wsum = 0.0;
            let mut y = None;
            for r in rs.iter() {
                if let Some(j) = r.0 {
                    let w = f(0, j) * r.1 * r.2;
                    wsum += w;
                    if w > 0.0 && rng.next() * wsum < w {
                        y = Some(j);
                    }
                }
            }
            if let Some(j) = y {
                let z: f32 = (0..3).filter(|&d| f(d, j) > 0.0).map(|d| rs[d].2).sum();
                let w = wsum / (z * f(0, j));
                merged += (f(0, j) * w) as f64;
            }
        }
        let ris = ris / trials as f64;
        let merged = merged / trials as f64;
        assert!(
            (ris - exact as f64).abs() / (exact as f64) < 0.01,
            "RIS {ris} vs {exact}"
        );
        assert!(
            (merged - exact as f64).abs() / (exact as f64) < 0.01,
            "merge {merged} vs {exact}"
        );
    }

    #[test]
    fn free_stack_pops_are_unique() {
        // Simulate interleaved atomic pops with failure restore.
        let initial = 5i32;
        let mut counter = initial;
        let mut pending: Vec<usize> = Vec::new();
        let mut taken = Vec::new();
        let mut rng = Rng(1);
        let mut threads: Vec<u8> = vec![0; 40];
        // state 0 = not started, 1 = failed awaiting restore, 2 = done
        for _ in 0..10000 {
            let t = (rng.next() * threads.len() as f32) as usize % threads.len();
            match threads[t] {
                0 => {
                    let prev = counter;
                    counter -= 1;
                    if prev <= 0 {
                        threads[t] = 1;
                        pending.push(t);
                    } else {
                        taken.push(prev - 1);
                        threads[t] = 2;
                    }
                }
                1 => {
                    counter += 1;
                    threads[t] = 2;
                }
                _ => {}
            }
        }
        let mut sorted = taken.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), taken.len(), "a slot was handed out twice");
        assert!(taken.iter().all(|&s| s >= 0 && s < initial));
    }
}
