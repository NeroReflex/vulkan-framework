//! Invariants of the GI state, checked on a host readback taken at the end of
//! a frame (`ART_RTIC_SELFTEST=frame,frame,...`).

use crate::rendering::pipeline::{
    gi_math::*,
    global_illumination::{GiReadback, ReadbackItem},
};

pub struct Report {
    pub failures: u32,
    pub lines: Vec<String>,
}

impl Report {
    fn check(&mut self, ok: bool, what: impl Into<String>) {
        if !ok {
            self.failures += 1;
            self.lines.push(format!("FAIL {}", what.into()));
        }
    }

    fn info(&mut self, what: impl Into<String>) {
        self.lines.push(what.into());
    }
}

fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn float(bytes: &[u8], offset: usize) -> f32 {
    f32::from_bits(word(bytes, offset))
}

pub fn check(readback: &GiReadback) -> Report {
    let mut report = Report {
        failures: 0,
        lines: vec![],
    };
    let max = MAX_SURFELS as usize;
    let state = readback.u32s(ReadbackItem::State);
    let surfels = readback.bytes(ReadbackItem::Surfels);
    let free_ids = readback.u32s(ReadbackItem::FreeIds);
    let alive_ids = readback.u32s(ReadbackItem::AliveIds);
    let geo = readback.f32s(ReadbackItem::Geo);
    let light = readback.f32s(ReadbackItem::Light);
    let grid_start = readback.u32s(ReadbackItem::GridStart);
    let grid_items = readback.u32s(ReadbackItem::GridItems);
    let flux_cdf = readback.u32s(ReadbackItem::FluxCdf);
    let records = readback.u32s(ReadbackItem::LightRecords);
    let normals = readback.u32s(ReadbackItem::Normals);
    let tiles = readback.u32s(ReadbackItem::LightTiles);
    let overlay = readback.u32s(ReadbackItem::Overlay);
    let coverage = readback.f32s(ReadbackItem::Coverage);
    let albedo = readback.f32s(ReadbackItem::Albedo);
    let e_dir = readback.f32s(ReadbackItem::EDir);
    let e_ind = readback.f32s(ReadbackItem::EInd);

    let alive_count = state[ST_ALIVE_OUT] as usize;
    let survivors = state[ST_SURVIVORS] as usize;
    let free_count = state[ST_FREE] as i32;
    report.info(format!(
        "alive {alive_count} (survivors {survivors}, spawned screen {} ray {}, evicted {}, doomed {}, failed spawns {}), free {free_count}, grid items {}, unchanged for {} frames",
        state[ST_SPAWN_SCREEN] as i32,
        state[ST_SPAWN_RAY] as i32,
        state[ST_EVICTED],
        state[ST_DOOMED],
        state[ST_SPAWN_FAILED],
        state[ST_GRID_ITEMS],
        state[ST_STABLE_FRAMES],
    ));

    // ---- pool: every slot is either alive (listed once) or free (stacked once) ----
    report.check(free_count >= 0, format!("free count {free_count} negative"));
    report.check(
        alive_count + free_count.max(0) as usize == max,
        format!("alive {alive_count} + free {free_count} != {max}"),
    );
    report.check(survivors <= alive_count, "survivors exceed alive count");
    let out_base = (((readback.frame_index + 1) & 1) as usize) * max;
    let alive: Vec<u32> = alive_ids[out_base..out_base + alive_count.min(max)].to_vec();
    let mut owner = vec![0u8; max];
    for &id in alive.iter() {
        if (id as usize) >= max {
            report.check(false, format!("alive id {id} out of range"));
            continue;
        }
        owner[id as usize] += 1;
        let flags = word(surfels, id as usize * SURFEL_STRIDE + SURFEL_OFFSET_FLAGS);
        report.check(
            flags & SURFEL_FLAG_ALIVE != 0,
            format!("alive id {id} without ALIVE flag"),
        );
        let r = float(surfels, id as usize * SURFEL_STRIDE + SURFEL_OFFSET_RADIUS);
        report.check(
            (SURFEL_MIN_RADIUS..=SURFEL_MAX_RADIUS).contains(&r),
            format!("surfel {id} radius {r} out of range"),
        );
    }
    for &id in free_ids.iter().take(free_count.max(0) as usize) {
        if (id as usize) >= max {
            report.check(false, format!("free id {id} out of range"));
            continue;
        }
        owner[id as usize] += 2;
        let flags = word(surfels, id as usize * SURFEL_STRIDE + SURFEL_OFFSET_FLAGS);
        report.check(
            flags & SURFEL_FLAG_ALIVE == 0,
            format!("free id {id} still ALIVE"),
        );
    }
    let bad_owner = owner.iter().filter(|&&o| o != 1 && o != 2).count();
    report.check(
        bad_owner == 0,
        format!("{bad_owner} slots are not exactly once alive or once free"),
    );

    // ---- grid: exact CSR over the survivors ----
    let buckets = GRID_BUCKETS as usize;
    let total_items = grid_start[buckets] as usize;
    report.check(
        total_items == state[ST_GRID_ITEMS] as usize,
        "grid_start[B] differs from the item count",
    );
    report.check(
        grid_start.windows(2).take(buckets).all(|w| w[0] <= w[1]),
        "grid_start is not monotonic",
    );
    let mut expected_items = 0usize;
    let mut missing = 0usize;
    let mut level_mask = 0u32;
    let mut is_survivor = vec![false; max];
    for &id in alive.iter().take(survivors) {
        is_survivor[id as usize] = true;
    }
    for &id in alive.iter().take(survivors) {
        let g = &geo[id as usize * 4..id as usize * 4 + 4];
        let c = [g[0], g[1], g[2]];
        level_mask |= 1 << grid_level(g[3]);
        let inserted = surfel_buckets(c, g[3]);
        expected_items += inserted.len();
        for b in inserted {
            let range = grid_start[b as usize] as usize..grid_start[b as usize + 1] as usize;
            if !grid_items[range].contains(&id) {
                missing += 1;
            }
        }
    }
    report.check(
        expected_items == total_items,
        format!("grid holds {total_items} items, expected {expected_items}"),
    );
    report.check(
        missing == 0,
        format!("{missing} surfel/bucket memberships missing"),
    );
    report.check(
        level_mask == state[ST_LEVEL_MASK],
        format!(
            "level mask {level_mask:#x} vs GPU {:#x}",
            state[ST_LEVEL_MASK]
        ),
    );
    let strays = grid_items[..total_items.min(grid_items.len())]
        .iter()
        .filter(|&&id| (id as usize) >= max || !is_survivor[id as usize])
        .count();
    report.check(
        strays == 0,
        format!("{strays} grid items are not survivors"),
    );

    // ---- lights: records, integer CDF and presampled tiles ----
    let light_count = state[ST_LIGHT_COUNT] as usize;
    report.check(
        light_count == alive_count,
        "light count differs from the alive count",
    );
    let lum = |c: &[f32]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let mut lit = 0usize;
    let mut area_violations = 0usize;
    let mut record_mismatches = 0usize;
    let mut max_flux = 0.0f32;
    for &id in alive.iter() {
        let base = id as usize * SURFEL_STRIDE;
        let la = &light[id as usize * 4..id as usize * 4 + 4];
        let rec = &records[id as usize * 8..id as usize * 8 + 8];
        let rec_flux = f32::from_bits(rec[5]);
        let g = &geo[id as usize * 4..id as usize * 4 + 4];
        let flux = lum(la).max(0.0);
        if rec[6] != word(surfels, base + SURFEL_OFFSET_GENERATION)
            || (rec_flux - flux).abs() > 1e-6 * flux.max(1e-30)
            || (0..4).any(|c| f32::from_bits(rec[c]) != g[c])
            || rec[4] != normals[id as usize]
        {
            record_mismatches += 1;
        }
        if la.iter().any(|v| !v.is_finite() || *v < 0.0) {
            report.check(false, format!("surfel {id} light {la:?} invalid"));
        }
        if la[3] > 0.0 {
            lit += 1;
            let r = float(surfels, base + SURFEL_OFFSET_RADIUS);
            if la[3] > std::f32::consts::PI * r * r * 1.001 {
                area_violations += 1;
            }
        }
        max_flux = max_flux.max(rec_flux);
    }
    report.check(
        record_mismatches == 0,
        format!("{record_mismatches} light records disagree with the surfel state"),
    );
    let stale_free_records = free_ids
        .iter()
        .take(free_count.max(0) as usize)
        .filter(|&&id| (id as usize) < max && f32::from_bits(records[id as usize * 8 + 5]) != 0.0)
        .count();
    report.check(
        stale_free_records == 0,
        format!("{stale_free_records} free slots still carry a light record"),
    );
    report.check(
        area_violations == 0,
        format!("{area_violations} surfels with area above pi r^2"),
    );
    report.check(
        f32::from_bits(state[ST_MAX_FLUX]) == max_flux,
        format!(
            "max flux {} vs GPU {}",
            max_flux,
            f32::from_bits(state[ST_MAX_FLUX])
        ),
    );
    // The CDF is an integer prefix sum of q_i in [1, FLUX_QUANT_MAX] for lights
    // (0 otherwise), each within one unit of the CPU quantization (GPU division
    // is not correctly rounded).
    let scale = f32::from_bits(state[ST_PREV_MAX_FLUX]);
    let mut acc = 0u64;
    let mut cdf_errors = 0usize;
    for (i, &id) in alive.iter().enumerate() {
        let below = if i == 0 { 0 } else { flux_cdf[i - 1] as u64 };
        let q = (flux_cdf[i] as u64).wrapping_sub(below);
        let flux = f32::from_bits(records[id as usize * 8 + 5]);
        let expected = quantize_flux(flux, scale) as u64;
        let in_range = if flux > 0.0 {
            (1..=FLUX_QUANT_MAX as u64).contains(&q)
        } else {
            q == 0
        };
        if (flux_cdf[i] as u64) < below || !in_range || q.abs_diff(expected) > 1 {
            cdf_errors += 1;
        }
        acc = flux_cdf[i] as u64;
    }
    report.check(
        cdf_errors == 0,
        format!("{cdf_errors} flux CDF entries are inconsistent"),
    );
    report.check(
        light_count == 0 || state[ST_TOTAL_FLUX] as u64 == acc,
        format!("total {} vs CDF end {acc}", state[ST_TOTAL_FLUX]),
    );
    if light_count > 0 && acc > 0 {
        let total = acc as f64;
        let mut index_of = vec![u32::MAX; max];
        for (i, &id) in alive.iter().enumerate() {
            index_of[id as usize] = i as u32;
        }
        let mut bad_entries = 0usize;
        let mut valid_entries = 0usize;
        for entry in tiles.chunks_exact(8) {
            let id = entry[4];
            if id == u32::MAX {
                continue;
            }
            valid_entries += 1;
            if (id as usize) >= max || index_of[id as usize] == u32::MAX {
                bad_entries += 1;
                continue;
            }
            let i = index_of[id as usize] as usize;
            let below = if i == 0 { 0 } else { flux_cdf[i - 1] };
            let expected = (flux_cdf[i] - below) as f64 / total;
            let pmf = f32::from_bits(entry[7]) as f64;
            let rec = &records[id as usize * 8..id as usize * 8 + 8];
            if expected <= 0.0
                || (pmf - expected).abs() > 1e-6 * expected
                || entry[..4] != rec[..4]
                || entry[5] != rec[4]
                || entry[6] != rec[5]
            {
                bad_entries += 1;
            }
        }
        report.check(
            bad_entries == 0,
            format!("{bad_entries} of {valid_entries} light tile entries are inconsistent"),
        );
        report.info(format!(
            "lights {lit} with area of {alive_count}, max flux {max_flux:.3e}, quantized total {acc}"
        ));
    }

    // ---- images ----
    let pixels = (readback.width * readback.height) as usize;
    let mut valid_pixels = 0usize;
    let mut covered = 0usize;
    let mut bad_ids = 0usize;
    let mut bad_values = 0usize;
    let mut is_alive = vec![false; max];
    for &id in alive.iter() {
        is_alive[id as usize] = true;
    }
    for i in 0..pixels {
        let id = overlay[i];
        if id != u32::MAX && ((id as usize) >= max || !is_alive[id as usize]) {
            bad_ids += 1;
        }
        let has_surface = coverage[i * 4 + 2] > 0.0;
        if has_surface {
            valid_pixels += 1;
            if coverage[i * 4] >= 0.5 {
                covered += 1;
            }
        }
        for c in 0..3 {
            let d = e_dir[i * 4 + c];
            let n = e_ind[i * 4 + c];
            if !d.is_finite() || !n.is_finite() || d < 0.0 || n < 0.0 {
                bad_values += 1;
            }
        }
        let _ = albedo[i * 4];
    }
    report.check(
        bad_ids == 0,
        format!("{bad_ids} overlay pixels reference dead surfels"),
    );
    report.check(
        bad_values == 0,
        format!("{bad_values} negative or non-finite irradiance values"),
    );
    let coverage_ratio = covered as f64 / valid_pixels.max(1) as f64;
    report.info(format!(
        "coverage: {:.2}% of {valid_pixels} surface pixels have weight >= 0.5",
        coverage_ratio * 100.0
    ));
    if readback.frame_index >= 120 {
        report.check(
            coverage_ratio >= 0.85,
            format!("coverage {:.2}% below 85%", coverage_ratio * 100.0),
        );
    }
    report
}
