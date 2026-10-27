//! Compares a GI capture (`ART_RTIC_CAPTURE`) against its path traced reference.
//!
//! usage: artrtic-compare DIR TAG [--max-bias PCT] [--max-block-rmse PCT]
//!
//! Prints the bias and RMSE of the indirect irradiance (luminance) at pixel and
//! 16x16 block scale, the distribution of block ratios, and writes
//! `TAG_ratio.jpg` (log2 of the block ratio: blue = too dark, red = too bright,
//! white = match). Exits with 1 when a gate fails.

use std::path::PathBuf;

use artrtic::rendering::gi_capture::{RgbImage, downsample, error_stats, luminance};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: artrtic-compare DIR TAG [--max-bias PCT] [--max-block-rmse PCT]");
        std::process::exit(2);
    }
    let dir = PathBuf::from(&args[1]);
    let tag = &args[2];
    let mut max_bias = None;
    let mut max_block_rmse = None;
    let mut i = 3;
    while i + 1 < args.len() {
        let value = args[i + 1].parse::<f64>().expect("gate value");
        match args[i].as_str() {
            "--max-bias" => max_bias = Some(value),
            "--max-block-rmse" => max_block_rmse = Some(value),
            other => panic!("unknown option {other}"),
        }
        i += 2;
    }

    let load = |name: &str| {
        let path = dir.join(format!("{tag}_{name}.pfm"));
        RgbImage::read_pfm(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    };
    let e_ind = load("e_ind");
    let e_ref = load("e_ref");
    let albedo = load("albedo");
    let e_dir = load("e_dir");

    let mask: Vec<bool> = albedo.data.iter().map(|a| luminance(*a) > 0.0).collect();
    let pixel = error_stats(&e_ind, &e_ref, &mask);
    let block = 16;
    let est_b = downsample(&e_ind, block);
    let ref_b = downsample(&e_ref, block);
    let coverage_b = downsample(
        &albedo.map(|_, a| {
            if luminance(a) > 0.0 {
                [1.0; 3]
            } else {
                [0.0; 3]
            }
        }),
        block,
    );
    let block_mask: Vec<bool> = coverage_b.data.iter().map(|c| c[0] > 0.99).collect();
    let blocks = error_stats(&est_b, &ref_b, &block_mask);

    // Ratio distribution over blocks with meaningful indirect light.
    let mean_ref = blocks.mean_reference as f32;
    let mut ratios: Vec<f32> = Vec::new();
    for (k, ok) in block_mask.iter().enumerate() {
        let r = luminance(ref_b.data[k]);
        if *ok && r > 0.05 * mean_ref {
            ratios.push(luminance(est_b.data[k]) / r);
        }
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f32| ratios[((ratios.len() - 1) as f32 * p) as usize];
    let within = |t: f32| {
        ratios.iter().filter(|r| (*r - 1.0).abs() <= t).count() as f32 / ratios.len() as f32
    };

    // Direct vs indirect share, for scale.
    let dir_mean: f64 = e_dir
        .data
        .iter()
        .zip(mask.iter())
        .filter(|(_, m)| **m)
        .map(|(c, _)| luminance(*c) as f64)
        .sum::<f64>()
        / pixel.pixels.max(1) as f64;

    println!("{tag}: {} pixels, mean E_dir {:.3}", pixel.pixels, dir_mean);
    println!(
        "  E_ind mean {:.4} vs reference {:.4}: bias {:+.2}%",
        pixel.mean_estimate,
        pixel.mean_reference,
        pixel.relative_bias * 100.0
    );
    println!(
        "  relative RMSE: pixel {:.1}%, 16x16 blocks {:.1}%",
        pixel.relative_rmse * 100.0,
        blocks.relative_rmse * 100.0
    );
    if !ratios.is_empty() {
        println!(
            "  block ratio estimate/reference: p5 {:.3} p25 {:.3} median {:.3} p75 {:.3} p95 {:.3}; within 10% {:.1}%, within 25% {:.1}%",
            pct(0.05),
            pct(0.25),
            pct(0.5),
            pct(0.75),
            pct(0.95),
            within(0.10) * 100.0,
            within(0.25) * 100.0
        );
    }

    let heat = e_ind.map(|i, _| {
        let x = (i as u32 % e_ind.width) / block;
        let y = (i as u32 / e_ind.width) / block;
        if x >= est_b.width || y >= est_b.height {
            return [0.0; 3];
        }
        let k = (y * est_b.width + x) as usize;
        let r = luminance(ref_b.data[k]);
        if !block_mask[k] || r <= 0.0 {
            return [0.0; 3];
        }
        let t = (luminance(est_b.data[k]) / r).log2().clamp(-1.0, 1.0);
        if t >= 0.0 {
            [1.0, 1.0 - t, 1.0 - t]
        } else {
            [1.0 + t, 1.0 + t, 1.0]
        }
    });
    // write_jpeg tonemaps v/(1+v): pre-invert so the colors land as written.
    let heat = heat.map(|_, c| {
        c.map(|v| {
            let v = v.clamp(0.0, 0.999);
            v / (1.0 - v)
        })
    });
    let out = dir.join(format!("{tag}_ratio.jpg"));
    heat.write_jpeg(&out, 1.0).expect("write heatmap");
    println!("  heatmap: {}", out.display());

    let mut failed = false;
    if let Some(gate) = max_bias {
        let ok = (pixel.relative_bias * 100.0).abs() <= gate;
        println!(
            "  gate |bias| <= {gate}%: {}",
            if ok { "pass" } else { "FAIL" }
        );
        failed |= !ok;
    }
    if let Some(gate) = max_block_rmse {
        let ok = blocks.relative_rmse * 100.0 <= gate;
        println!(
            "  gate block RMSE <= {gate}%: {}",
            if ok { "pass" } else { "FAIL" }
        );
        failed |= !ok;
    }
    if failed {
        std::process::exit(1);
    }
}
