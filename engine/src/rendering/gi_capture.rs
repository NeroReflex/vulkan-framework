//! Writes GI readbacks to disk: linear PFM images plus tonemapped JPEG previews,
//! and prints the indirect-light error against the path traced reference.

use std::{io::Write, path::Path};

use crate::rendering::pipeline::global_illumination::{GiReadback, ReadbackItem};

/// RGB planes of an rgba32f image readback, row 0 at the top.
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<[f32; 3]>,
}

impl RgbImage {
    pub fn from_rgba(width: u32, height: u32, rgba: &[f32]) -> Self {
        let data = rgba.chunks_exact(4).map(|p| [p[0], p[1], p[2]]).collect();
        Self {
            width,
            height,
            data,
        }
    }

    pub fn map(&self, f: impl Fn(usize, [f32; 3]) -> [f32; 3]) -> Self {
        Self {
            width: self.width,
            height: self.height,
            data: self
                .data
                .iter()
                .enumerate()
                .map(|(i, p)| f(i, *p))
                .collect(),
        }
    }

    /// Portable float map, bottom row first as the format requires.
    pub fn write_pfm(&self, path: &Path) -> std::io::Result<()> {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        write!(file, "PF\n{} {}\n-1.0\n", self.width, self.height)?;
        for y in (0..self.height as usize).rev() {
            for x in 0..self.width as usize {
                for c in self.data[y * self.width as usize + x] {
                    file.write_all(&c.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    pub fn read_pfm(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let mut lines = 0;
        let mut header_end = 0;
        for (i, b) in bytes.iter().enumerate() {
            if *b == b'\n' {
                lines += 1;
                if lines == 3 {
                    header_end = i + 1;
                    break;
                }
            }
        }
        let header = String::from_utf8_lossy(&bytes[..header_end]).to_string();
        let mut parts = header.split_whitespace();
        let invalid = || std::io::Error::new(std::io::ErrorKind::InvalidData, "bad PFM header");
        if parts.next() != Some("PF") {
            return Err(invalid());
        }
        let width: u32 = parts
            .next()
            .and_then(|v| v.parse().ok())
            .ok_or_else(invalid)?;
        let height: u32 = parts
            .next()
            .and_then(|v| v.parse().ok())
            .ok_or_else(invalid)?;
        let floats: Vec<f32> = bytes[header_end..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if floats.len() < (width * height * 3) as usize {
            return Err(invalid());
        }
        let mut data = vec![[0.0f32; 3]; (width * height) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let src = ((height as usize - 1 - y) * width as usize + x) * 3;
                data[y * width as usize + x] = [floats[src], floats[src + 1], floats[src + 2]];
            }
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Reinhard-tonemapped sRGB JPEG with a fixed exposure.
    pub fn write_jpeg(&self, path: &Path, exposure: f32) -> std::io::Result<()> {
        let mut rgb = Vec::with_capacity(self.data.len() * 3);
        for p in self.data.iter() {
            for c in p {
                let v = (c * exposure).max(0.0);
                let mapped = v / (1.0 + v);
                let srgb = if mapped <= 0.003_130_8 {
                    mapped * 12.92
                } else {
                    1.055 * mapped.powf(1.0 / 2.4) - 0.055
                };
                rgb.push((srgb.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
        }
        let mut out = Vec::new();
        jpeg_encoder::Encoder::new(&mut out, 90)
            .encode(
                &rgb,
                self.width as u16,
                self.height as u16,
                jpeg_encoder::ColorType::Rgb,
            )
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        std::fs::write(path, out)
    }
}

pub fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Error statistics of an indirect irradiance estimate against a reference,
/// over the pixels where `mask` is true.
#[derive(Debug, Clone, Copy)]
pub struct ErrorStats {
    pub pixels: usize,
    pub mean_estimate: f64,
    pub mean_reference: f64,
    /// (mean estimate - mean reference) / mean reference
    pub relative_bias: f64,
    /// RMSE / mean reference (luminance)
    pub relative_rmse: f64,
}

pub fn error_stats(estimate: &RgbImage, reference: &RgbImage, mask: &[bool]) -> ErrorStats {
    let mut n = 0usize;
    let (mut se, mut sr, mut sq) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..estimate.data.len() {
        if !mask[i] {
            continue;
        }
        let e = luminance(estimate.data[i]) as f64;
        let r = luminance(reference.data[i]) as f64;
        if !e.is_finite() || !r.is_finite() {
            continue;
        }
        n += 1;
        se += e;
        sr += r;
        sq += (e - r) * (e - r);
    }
    let nf = n.max(1) as f64;
    let mean_r = sr / nf;
    ErrorStats {
        pixels: n,
        mean_estimate: se / nf,
        mean_reference: mean_r,
        relative_bias: if mean_r > 0.0 {
            (se / nf - mean_r) / mean_r
        } else {
            0.0
        },
        relative_rmse: if mean_r > 0.0 {
            (sq / nf).sqrt() / mean_r
        } else {
            0.0
        },
    }
}

/// Mean over `block` x `block` tiles; used to compare low-frequency content.
pub fn downsample(image: &RgbImage, block: u32) -> RgbImage {
    let w = image.width / block;
    let h = image.height / block;
    let mut data = vec![[0.0f32; 3]; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for dy in 0..block {
                for dx in 0..block {
                    let p = image.data[((y * block + dy) * image.width + x * block + dx) as usize];
                    for c in 0..3 {
                        acc[c] += p[c];
                    }
                }
            }
            let inv = 1.0 / (block * block) as f32;
            data[(y * w + x) as usize] = [acc[0] * inv, acc[1] * inv, acc[2] * inv];
        }
    }
    RgbImage {
        width: w,
        height: h,
        data,
    }
}

pub fn write_capture(dir: &Path, readback: &GiReadback) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let (w, h) = (readback.width, readback.height);
    let e_dir = RgbImage::from_rgba(w, h, &readback.f32s(ReadbackItem::EDir));
    let e_ind = RgbImage::from_rgba(w, h, &readback.f32s(ReadbackItem::EInd));
    let e_ref = RgbImage::from_rgba(w, h, &readback.f32s(ReadbackItem::ERef));
    let albedo = RgbImage::from_rgba(w, h, &readback.f32s(ReadbackItem::Albedo));
    let overlay = readback.u32s(ReadbackItem::Overlay);
    let reference_samples: Vec<f32> = readback
        .f32s(ReadbackItem::ERef)
        .chunks_exact(4)
        .map(|p| p[3])
        .collect();

    let accumulated_frames = RgbImage {
        width: w,
        height: h,
        data: readback
            .f32s(ReadbackItem::EInd)
            .chunks_exact(4)
            .map(|p| [p[3]; 3])
            .collect(),
    };

    let tag = format!("f{:05}", readback.frame_index);
    e_dir.write_pfm(&dir.join(format!("{tag}_e_dir.pfm")))?;
    e_ind.write_pfm(&dir.join(format!("{tag}_e_ind.pfm")))?;
    accumulated_frames.write_pfm(&dir.join(format!("{tag}_e_ind_frames.pfm")))?;
    e_ref.write_pfm(&dir.join(format!("{tag}_e_ref.pfm")))?;
    albedo.write_pfm(&dir.join(format!("{tag}_albedo.pfm")))?;

    let shade = |irr: &RgbImage| {
        irr.map(|i, e| {
            let a = albedo.data[i];
            [a[0] * e[0], a[1] * e[1], a[2] * e[2]]
        })
    };
    let total = |i: usize, extra: &RgbImage| -> [f32; 3] {
        let a = albedo.data[i];
        let d = e_dir.data[i];
        let x = extra.data[i];
        [
            a[0] * (d[0] + x[0]),
            a[1] * (d[1] + x[1]),
            a[2] * (d[2] + x[2]),
        ]
    };
    let final_image = e_dir.map(|i, _| total(i, &e_ind));
    let reference_image = e_dir.map(|i, _| total(i, &e_ref));
    let exposure = 0.05;
    final_image.write_jpeg(&dir.join(format!("{tag}_final.jpg")), exposure)?;
    reference_image.write_jpeg(&dir.join(format!("{tag}_reference.jpg")), exposure)?;
    shade(&e_ind).write_jpeg(&dir.join(format!("{tag}_indirect.jpg")), exposure * 4.0)?;
    shade(&e_ref).write_jpeg(
        &dir.join(format!("{tag}_reference_indirect.jpg")),
        exposure * 4.0,
    )?;
    let ids = e_dir.map(|i, _| {
        let id = overlay[i];
        if id == u32::MAX {
            [0.0, 0.0, 0.0]
        } else {
            let hsh = crate::rendering::pipeline::gi_math::hash_u32(id);
            [
                (hsh & 0xFF) as f32 / 255.0,
                ((hsh >> 8) & 0xFF) as f32 / 255.0,
                ((hsh >> 16) & 0xFF) as f32 / 255.0,
            ]
        }
    });
    ids.write_jpeg(&dir.join(format!("{tag}_surfels.jpg")), 4.0)?;

    let mask: Vec<bool> = (0..e_dir.data.len())
        .map(|i| luminance(albedo.data[i]) > 0.0)
        .collect();
    let stats = error_stats(&e_ind, &e_ref, &mask);
    let block = 16;
    let coarse_mask: Vec<bool> = vec![true; ((w / block) * (h / block)) as usize];
    let coarse = error_stats(
        &downsample(&e_ind, block),
        &downsample(&e_ref, block),
        &coarse_mask,
    );
    let samples = reference_samples.iter().cloned().fold(0.0f32, f32::max);
    println!(
        "CAPTURE {tag}: reference frames {samples:.0}, pixels {}, mean E_ind {:.4} vs reference {:.4}, bias {:+.2}%, rel. RMSE {:.1}% (16x16 blocks: {:.1}%)",
        stats.pixels,
        stats.mean_estimate,
        stats.mean_reference,
        stats.relative_bias * 100.0,
        stats.relative_rmse * 100.0,
        coarse.relative_rmse * 100.0,
    );
    Ok(())
}
