//! Per-frame bookkeeping of the GI pipeline on the CPU: the status block,
//! history invalidation, GPU timings, captures (`ART_RTIC_CAPTURE`) and the
//! self-test (`ART_RTIC_SELFTEST`).

use std::path::PathBuf;

use crate::rendering::{
    debug_state::DebugState,
    pipeline::global_illumination::{
        GI_FLAG_ACCUMULATE, GI_FLAG_CAPTURE, GI_FLAG_CONVERGE, GI_FLAG_MULTIBOUNCE,
        GI_FLAG_REFERENCE, GI_FLAG_SPATIAL, GI_FLAG_SPAWN, GI_FLAG_TEMPORAL, GILighting,
        GiReadback,
    },
};

/// `status_buffer` in `engine/shaders/status.glsl` (std140, 176 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GiStatusBlock {
    pub gi_reuse_frames: u32,
    pub frame_index: u32,
    pub still_frames: u32,
    pub flags: u32,
    pub reference_bounces: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub pixel_angle: f32,
    pub eye_position: [f32; 4],
    pub prev_view: [f32; 16],
    pub prev_proj: [f32; 16],
}

pub const GI_STATUS_BYTES: u64 = 256;

impl GiStatusBlock {
    pub fn words(&self) -> Vec<u32> {
        let mut words = vec![
            self.gi_reuse_frames,
            self.frame_index,
            self.still_frames,
            self.flags,
            self.reference_bounces,
            self.render_width,
            self.render_height,
            self.pixel_angle.to_bits(),
        ];
        words.extend(self.eye_position.iter().map(|v| v.to_bits()));
        words.extend(self.prev_view.iter().map(|v| v.to_bits()));
        words.extend(self.prev_proj.iter().map(|v| v.to_bits()));
        words
    }
}

pub fn mat4_columns(m: &glm::Mat4) -> [f32; 16] {
    let mut out = [0.0; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[c * 4 + r] = m[c][r];
        }
    }
    out
}

enum ReadbackPurpose {
    Capture,
    Selftest,
}

struct TimingAccumulator {
    sums: Vec<(&'static str, f64)>,
    frames: u32,
}

pub struct GiDriver {
    frame_index: u32,
    still_frames: u32,
    last_pose: Option<[f32; 32]>,
    prev_view: [f32; 16],
    prev_proj: [f32; 16],
    has_prev: bool,
    print_timings: bool,
    timings: TimingAccumulator,
    capture_dir: Option<PathBuf>,
    capture_frames: Vec<u32>,
    selftest_frames: Vec<u32>,
    pending: Option<ReadbackPurpose>,
    selftest_failures: u32,
    selftest_runs: u32,
}

fn parse_frames(list: &str) -> Vec<u32> {
    list.split(',')
        .filter_map(|v| v.trim().parse::<u32>().ok())
        .collect()
}

impl GiDriver {
    pub fn from_env() -> Self {
        let (capture_dir, capture_frames) = match std::env::var("ART_RTIC_CAPTURE") {
            Ok(value) => match value.split_once('@') {
                Some((dir, frames)) => (Some(PathBuf::from(dir)), parse_frames(frames)),
                None => (Some(PathBuf::from(value)), vec![600]),
            },
            Err(_) => (None, vec![]),
        };
        let selftest_frames = std::env::var("ART_RTIC_SELFTEST")
            .map(|v| parse_frames(&v))
            .unwrap_or_default();
        Self {
            frame_index: 0,
            still_frames: 0,
            last_pose: None,
            prev_view: [0.0; 16],
            prev_proj: [0.0; 16],
            has_prev: false,
            print_timings: std::env::var("ART_RTIC_TIMINGS").is_ok_and(|v| v != "0"),
            timings: TimingAccumulator {
                sums: vec![],
                frames: 0,
            },
            capture_dir,
            capture_frames,
            selftest_frames,
            pending: None,
            selftest_failures: 0,
            selftest_runs: 0,
        }
    }

    pub fn readback_enabled(&self) -> bool {
        self.capture_dir.is_some() || !self.selftest_frames.is_empty()
    }

    pub fn frame_index(&self) -> u32 {
        self.frame_index
    }

    pub fn selftest_failures(&self) -> u32 {
        self.selftest_failures
    }

    /// Builds the status block of the frame about to be recorded and decides
    /// whether this frame copies the GI state back to the host.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_frame(
        &mut self,
        gi: &GILighting,
        debug: &DebugState,
        reuse: u32,
        scene_animating: bool,
        view: &glm::Mat4,
        proj: &glm::Mat4,
        eye: glm::Vec3,
        render_width: u32,
        render_height: u32,
    ) -> GiStatusBlock {
        let view_cols = mat4_columns(view);
        let proj_cols = mat4_columns(proj);
        let mut pose = [0.0f32; 32];
        pose[..16].copy_from_slice(&view_cols);
        pose[16..].copy_from_slice(&proj_cols);
        if reuse == 0 || scene_animating || self.last_pose != Some(pose) {
            self.still_frames = 0;
        } else {
            self.still_frames = self.still_frames.saturating_add(1);
        }
        self.last_pose = Some(pose);
        if !self.has_prev || reuse == 0 {
            self.prev_view = view_cols;
            self.prev_proj = proj_cols;
        }

        let settings = debug.gi;
        let mut flags = 0;
        if settings.multibounce {
            flags |= GI_FLAG_MULTIBOUNCE;
        }
        if settings.temporal_reuse {
            flags |= GI_FLAG_TEMPORAL;
        }
        if settings.spatial_reuse {
            flags |= GI_FLAG_SPATIAL;
        }
        if settings.accumulate {
            flags |= GI_FLAG_ACCUMULATE;
        }
        if settings.spawn {
            flags |= GI_FLAG_SPAWN;
        }
        if settings.converge {
            flags |= GI_FLAG_CONVERGE;
        }
        let capture_now = self.capture_frames.contains(&self.frame_index);
        let selftest_now = self.selftest_frames.contains(&self.frame_index);
        if debug.reference_active() || self.capture_dir.is_some() {
            flags |= GI_FLAG_REFERENCE;
        }
        if self.pending.is_none() && (capture_now || selftest_now) && gi.request_readback() {
            flags |= GI_FLAG_CAPTURE;
            self.pending = Some(if capture_now {
                ReadbackPurpose::Capture
            } else {
                ReadbackPurpose::Selftest
            });
        }

        // tan(fov/2) = 1 / proj[1][1]; one pixel spans fov / height at the center.
        let tan_half = 1.0 / proj[1][1].abs().max(1e-6);
        let pixel_angle = 2.0 * tan_half.atan() / render_height as f32;

        GiStatusBlock {
            gi_reuse_frames: reuse,
            frame_index: self.frame_index,
            still_frames: self.still_frames,
            flags,
            reference_bounces: if settings.multibounce { 8 } else { 1 },
            render_width,
            render_height,
            pixel_angle,
            eye_position: [eye.x, eye.y, eye.z, 1.0],
            prev_view: self.prev_view,
            prev_proj: self.prev_proj,
        }
    }

    /// Call once the frame has been recorded with `status`.
    pub fn end_frame(&mut self, view: &glm::Mat4, proj: &glm::Mat4) {
        self.prev_view = mat4_columns(view);
        self.prev_proj = mat4_columns(proj);
        self.has_prev = true;
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// Call after the fence of `slot` has been waited.
    pub fn after_fence(&mut self, gi: &GILighting, slot: usize) {
        if let Some(timings) = gi.read_timings(slot) {
            if self.timings.sums.is_empty() {
                self.timings.sums = timings.iter().map(|(l, _)| (*l, 0.0)).collect();
            }
            for (sum, (_, ms)) in self.timings.sums.iter_mut().zip(timings.iter()) {
                sum.1 += ms;
            }
            self.timings.frames += 1;
            if self.timings.frames == 120 {
                if self.print_timings {
                    let total: f64 = self.timings.sums.iter().map(|(_, s)| s).sum::<f64>() / 120.0;
                    let parts: Vec<String> = self
                        .timings
                        .sums
                        .iter()
                        .map(|(label, s)| format!("{label} {:.3}", s / 120.0))
                        .collect();
                    println!(
                        "GI GPU ms (avg of 120): total {total:.3} | {}",
                        parts.join(" | ")
                    );
                }
                for sum in self.timings.sums.iter_mut() {
                    sum.1 = 0.0;
                }
                self.timings.frames = 0;
            }
        }

        if self.pending.is_none() {
            return;
        }
        let Some(readback) = gi.take_readback(slot) else {
            return;
        };
        match self.pending.take() {
            Some(ReadbackPurpose::Capture) => {
                if let Some(dir) = &self.capture_dir {
                    if let Err(err) = crate::rendering::gi_capture::write_capture(dir, &readback) {
                        eprintln!("GI capture failed: {err}");
                    }
                }
                // A capture frame doubles as a self-test frame when both were requested.
                if self.selftest_frames.contains(&readback.frame_index) {
                    self.run_selftest(&readback);
                }
            }
            Some(ReadbackPurpose::Selftest) => self.run_selftest(&readback),
            None => {}
        }
    }

    fn run_selftest(&mut self, readback: &GiReadback) {
        let report = crate::rendering::gi_selftest::check(readback);
        self.selftest_runs += 1;
        for line in report.lines.iter() {
            println!("SELFTEST frame {}: {line}", readback.frame_index);
        }
        if report.failures > 0 {
            self.selftest_failures += report.failures;
            println!(
                "SELFTEST frame {}: FAILED ({} checks)",
                readback.frame_index, report.failures
            );
        } else {
            println!("SELFTEST frame {}: PASSED", readback.frame_index);
        }
    }
}
