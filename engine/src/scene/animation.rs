//! CPU-side animation play state for skinned object archives.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AnimationClipInfo {
    pub name: String,
    pub duration_ticks: f64,
    pub ticks_per_second: f64,
    pub channel_count: u32,
}

#[derive(Debug, Clone, Default)]
pub struct AnimationPlayState {
    pub clip: Option<String>,
    pub time_seconds: f32,
}

impl AnimationPlayState {
    pub fn advance(&mut self, delta_seconds: f32, clips: &HashMap<String, AnimationClipInfo>) {
        let Some(name) = self.clip.clone() else {
            return;
        };
        let Some(clip) = clips.get(&name) else {
            self.clip = None;
            return;
        };
        self.time_seconds += delta_seconds;
        let duration_seconds = if clip.ticks_per_second > 0.0 {
            clip.duration_ticks / clip.ticks_per_second
        } else {
            clip.duration_ticks
        };
        if duration_seconds > 0.0 && self.time_seconds >= duration_seconds as f32 {
            self.clip = None;
            self.time_seconds = 0.0;
        }
    }

    pub fn time_in_ticks(&self, clips: &HashMap<String, AnimationClipInfo>) -> Option<f32> {
        let name = self.clip.as_ref()?;
        let clip = clips.get(name)?;
        let tps = if clip.ticks_per_second > 0.0 {
            clip.ticks_per_second as f32
        } else {
            1.0
        };
        Some(self.time_seconds * tps)
    }

    pub fn active_channel_count(&self, clips: &HashMap<String, AnimationClipInfo>) -> u32 {
        let Some(name) = self.clip.as_ref() else {
            return 0;
        };
        clips.get(name).map(|c| c.channel_count).unwrap_or(0)
    }
}
