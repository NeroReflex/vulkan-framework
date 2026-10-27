//! CPU-owned debug flags uploaded each frame for GPU shading and overlays.

/// Must match `engine/shaders/debug.glsl` and std140 layout (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DebugGpuBlock {
    pub flags: u32,
    pub view_mode: u32,
    pub exposure: f32,
    /// `AoAlgorithm` as `u32`.
    pub ao_algorithm: u32,
    pub padding: [u32; 4],
}

pub const DEBUG_FLAG_SHOW_SURFELS: u32 = 1 << 0;
/// Multiply indirect light by the ambient-occlusion image.
pub const DEBUG_FLAG_APPLY_AO: u32 = 1 << 1;

/// Which pass fills the ambient-occlusion image. Must match the dispatch in
/// `AmbientOcclusion::record_rendering_commands`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum AoAlgorithm {
    #[default]
    Ssao = 0,
    Rtao = 1,
}

/// What the final composite shows. Must match `VIEW_*` in `debug.glsl`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum ViewMode {
    /// albedo * (E_dir + E_ind)
    #[default]
    Final = 0,
    /// albedo shading tinted by the dominant surfel id
    SurfelIds = 1,
    /// partition-of-unity weight sum: red = uncovered, green = 1, blue = over-covered
    Coverage = 2,
    /// albedo * E_ind
    Indirect = 3,
    /// E_ind (irradiance, no albedo)
    IndirectIrradiance = 4,
    /// albedo * (E_dir + E_ref), the path traced reference
    Reference = 5,
    /// albedo * E_ref
    ReferenceIndirect = 6,
    /// relative luminance error of E_ind against E_ref: red = too bright, blue = too dark
    Difference = 7,
    /// the ambient-occlusion image: white = open, black = occluded
    AmbientOcclusion = 8,
}

impl AoAlgorithm {
    pub const ALL: [AoAlgorithm; 2] = [AoAlgorithm::Ssao, AoAlgorithm::Rtao];

    pub fn label(self) -> &'static str {
        match self {
            AoAlgorithm::Ssao => "SSAO",
            AoAlgorithm::Rtao => "RTAO",
        }
    }

    /// `ART_RTIC_AO=ssao|rtao` selects the startup algorithm. Anything else is SSAO.
    pub fn from_env() -> Self {
        match std::env::var("ART_RTIC_AO") {
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "" | "ssao" => AoAlgorithm::Ssao,
                "rtao" => AoAlgorithm::Rtao,
                other => {
                    eprintln!("ART_RTIC_AO: unknown algorithm {other}");
                    AoAlgorithm::Ssao
                }
            },
            Err(_) => AoAlgorithm::Ssao,
        }
    }
}

impl ViewMode {
    pub const ALL: [ViewMode; 9] = [
        ViewMode::Final,
        ViewMode::SurfelIds,
        ViewMode::Coverage,
        ViewMode::Indirect,
        ViewMode::IndirectIrradiance,
        ViewMode::Reference,
        ViewMode::ReferenceIndirect,
        ViewMode::Difference,
        ViewMode::AmbientOcclusion,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ViewMode::Final => "Final",
            ViewMode::SurfelIds => "Surfel ids",
            ViewMode::Coverage => "Surfel coverage",
            ViewMode::Indirect => "Indirect (albedo x E_ind)",
            ViewMode::IndirectIrradiance => "Indirect irradiance",
            ViewMode::Reference => "Reference",
            ViewMode::ReferenceIndirect => "Reference indirect",
            ViewMode::Difference => "Difference vs reference",
            ViewMode::AmbientOcclusion => "Ambient occlusion",
        }
    }

    pub fn needs_reference(self) -> bool {
        matches!(
            self,
            ViewMode::Reference | ViewMode::ReferenceIndirect | ViewMode::Difference
        )
    }
}

/// Switches of the GI pipeline. Toggling any of them restarts the GI history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GiSettings {
    pub multibounce: bool,
    pub temporal_reuse: bool,
    pub spatial_reuse: bool,
    pub accumulate: bool,
    pub spawn: bool,
    /// Let the surfel cache settle while the view is still, so the image converges.
    pub converge: bool,
    /// Run the path traced reference every frame (expensive).
    pub reference: bool,
}

impl Default for GiSettings {
    fn default() -> Self {
        Self {
            multibounce: true,
            temporal_reuse: true,
            spatial_reuse: true,
            accumulate: true,
            spawn: true,
            converge: true,
            reference: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DebugState {
    flags: u32,
    view_mode: ViewMode,
    apply_ao: bool,
    ao_algorithm: AoAlgorithm,
    pub gi: GiSettings,
}

impl GiSettings {
    /// Defaults with `ART_RTIC_GI_DISABLE=multibounce,temporal,spatial,accumulate,spawn,converge`
    /// switches turned off and `ART_RTIC_GI_REFERENCE=1` turning the reference on.
    pub fn from_env() -> Self {
        let mut settings = Self::default();
        if let Ok(list) = std::env::var("ART_RTIC_GI_DISABLE") {
            for name in list.split(',').map(str::trim) {
                match name {
                    "multibounce" => settings.multibounce = false,
                    "temporal" => settings.temporal_reuse = false,
                    "spatial" => settings.spatial_reuse = false,
                    "accumulate" => settings.accumulate = false,
                    "spawn" => settings.spawn = false,
                    "converge" => settings.converge = false,
                    "" => {}
                    other => eprintln!("ART_RTIC_GI_DISABLE: unknown switch {other}"),
                }
            }
        }
        if std::env::var("ART_RTIC_GI_REFERENCE").is_ok_and(|v| v != "0") {
            settings.reference = true;
        }
        settings
    }
}

impl ViewMode {
    /// `ART_RTIC_VIEW=<index into ViewMode::ALL>` selects the startup view.
    pub fn from_env() -> Self {
        std::env::var("ART_RTIC_VIEW")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .and_then(|index| Self::ALL.get(index).copied())
            .unwrap_or_default()
    }
}

impl Default for DebugState {
    fn default() -> Self {
        Self {
            flags: 0,
            view_mode: ViewMode::from_env(),
            apply_ao: std::env::var("ART_RTIC_AO_APPLY").is_ok_and(|v| v != "0"),
            ao_algorithm: AoAlgorithm::from_env(),
            gi: GiSettings::from_env(),
        }
    }
}

impl DebugState {
    pub fn show_surfels(&self) -> bool {
        self.flags & DEBUG_FLAG_SHOW_SURFELS != 0
    }

    pub fn set_show_surfels(&mut self, enabled: bool) {
        if enabled {
            self.flags |= DEBUG_FLAG_SHOW_SURFELS;
        } else {
            self.flags &= !DEBUG_FLAG_SHOW_SURFELS;
        }
    }

    pub fn view_mode(&self) -> ViewMode {
        self.view_mode
    }

    pub fn set_view_mode(&mut self, mode: ViewMode) {
        self.view_mode = mode;
    }

    pub fn apply_ao(&self) -> bool {
        self.apply_ao
    }

    pub fn set_apply_ao(&mut self, enabled: bool) {
        self.apply_ao = enabled;
    }

    pub fn ao_algorithm(&self) -> AoAlgorithm {
        self.ao_algorithm
    }

    pub fn set_ao_algorithm(&mut self, algorithm: AoAlgorithm) {
        self.ao_algorithm = algorithm;
    }

    /// The reference path tracer runs when requested or when the view needs it.
    pub fn reference_active(&self) -> bool {
        self.gi.reference || self.view_mode.needs_reference()
    }

    pub fn gpu_block(&self) -> DebugGpuBlock {
        let view_mode = if self.view_mode == ViewMode::Final && self.show_surfels() {
            ViewMode::SurfelIds
        } else {
            self.view_mode
        };
        let mut flags = self.flags;
        if self.apply_ao {
            flags |= DEBUG_FLAG_APPLY_AO;
        }
        DebugGpuBlock {
            flags,
            view_mode: view_mode as u32,
            exposure: 1.0,
            ao_algorithm: self.ao_algorithm as u32,
            padding: [0; 4],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_block_matches_std140() {
        assert_eq!(std::mem::size_of::<DebugGpuBlock>(), 32);
        assert_eq!(std::mem::align_of::<DebugGpuBlock>(), 4);
    }

    #[test]
    fn apply_ao_is_its_own_flag() {
        let mut debug = DebugState {
            flags: DEBUG_FLAG_SHOW_SURFELS,
            view_mode: ViewMode::Final,
            apply_ao: true,
            ao_algorithm: AoAlgorithm::Rtao,
            gi: GiSettings::default(),
        };
        let block = debug.gpu_block();
        assert_eq!(
            block.flags & DEBUG_FLAG_SHOW_SURFELS,
            DEBUG_FLAG_SHOW_SURFELS
        );
        assert_eq!(block.flags & DEBUG_FLAG_APPLY_AO, DEBUG_FLAG_APPLY_AO);
        assert_eq!(block.ao_algorithm, AoAlgorithm::Rtao as u32);
        debug.set_apply_ao(false);
        assert_eq!(debug.gpu_block().flags & DEBUG_FLAG_APPLY_AO, 0);
    }
}
