//! CPU-owned debug flags uploaded each frame for GPU shading and overlays.

/// Must match `engine/shaders/debug.glsl` and std140 layout (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DebugGpuBlock {
    pub flags: u32,
    pub padding: [u32; 7],
}

pub const DEBUG_FLAG_SHOW_SURFELS: u32 = 1 << 0;

#[derive(Clone, Debug, Default)]
pub struct DebugState {
    flags: u32,
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

    pub fn gpu_block(&self) -> DebugGpuBlock {
        DebugGpuBlock {
            flags: self.flags,
            padding: [0; 7],
        }
    }
}
