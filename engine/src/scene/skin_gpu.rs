//! GPU layout shared with `engine/shaders/skin/animate_channels.comp` and the FBX cooker.

pub const MAX_ANIMATION_CHANNELS: usize = 128;
pub const MAX_ANIMATION_KEYS_PER_CHANNEL: usize = 64;

/// Column-major `mat4`, 80 bytes (matches UNIVR `SkeletonGPUElement`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SkeletonGpuElement {
    pub offset_matrix: [f32; 16],
    pub armature_node_index: u32,
    pub _padding: [u32; 3],
}

/// Column-major `mat4`, 80 bytes (matches UNIVR `ArmatureGPUElement`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ArmatureGpuElement {
    pub transform: [f32; 16],
    pub parent_index: u32,
    pub _padding: [u32; 3],
}

/// Fixed-size channel payload for one animated armature node.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AnimationGpuChannel {
    pub armature_element_index: u32,
    pub position_key_count: u32,
    pub position_key_times: [f32; MAX_ANIMATION_KEYS_PER_CHANNEL],
    pub position_key_value_x: [f32; MAX_ANIMATION_KEYS_PER_CHANNEL],
    pub position_key_value_y: [f32; MAX_ANIMATION_KEYS_PER_CHANNEL],
    pub position_key_value_z: [f32; MAX_ANIMATION_KEYS_PER_CHANNEL],
    pub rotation_key_count: u32,
    pub rotation_key_times: [f32; MAX_ANIMATION_KEYS_PER_CHANNEL],
    pub rotation_key_value_x: [f32; MAX_ANIMATION_KEYS_PER_CHANNEL],
    pub rotation_key_value_y: [f32; MAX_ANIMATION_KEYS_PER_CHANNEL],
    pub rotation_key_value_z: [f32; MAX_ANIMATION_KEYS_PER_CHANNEL],
    pub rotation_key_value_w: [f32; MAX_ANIMATION_KEYS_PER_CHANNEL],
    pub scaling_key_count: u32,
    pub scaling_key_times: [f32; MAX_ANIMATION_KEYS_PER_CHANNEL],
    pub scaling_key_value_x: [f32; MAX_ANIMATION_KEYS_PER_CHANNEL],
    pub scaling_key_value_y: [f32; MAX_ANIMATION_KEYS_PER_CHANNEL],
    pub scaling_key_value_z: [f32; MAX_ANIMATION_KEYS_PER_CHANNEL],
}

impl Default for AnimationGpuChannel {
    fn default() -> Self {
        Self {
            armature_element_index: 0,
            position_key_count: 0,
            position_key_times: [0.0; MAX_ANIMATION_KEYS_PER_CHANNEL],
            position_key_value_x: [0.0; MAX_ANIMATION_KEYS_PER_CHANNEL],
            position_key_value_y: [0.0; MAX_ANIMATION_KEYS_PER_CHANNEL],
            position_key_value_z: [0.0; MAX_ANIMATION_KEYS_PER_CHANNEL],
            rotation_key_count: 0,
            rotation_key_times: [0.0; MAX_ANIMATION_KEYS_PER_CHANNEL],
            rotation_key_value_x: [0.0; MAX_ANIMATION_KEYS_PER_CHANNEL],
            rotation_key_value_y: [0.0; MAX_ANIMATION_KEYS_PER_CHANNEL],
            rotation_key_value_z: [0.0; MAX_ANIMATION_KEYS_PER_CHANNEL],
            rotation_key_value_w: [0.0; MAX_ANIMATION_KEYS_PER_CHANNEL],
            scaling_key_count: 0,
            scaling_key_times: [0.0; MAX_ANIMATION_KEYS_PER_CHANNEL],
            scaling_key_value_x: [0.0; MAX_ANIMATION_KEYS_PER_CHANNEL],
            scaling_key_value_y: [0.0; MAX_ANIMATION_KEYS_PER_CHANNEL],
            scaling_key_value_z: [0.0; MAX_ANIMATION_KEYS_PER_CHANNEL],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn gpu_struct_sizes_match_univr() {
        assert_eq!(size_of::<SkeletonGpuElement>(), 80);
        assert_eq!(size_of::<ArmatureGpuElement>(), 80);
        assert_eq!(align_of::<SkeletonGpuElement>(), 4);
        assert_eq!(align_of::<ArmatureGpuElement>(), 4);
    }
}
