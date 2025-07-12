use std::sync::Arc;

use crate::shader_stage_access::ShaderStagesAccess;

pub struct PushConstanRange {
    offset: u32,
    size: u32,
    shader_access: ShaderStagesAccess,
}

impl PushConstanRange {
    pub(crate) fn ash_handle(&self) -> ash::vk::PushConstantRange {
        ash::vk::PushConstantRange::default()
            .offset(self.offset)
            .size(self.size)
            .stage_flags(self.shader_access.ash_stage_access_mask())
    }

    pub fn shader_access(&self) -> ShaderStagesAccess {
        self.shader_access
    }

    pub fn new(offset: u32, size: u32, shader_access: ShaderStagesAccess) -> Arc<Self> {
        Arc::new(Self {
            offset,
            size,
            shader_access,
        })
    }
}
