use std::sync::Arc;

use crate::shader_stage_access::ShaderStageAccess;

pub struct PushConstanRange {
    offset: u32,
    size: u32,
    shader_access: ShaderStageAccess,
}

impl PushConstanRange {
    pub(crate) fn ash_handle(&self) -> ash::vk::PushConstantRange {
        ash::vk::PushConstantRange::builder()
            .offset(self.offset)
            .size(self.size)
            .stage_flags(self.shader_access.ash_stage_access_mask())
            .build()
    }

    pub fn shader_access(&self) -> ShaderStageAccess {
        self.shader_access
    }

    pub fn new(
        offset: u32,
        size: u32,
        shader_access: ShaderStageAccess,
    ) -> Arc<Self> {
        Arc::new(
            Self {
                offset,
                size,
                shader_access,
            }
        )
    }
}
