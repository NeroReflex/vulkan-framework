use std::sync::Arc;

use crate::shader_stage_access::ShaderStagesAccess;

pub struct PushConstanRange {
    offset: u32,
    size: u32,
    shader_access: ShaderStagesAccess,
}

impl From<PushConstanRange> for crate::ash::vk::PushConstantRange {
    fn from(range: PushConstanRange) -> crate::ash::vk::PushConstantRange {
        (&range).into()
    }
}

impl From<&PushConstanRange> for crate::ash::vk::PushConstantRange {
    fn from(range: &PushConstanRange) -> crate::ash::vk::PushConstantRange {
        ash::vk::PushConstantRange::default()
            .offset(range.offset)
            .size(range.size)
            .stage_flags(range.shader_access.into())
    }
}

impl PushConstanRange {
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
