use crate::shader_stage_access::ShaderStageAccess;

pub struct PushConstanRange {
    offset: u32,
    size: u32,
    shader_access: ShaderStageAccess,
}

impl PushConstanRange {
    pub fn shader_access(&self) -> ShaderStageAccess {
        self.shader_access
    }
}
