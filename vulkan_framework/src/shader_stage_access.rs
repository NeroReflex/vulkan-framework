use crate::shader_trait::ShaderType;

#[derive(Copy, Clone)]
pub struct ShaderStageAccessRayTracingKHR {}

impl ShaderStageAccessRayTracingKHR {
    pub(crate) fn ash_stage_access_mask(&self) -> ash::vk::ShaderStageFlags {
        ash::vk::ShaderStageFlags::empty()
    }
}

#[derive(Copy, Clone)]
pub struct ShaderStageAccess {
    compute: bool,
    vertex: bool,
    geometry: bool,
    fragment: bool,
    ray_tracing: ShaderStageAccessRayTracingKHR,
}

impl ShaderStageAccess {
    pub fn is_accessible_by(&self, shader_type: &ShaderType) -> bool {
        match shader_type {
            ShaderType::Compute => self.compute,
            ShaderType::Vertex => self.vertex,
            ShaderType::Geometry => self.geometry,
            ShaderType::Fragment => self.fragment,
        }
    }

    pub(crate) fn ash_stage_access_mask(&self) -> ash::vk::ShaderStageFlags {
        (match self.vertex {
            true => ash::vk::ShaderStageFlags::VERTEX,
            false => ash::vk::ShaderStageFlags::empty(),
        }) | (match self.geometry {
            true => ash::vk::ShaderStageFlags::GEOMETRY,
            false => ash::vk::ShaderStageFlags::empty(),
        }) | (match self.fragment {
            true => ash::vk::ShaderStageFlags::FRAGMENT,
            false => ash::vk::ShaderStageFlags::empty(),
        }) | (match self.compute {
            true => ash::vk::ShaderStageFlags::COMPUTE,
            false => ash::vk::ShaderStageFlags::empty(),
        }) | self.ray_tracing.ash_stage_access_mask()
    }
}