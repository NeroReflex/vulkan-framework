use crate::shader_trait::{ShaderType, ShaderTypeRayTracingKHR};

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct ShaderStageAccessRayTracingKHR {
    rgen: bool,
    miss: bool,
    callable: bool,
    closest_hit: bool,
    any_hit: bool,
    intersection: bool,
}

impl ShaderStageAccessRayTracingKHR {
    pub(crate) fn ash_stage_access_mask(&self) -> ash::vk::ShaderStageFlags {
        match self.rgen {
            true => ash::vk::ShaderStageFlags::RAYGEN_KHR,
            false => ash::vk::ShaderStageFlags::empty(),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct ShaderStageAccess {
    compute: bool,
    vertex: bool,
    geometry: bool,
    fragment: bool,
    ray_tracing: ShaderStageAccessRayTracingKHR,
}

impl ShaderStageAccess {
    /*
    // TODO: the proper way
    pub fn from() -> Self {
        Self {

        }
    }
    */

    pub fn raytracing() -> Self {
        Self {
            compute: false,
            vertex: false,
            geometry: false,
            fragment: false,
            ray_tracing: ShaderStageAccessRayTracingKHR {
                rgen: true,
                miss: true,
                callable: true,
                closest_hit: true,
                any_hit: true,
                intersection: true,
            },
        }
    }

    pub fn graphics() -> Self {
        Self {
            compute: false,
            vertex: true,
            geometry: true,
            fragment: true,
            ray_tracing: ShaderStageAccessRayTracingKHR {
                rgen: false,
                miss: false,
                callable: false,
                closest_hit: false,
                any_hit: false,
                intersection: false,
            },
        }
    }

    pub fn compute() -> Self {
        Self {
            compute: true,
            vertex: false,
            geometry: false,
            fragment: false,
            ray_tracing: ShaderStageAccessRayTracingKHR {
                rgen: false,
                miss: false,
                callable: false,
                closest_hit: false,
                any_hit: false,
                intersection: false,
            },
        }
    }

    pub fn is_accessible_by(&self, shader_type: &ShaderType) -> bool {
        match shader_type {
            ShaderType::Compute => self.compute,
            ShaderType::Vertex => self.vertex,
            ShaderType::Geometry => self.geometry,
            ShaderType::Fragment => self.fragment,
            ShaderType::RayTracingKHR(raytracing_khr) => match raytracing_khr {
                ShaderTypeRayTracingKHR::RayGen => self.ray_tracing.rgen,
                ShaderTypeRayTracingKHR::Miss => self.ray_tracing.miss,
                ShaderTypeRayTracingKHR::Callable => self.ray_tracing.callable,
                ShaderTypeRayTracingKHR::ClosestHit => self.ray_tracing.closest_hit,
                ShaderTypeRayTracingKHR::AnyHit => self.ray_tracing.any_hit,
                ShaderTypeRayTracingKHR::Intersection => self.ray_tracing.intersection,
            },
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
