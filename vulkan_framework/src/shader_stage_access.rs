use crate::shader_trait::{ShaderType, ShaderTypeRayTracingKHR};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ShaderStageAccessInRayTracingKHR {
    RayGen,
    Callable,
    Miss,
    ClosestHit,
    AnyHit,
    Intersection,
}

impl From<&ShaderStageAccessInRayTracingKHR> for crate::ash::vk::ShaderStageFlags {
    fn from(value: &ShaderStageAccessInRayTracingKHR) -> Self {
        match value {
            ShaderStageAccessInRayTracingKHR::RayGen => ash::vk::ShaderStageFlags::RAYGEN_KHR,
            ShaderStageAccessInRayTracingKHR::Callable => ash::vk::ShaderStageFlags::CALLABLE_KHR,
            ShaderStageAccessInRayTracingKHR::Miss => ash::vk::ShaderStageFlags::MISS_KHR,
            ShaderStageAccessInRayTracingKHR::ClosestHit => {
                ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR
            }
            ShaderStageAccessInRayTracingKHR::AnyHit => ash::vk::ShaderStageFlags::ANY_HIT_KHR,
            ShaderStageAccessInRayTracingKHR::Intersection => {
                ash::vk::ShaderStageFlags::INTERSECTION_KHR
            }
        }
    }
}

impl From<ShaderStageAccessInRayTracingKHR> for crate::ash::vk::ShaderStageFlags {
    fn from(value: ShaderStageAccessInRayTracingKHR) -> Self {
        (&value).into()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ShaderStageAccessIn {
    Compute,
    Vertex,
    Geometry,
    Fragment,
    RayTracing(ShaderStageAccessInRayTracingKHR),
}

impl From<&ShaderStageAccessIn> for crate::ash::vk::ShaderStageFlags {
    fn from(value: &ShaderStageAccessIn) -> Self {
        match value {
            ShaderStageAccessIn::Compute => ash::vk::ShaderStageFlags::COMPUTE,
            ShaderStageAccessIn::Vertex => ash::vk::ShaderStageFlags::VERTEX,
            ShaderStageAccessIn::Geometry => ash::vk::ShaderStageFlags::GEOMETRY,
            ShaderStageAccessIn::Fragment => ash::vk::ShaderStageFlags::FRAGMENT,
            ShaderStageAccessIn::RayTracing(ray_tracing_khr) => ray_tracing_khr.into(),
        }
    }
}

impl From<ShaderStageAccessIn> for crate::ash::vk::ShaderStageFlags {
    fn from(value: ShaderStageAccessIn) -> Self {
        (&value).into()
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct ShaderStagesAccess(crate::ash::vk::ShaderStageFlags);

impl From<u32> for ShaderStagesAccess {
    fn from(value: u32) -> Self {
        Self(ash::vk::ShaderStageFlags::from_raw(value))
    }
}

impl From<crate::ash::vk::ShaderStageFlags> for ShaderStagesAccess {
    fn from(stages: crate::ash::vk::ShaderStageFlags) -> Self {
        Self(stages)
    }
}

impl From<ShaderStagesAccess> for crate::ash::vk::ShaderStageFlags {
    fn from(val: ShaderStagesAccess) -> Self {
        val.0.to_owned()
    }
}

impl From<&[ShaderStageAccessIn]> for ShaderStagesAccess {
    fn from(value: &[ShaderStageAccessIn]) -> Self {
        let mut access = ash::vk::ShaderStageFlags::empty();
        for flag in value.iter() {
            access |= flag.into()
        }

        Self(access)
    }
}

impl From<&[&ShaderStageAccessIn]> for ShaderStagesAccess {
    fn from(value: &[&ShaderStageAccessIn]) -> Self {
        let mut access = ash::vk::ShaderStageFlags::empty();
        for flag in value.iter() {
            access |= (*flag).into()
        }

        Self(access)
    }
}

impl ShaderStagesAccess {
    pub fn raytracing() -> Self {
        [
            ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::RayGen),
            ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::Callable),
            ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::Miss),
            ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::ClosestHit),
            ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::AnyHit),
            ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::Intersection),
        ]
        .as_slice()
        .into()
    }

    pub fn graphics() -> Self {
        Self(ash::vk::ShaderStageFlags::ALL_GRAPHICS)
    }

    pub fn compute() -> Self {
        [ShaderStageAccessIn::Compute].as_slice().into()
    }

    pub fn is_accessible_by(&self, shader_type: &ShaderType) -> bool {
        match shader_type {
            ShaderType::Compute => self.0.contains(ash::vk::ShaderStageFlags::COMPUTE),
            ShaderType::Vertex => self.0.contains(ash::vk::ShaderStageFlags::VERTEX),
            ShaderType::Geometry => self.0.contains(ash::vk::ShaderStageFlags::GEOMETRY),
            ShaderType::Fragment => self.0.contains(ash::vk::ShaderStageFlags::FRAGMENT),
            ShaderType::RayTracingKHR(raytracing_khr) => match raytracing_khr {
                ShaderTypeRayTracingKHR::RayGen => {
                    self.0.contains(ash::vk::ShaderStageFlags::RAYGEN_KHR)
                }
                ShaderTypeRayTracingKHR::Miss => {
                    self.0.contains(ash::vk::ShaderStageFlags::MISS_KHR)
                }
                ShaderTypeRayTracingKHR::Callable => {
                    self.0.contains(ash::vk::ShaderStageFlags::CALLABLE_KHR)
                }
                ShaderTypeRayTracingKHR::ClosestHit => {
                    self.0.contains(ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                }
                ShaderTypeRayTracingKHR::AnyHit => {
                    self.0.contains(ash::vk::ShaderStageFlags::ANY_HIT_KHR)
                }
                ShaderTypeRayTracingKHR::Intersection => {
                    self.0.contains(ash::vk::ShaderStageFlags::INTERSECTION_KHR)
                }
            },
        }
    }
}
