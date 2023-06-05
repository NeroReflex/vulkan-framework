use crate::{device::DeviceOwned};

#[derive(Copy, Clone)]
pub enum ShaderType {
    Compute,
    Vertex,
    Geometry,
    Fragment,
}

pub trait ShaderTrait: DeviceOwned {
    fn shader_type(&self) -> ShaderType;

    fn native_handle(&self) -> u64;
}

pub(crate) trait PrivateShaderTrait: ShaderTrait {
    fn ash_handle(&self) -> ash::vk::ShaderModule;
}
