use crate::{device::DeviceOwned, shader_layout_binding::BindingDescriptorDependant};

#[derive(Copy, Clone)]
pub enum ShaderType {
    Compute,
    Vertex,
    Geometry,
    Fragment,
}

pub trait ShaderTrait: DeviceOwned + BindingDescriptorDependant {
    fn shader_type(&self) -> ShaderType;

    fn native_handle(&self) -> u64;
}

pub(crate) trait PrivateShaderTrait: ShaderTrait {
    fn ash_handle(&self) -> ash::vk::ShaderModule;
}
