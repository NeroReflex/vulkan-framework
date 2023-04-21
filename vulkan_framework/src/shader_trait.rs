use crate::{shader_layout_binding::BindingDescriptor, descriptor_set_layout::DescriptorSetLayoutDependant};

#[derive(Copy, Clone)]
pub enum ShaderType {
    Compute,
    Vertex,
    Geometry,
    Fragment,
}

pub trait ShaderTrait : DescriptorSetLayoutDependant {
    fn shader_type(&self) -> ShaderType;

    fn native_handle(&self) -> u64;
}

pub(crate) trait PrivateShaderTrait : ShaderTrait {
    fn ash_handle(&self) -> ash::vk::ShaderModule;
}