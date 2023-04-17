use crate::shader_layout_binding::BindingDescriptor;

#[derive(Copy, Clone)]
pub enum ShaderType {
    Compute,
    Vertex,
    Geometry,
    Fragment,
}

pub trait ShaderTrait {
    fn shader_type() -> ShaderType;

    fn native_handle() -> u64;

    fn binding_descriptors() -> Vec<BindingDescriptor>;
}