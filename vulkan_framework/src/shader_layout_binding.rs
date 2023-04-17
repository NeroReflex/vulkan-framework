use crate::shader_trait::ShaderType;

#[derive(Copy, Clone)]
pub enum NativeBindingType {
    Sampler,
    SampledImage,
    CombinedImageSampler,
    StorageImage,
    UniformTexelStorage,
    UniformTexelBuffer,
    UniformBuffer,
    StorageBuffer,
    InputAttachment,
}

impl NativeBindingType {
    pub(crate) fn ash_descriptor_type(&self) -> ash::vk::DescriptorType {
        todo!()
    }
}

/**
 * Provided by VK_KHR_acceleration_structure
 */
#[derive(Copy, Clone)]
pub enum AccelerationStructureBindingType {
    AccelerationStructure
}

impl AccelerationStructureBindingType {
    pub(crate) fn ash_descriptor_type(&self) -> ash::vk::DescriptorType {
        match self {
            Self::AccelerationStructure => ash::vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        }
    }
}

#[derive(Copy, Clone)]
pub enum BindingType {
    Native(NativeBindingType),
    AccelerationStructure(AccelerationStructureBindingType),
}

impl BindingType {
    pub(crate) fn ash_descriptor_type(&self) -> ash::vk::DescriptorType {
        match self {
            Self::Native(native) => native.ash_descriptor_type(),
            Self::AccelerationStructure(accel_s) => accel_s.ash_descriptor_type(),
        }
    }
}

#[derive(Copy, Clone)]
pub struct BindingDescriptorStageAccessRayTracingKHR {

}

impl BindingDescriptorStageAccessRayTracingKHR {
    pub(crate) fn ash_stage_access_mask(&self) -> ash::vk::ShaderStageFlags {
        ash::vk::ShaderStageFlags::empty()
    }
}

#[derive(Copy, Clone)]
pub struct BindingDescriptorStageAccess {
    vertex: bool,
    geometry: bool,
    fragment: bool,
    ray_tracing: BindingDescriptorStageAccessRayTracingKHR
}

impl BindingDescriptorStageAccess {
    pub(crate) fn ash_stage_access_mask(&self) -> ash::vk::ShaderStageFlags {
        (match self.vertex {
            true => ash::vk::ShaderStageFlags::VERTEX,
            false => ash::vk::ShaderStageFlags::empty()
        }) |
        (match self.geometry {
            true => ash::vk::ShaderStageFlags::GEOMETRY,
            false => ash::vk::ShaderStageFlags::empty()
        }) |
        (match self.fragment {
            true => ash::vk::ShaderStageFlags::FRAGMENT,
            false => ash::vk::ShaderStageFlags::empty()
        }) |
        (match self.fragment {
            true => ash::vk::ShaderStageFlags::COMPUTE,
            false => ash::vk::ShaderStageFlags::empty()
        }) |
        self.ray_tracing.ash_stage_access_mask()
    }
}

#[derive(Copy, Clone)]
pub struct BindingDescriptor {
    shader_access: BindingDescriptorStageAccess,
    binding_type: BindingType,
    binding_point: u32,
    binding_count: u32,
}

impl BindingDescriptor {
    pub(crate) fn ash_handle(&self) -> ash::vk::DescriptorSetLayoutBinding {
        ash::vk::DescriptorSetLayoutBinding::builder()
            .binding(self.binding_point)
            .stage_flags(self.shader_access.ash_stage_access_mask())
            .descriptor_count(self.binding_count)
            .descriptor_type(self.binding_type.ash_descriptor_type())
            .build()
    }
}