use std::sync::Arc;

use crate::shader_stage_access::ShaderStagesAccess;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NativeBindingType {
    Sampler,
    SampledImage,
    CombinedImageSampler,
    StorageImage,
    UniformTexelBuffer,
    StorageTexelBuffer,
    UniformBuffer,
    StorageBuffer,
    InputAttachment,
}

impl NativeBindingType {
    pub(crate) fn ash_descriptor_type(&self) -> ash::vk::DescriptorType {
        match self {
            Self::Sampler => ash::vk::DescriptorType::SAMPLER,
            Self::SampledImage => ash::vk::DescriptorType::SAMPLED_IMAGE,
            Self::CombinedImageSampler => ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            Self::StorageImage => ash::vk::DescriptorType::STORAGE_IMAGE,
            Self::UniformTexelBuffer => ash::vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            Self::StorageTexelBuffer => ash::vk::DescriptorType::STORAGE_TEXEL_BUFFER,
            Self::UniformBuffer => ash::vk::DescriptorType::UNIFORM_BUFFER,
            Self::StorageBuffer => ash::vk::DescriptorType::STORAGE_BUFFER,
            Self::InputAttachment => ash::vk::DescriptorType::INPUT_ATTACHMENT,
        }
    }
}

/**
 * Provided by VK_KHR_acceleration_structure
 */
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AccelerationStructureBindingType {
    AccelerationStructure,
}

impl AccelerationStructureBindingType {
    pub(crate) fn ash_descriptor_type(&self) -> ash::vk::DescriptorType {
        match self {
            Self::AccelerationStructure => ash::vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
pub struct BindingDescriptor {
    shader_access: ShaderStagesAccess,
    binding_type: BindingType,
    binding_point: u32,
    binding_count: u32,
}

impl BindingDescriptor {
    pub fn shader_access(&self) -> ShaderStagesAccess {
        self.shader_access
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::DescriptorSetLayoutBinding {
        ash::vk::DescriptorSetLayoutBinding::default()
            .binding(self.binding_point)
            .stage_flags(self.shader_access.ash_stage_access_mask())
            .descriptor_count(self.binding_count)
            .descriptor_type(self.binding_type.ash_descriptor_type())
    }

    pub fn binding_range(&self) -> (u32, u32) {
        (self.binding_point, self.binding_point + self.binding_count)
    }

    pub fn new(
        shader_access: ShaderStagesAccess,
        binding_type: BindingType,
        binding_point: u32,
        binding_count: u32,
    ) -> Arc<Self> {
        Arc::new(Self {
            shader_access,
            binding_type,
            binding_point,
            binding_count,
        })
    }
}
