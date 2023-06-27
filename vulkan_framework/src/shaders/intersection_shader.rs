use std::sync::Arc;

use crate::device::{Device, DeviceOwned};
use crate::instance::InstanceOwned;
use crate::prelude::{VulkanError, VulkanResult};

use crate::shader_trait::{PrivateShaderTrait, ShaderTrait, ShaderType, ShaderTypeRayTracingKHR};

pub struct IntersectionShader {
    device: Arc<Device>,
    module: ash::vk::ShaderModule,
}

impl Drop for IntersectionShader {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_shader_module(
                self.module,
                self.device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

impl DeviceOwned for IntersectionShader {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl ShaderTrait for IntersectionShader {
    fn shader_type(&self) -> ShaderType {
        ShaderType::RayTracingKHR(ShaderTypeRayTracingKHR::Intersection)
    }

    fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.module)
    }
}

impl PrivateShaderTrait for IntersectionShader {
    fn ash_handle(&self) -> ash::vk::ShaderModule {
        self.module
    }
}

impl IntersectionShader {
    pub fn new(device: Arc<Device>, code: &[u32]) -> VulkanResult<Arc<Self>> {
        let create_info = ash::vk::ShaderModuleCreateInfo::builder()
            .code(code)
            .build();

        match unsafe {
            device.ash_handle().create_shader_module(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        } {
            Ok(module) => Ok(Arc::new(Self {
                device,
                //push_constant_ranges: push_constant_ranges.iter().map(|cr| cr.clone()).collect(),
                //descriptor_bindings: descriptor_bindings.to_vec(),
                module,
            })),
            Err(err) => Err(VulkanError::Vulkan(
                err.as_raw(),
                Some(format!("Error creating the descriptor set: {}", err)),
            )),
        }
    }
}
