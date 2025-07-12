use std::sync::Arc;

use crate::device::{Device, DeviceOwned};
use crate::instance::InstanceOwned;
use crate::prelude::{VulkanError, VulkanResult};

use crate::shader_trait::{PrivateShaderTrait, ShaderTrait, ShaderType, ShaderTypeRayTracingKHR};

pub struct ClosestHitShader {
    device: Arc<Device>,
    module: ash::vk::ShaderModule,
}

impl Drop for ClosestHitShader {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_shader_module(
                self.module,
                self.device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

impl DeviceOwned for ClosestHitShader {
    #[inline]
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl ShaderTrait for ClosestHitShader {
    #[inline]
    fn shader_type(&self) -> ShaderType {
        ShaderType::RayTracingKHR(ShaderTypeRayTracingKHR::ClosestHit)
    }

    #[inline]
    fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.module)
    }
}

impl PrivateShaderTrait for ClosestHitShader {
    #[inline]
    fn ash_handle(&self) -> ash::vk::ShaderModule {
        self.module
    }
}

impl ClosestHitShader {
    pub fn new<'a, 'b>(device: Arc<Device>, code: &[u32]) -> VulkanResult<Arc<Self>> {
        let create_info = ash::vk::ShaderModuleCreateInfo::default()
            .code(code);

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
                Some(format!("Error creating the closesthit shader: {}", err)),
            )),
        }
    }
}
