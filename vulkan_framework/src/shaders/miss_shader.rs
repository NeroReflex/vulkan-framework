use std::sync::Arc;

use crate::device::{Device, DeviceOwned};
use crate::instance::InstanceOwned;
use crate::prelude::VulkanResult;

use crate::shader_trait::{PrivateShaderTrait, ShaderTrait, ShaderType, ShaderTypeRayTracingKHR};

pub struct MissShader {
    device: Arc<Device>,
    module: ash::vk::ShaderModule,
}

impl Drop for MissShader {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_shader_module(
                self.module,
                self.device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

impl DeviceOwned for MissShader {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl ShaderTrait for MissShader {
    #[inline]
    fn shader_type(&self) -> ShaderType {
        ShaderType::RayTracingKHR(ShaderTypeRayTracingKHR::Miss)
    }

    #[inline]
    fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.module)
    }
}

impl PrivateShaderTrait for MissShader {
    #[inline]
    fn ash_handle(&self) -> ash::vk::ShaderModule {
        self.module
    }
}

impl MissShader {
    pub fn new(device: Arc<Device>, code: &[u32]) -> VulkanResult<Arc<Self>> {
        let create_info = ash::vk::ShaderModuleCreateInfo::default().code(code);

        let module = unsafe {
            device.ash_handle().create_shader_module(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        }?;

        Ok(Arc::new(Self {
            device,
            //push_constant_ranges: push_constant_ranges.iter().map(|cr| cr.clone()).collect(),
            //descriptor_bindings: descriptor_bindings.to_vec(),
            module,
        }))
    }
}
