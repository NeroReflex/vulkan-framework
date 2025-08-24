use std::sync::Arc;

use crate::device::{Device, DeviceOwned};
use crate::instance::InstanceOwned;
use crate::prelude::VulkanResult;
use crate::push_constant_range::PushConstanRange;
use crate::shader_layout_binding::BindingDescriptor;
use crate::shader_trait::{PrivateShaderTrait, ShaderTrait, ShaderType};

pub struct FragmentShader {
    device: Arc<Device>,
    //push_constant_ranges: smallvec::SmallVec<[Arc<PushConstanRange>; 4]>,
    //descriptor_bindings: Vec<Arc<BindingDescriptor>>,
    module: ash::vk::ShaderModule,
}

impl Drop for FragmentShader {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_shader_module(
                self.module,
                self.device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

impl DeviceOwned for FragmentShader {
    #[inline]
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl ShaderTrait for FragmentShader {
    #[inline]
    fn shader_type(&self) -> ShaderType {
        ShaderType::Fragment
    }

    #[inline]
    fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.module)
    }
}

impl PrivateShaderTrait for FragmentShader {
    #[inline]
    fn ash_handle(&self) -> ash::vk::ShaderModule {
        self.module
    }
}

impl FragmentShader {
    pub fn new(device: Arc<Device>, code: &[u32]) -> VulkanResult<Arc<Self>> {
        let create_info = ash::vk::ShaderModuleCreateInfo::default().code(code);

        let module = unsafe {
            device.ash_handle().create_shader_module(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        }?;

        Ok(Arc::new(Self { device, module }))
    }
}
