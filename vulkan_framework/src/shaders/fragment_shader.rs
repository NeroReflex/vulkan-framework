use std::sync::Arc;

use crate::device::{Device, DeviceOwned};
use crate::instance::InstanceOwned;
use crate::prelude::{VulkanError, VulkanResult};
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
    pub fn new(
        device: Arc<Device>,
        push_constant_ranges: &[Arc<PushConstanRange>],
        descriptor_bindings: &[Arc<BindingDescriptor>],
        code: &[u32],
    ) -> VulkanResult<Arc<Self>> {
        for push_constant_range in push_constant_ranges.iter() {
            assert!(push_constant_range
                .shader_access()
                .is_accessible_by(&ShaderType::Fragment));
        }

        for descriptor_set_layout_binding in descriptor_bindings.iter() {
            assert!(descriptor_set_layout_binding
                .shader_access()
                .is_accessible_by(&ShaderType::Fragment));
        }

        let create_info = ash::vk::ShaderModuleCreateInfo::default().code(code);

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
                Some(format!("Error creating the fragment shader: {}", err)),
            )),
        }
    }
}
