use std::sync::Arc;

use crate::device::{Device, DeviceOwned};
use crate::instance::InstanceOwned;
use crate::prelude::{VulkanResult, VulkanError};
use crate::shader_trait::{PrivateShaderTrait, ShaderTrait, ShaderType};
use crate::shader_layout_binding::{BindingDescriptor, BindingDescriptorDependant};

pub struct ComputeShader {
    device: Arc<Device>,
    descriptor_bindings: Vec<Arc<BindingDescriptor>>,
    module: ash::vk::ShaderModule,
}

impl Drop for ComputeShader {
    fn drop(&mut self) {
        unsafe { self.device.ash_handle().destroy_shader_module(self.module, self.device.get_parent_instance().get_alloc_callbacks()) }
    }
}

impl DeviceOwned for ComputeShader {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl BindingDescriptorDependant for ComputeShader {
    fn get_parent_binding_descriptors(&self) -> Vec<Arc<BindingDescriptor>> {
        self.descriptor_bindings.clone()
    }
}

impl ShaderTrait for ComputeShader {
    fn shader_type(&self) -> ShaderType {
        ShaderType::Compute
    }

    fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.module.clone())
    }
}

impl PrivateShaderTrait for ComputeShader {
    fn ash_handle(&self) -> ash::vk::ShaderModule {
        self.module.clone()
    }
}

impl ComputeShader {
    pub fn new<'a, 'b>(
        device: Arc<Device>,
        descriptor_bindings: &'a [Arc<BindingDescriptor>],
        code: &'b [u32],
    ) -> VulkanResult<Arc<Self>> {

        // TODO: implement push constant(s)!

        for descriptor_set_layout_binding in descriptor_bindings.iter() {
            assert_eq!(descriptor_set_layout_binding.shader_access().is_accessible_by(&ShaderType::Compute), true);
        };

        let create_info = ash::vk::ShaderModuleCreateInfo::builder()
            .code(code)
            .build();

        match unsafe { device.ash_handle().create_shader_module(&create_info, device.get_parent_instance().get_alloc_callbacks()) } {
            Ok(module) => {
                Ok(Arc::new(Self {
                    device,
                    descriptor_bindings: descriptor_bindings.iter().map(|arc| arc.clone()).collect::<Vec<Arc<BindingDescriptor>>>(),
                    module
                }))
            },
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    println!("Error creating the descriptor set: {}", err);
                    assert_eq!(true, false)
                }

                return Err(VulkanError::Unspecified);
            }
        }
    }
}