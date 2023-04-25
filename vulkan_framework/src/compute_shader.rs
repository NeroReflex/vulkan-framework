use std::sync::Arc;

use crate::device::{Device, DeviceOwned};
use crate::instance::InstanceOwned;
use crate::prelude::{VulkanError, VulkanResult};
use crate::push_constant_range::PushConstanRange;
use crate::shader_layout_binding::{BindingDescriptor, BindingDescriptorDependant};
use crate::shader_trait::{PrivateShaderTrait, ShaderTrait, ShaderType};

pub struct ComputeShader {
    device: Arc<Device>,
    push_constant_ranges: Vec<Arc<PushConstanRange>>,
    descriptor_bindings: Vec<Arc<BindingDescriptor>>,
    module: ash::vk::ShaderModule,
}

impl Drop for ComputeShader {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_shader_module(
                self.module,
                self.device.get_parent_instance().get_alloc_callbacks(),
            )
        }
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
        ash::vk::Handle::as_raw(self.module)
    }
}

impl PrivateShaderTrait for ComputeShader {
    fn ash_handle(&self) -> ash::vk::ShaderModule {
        self.module
    }
}

impl ComputeShader {
    pub fn new<'a, 'b, 'c>(
        device: Arc<Device>,
        push_constant_ranges: &'a [Arc<PushConstanRange>],
        descriptor_bindings: &'b [Arc<BindingDescriptor>],
        code: &'c [u32],
    ) -> VulkanResult<Arc<Self>> {
        for push_constant_range in push_constant_ranges.iter() {
            assert_eq!(
                push_constant_range
                    .shader_access()
                    .is_accessible_by(&ShaderType::Compute),
                true
            );
        }

        for descriptor_set_layout_binding in descriptor_bindings.iter() {
            assert_eq!(
                descriptor_set_layout_binding
                    .shader_access()
                    .is_accessible_by(&ShaderType::Compute),
                true
            );
        }

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
                push_constant_ranges: push_constant_ranges.to_vec(),
                descriptor_bindings: descriptor_bindings.to_vec(),
                module,
            })),
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error creating the descriptor set: {}", err)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }
}
