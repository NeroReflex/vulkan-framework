use std::sync::Arc;

use crate::descriptor_set_layout::{DescriptorSetLayout, DescriptorSetLayoutDependant};
use crate::device::{Device, DeviceOwned};
use crate::instance::InstanceOwned;
use crate::prelude::{VulkanResult, VulkanError};
use crate::shader_trait::{PrivateShaderTrait, ShaderTrait, ShaderType};
use crate::shader_layout_binding::BindingDescriptor;

pub struct ComputeShader {
    descriptor_set: Arc<DescriptorSetLayout>,
    module: ash::vk::ShaderModule,
}

impl DescriptorSetLayoutDependant for ComputeShader {
    fn get_parent_descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.descriptor_set.clone()
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
    pub fn new<'a>(
        descriptor_set: Arc<DescriptorSetLayout>,
        code: &'a [u32],
    ) -> VulkanResult<Arc<Self>> {

        // TODO: implement push constant(s)!
        todo!();


        // Inoltre.... uno shader dovrebbe sapere solo di quali binding ha bisogno, quindi una collezione di BindingDescriptor
        // sarà poi quando si mette insieme la pipeline che tali informazioni confluiranno su un DescriptorSetLayout (che rappresenta un VkDescriptorSetLayout)

        for descriptor_set_layout_binding in descriptor_set.descriptors().iter() {
            assert_eq!(descriptor_set_layout_binding.shader_access().is_accessible_by(&ShaderType::Compute), true);
        };

        let create_info = ash::vk::ShaderModuleCreateInfo::builder()
            .code(code)
            .build();

        let device = descriptor_set.get_parent_device();

        match unsafe { device.ash_handle().create_shader_module(&create_info, device.get_parent_instance().get_alloc_callbacks()) } {
            Ok(module) => {
                Ok(Arc::new(Self {
                    descriptor_set,
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