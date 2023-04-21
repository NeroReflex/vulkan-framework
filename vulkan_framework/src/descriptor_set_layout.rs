use std::sync::Arc;

use crate::{device::{Device, DeviceOwned}, instance::InstanceOwned, prelude::{VulkanResult, VulkanError}, shader_layout_binding::BindingDescriptor};

pub struct DescriptorSetLayout {
    device: Arc<Device>,
    layout: ash::vk::DescriptorSetLayout,
    descriptors: Vec<Arc<BindingDescriptor>>
}

pub trait DescriptorSetLayoutDependant {
    fn get_parent_descriptor_set_layout(&self) -> Arc<DescriptorSetLayout>;
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe { self.get_parent_device().ash_handle().destroy_descriptor_set_layout(self.layout, self.get_parent_device().get_parent_instance().get_alloc_callbacks()) }
    }
}

impl DeviceOwned for DescriptorSetLayout {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl DescriptorSetLayout {
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.layout)
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::DescriptorSetLayout {
        self.layout.clone()
    }

    pub fn descriptors(&self) -> Vec<Arc<BindingDescriptor>> {
        self.descriptors.clone()
    }

    pub fn new(
        device: Arc<Device>,
        descriptors: &[Arc<BindingDescriptor>],
    ) -> VulkanResult<Arc<Self>> {
        // a collection of VkDescriptorSetLayoutBinding
        let bindings: Vec<ash::vk::DescriptorSetLayoutBinding> = descriptors.iter().map(|d| d.ash_handle()).collect();

        let create_info = ash::vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(bindings.as_slice())
            .build();

        match unsafe { device.ash_handle().create_descriptor_set_layout(&create_info, device.get_parent_instance().get_alloc_callbacks()) } {
            Ok(layout) => {
                Ok(
                    Arc::new(
                        Self {
                            device,
                            layout,
                            descriptors: descriptors.iter().map(|d| d.clone()).collect()
                        }
                    )
                )
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