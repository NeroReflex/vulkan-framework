use std::sync::Arc;

use crate::{descriptor_pool::{DescriptorPool, DescriptorPoolOwned}, device::DeviceOwned, prelude::{VulkanResult, VulkanError}, descriptor_set_layout::{DescriptorSetLayout, DescriptorSetLayoutDependant}};

pub struct DescriptorSet {
    pool: Arc<DescriptorPool>,
    layout: Arc<DescriptorSetLayout>,
    descriptor_set: ash::vk::DescriptorSet,
}

impl DescriptorSetLayoutDependant for DescriptorSet {
    fn get_parent_descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.layout.clone()
    }
}

impl DescriptorPoolOwned for DescriptorSet {
    fn get_parent_descriptor_pool(&self) -> Arc<DescriptorPool> {
        self.pool.clone()
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        let device = self.get_parent_descriptor_pool().get_parent_device();

        let _ = unsafe {
            device.ash_handle().free_descriptor_sets(self.pool.ash_handle(), [self.descriptor_set].as_slice())
        };
    }
}

impl DescriptorSet {
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.descriptor_set)
    }
    
    pub(crate) fn ash_handle(&self) -> ash::vk::DescriptorSet {
        self.descriptor_set
    }

    pub fn new(
        pool: Arc<DescriptorPool>,
        layout: Arc<DescriptorSetLayout>
    ) -> VulkanResult<Arc<Self>> {
        #[cfg(debug_assertions)]
        {
            assert_eq!(layout.get_parent_device().native_handle(), pool.get_parent_device().native_handle());
        }

        let create_info = ash::vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool.ash_handle())
            .set_layouts([layout.ash_handle()].as_slice())
            .build();

            match unsafe { pool.get_parent_device().ash_handle().allocate_descriptor_sets(&create_info) } {
                Ok(descriptor_set) => {
                    Ok(Arc::new(Self {
                        pool,
                        descriptor_set: descriptor_set[0],
                        layout
                    }))
                },
                Err(err) => {
                    #[cfg(debug_assertions)]
                    {
                        println!("Error creating the descriptor set: {}", err);
                        assert_eq!(true, false)
                    }

                    Err(VulkanError::Unspecified)
                }
            }
    }
}