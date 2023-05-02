use std::sync::Arc;

use ash::vk::Handle;

use crate::{
    buffer::BufferTrait,
    descriptor_pool::{DescriptorPool, DescriptorPoolOwned},
    descriptor_set_layout::{DescriptorSetLayout, DescriptorSetLayoutDependant},
    device::DeviceOwned,
    memory_allocator::MemoryAllocator,
    prelude::{VulkanError, VulkanResult},
    resource_tracking::ResourcesInUseByGPU, image::{ImageTrait, ImageLayout}, image_view::ImageView,
};

pub struct DescriptorSetWriter<'a> {
    //device: Arc<Device>,
    descriptor_set: &'a DescriptorSet,
    writer: Vec<ash::vk::WriteDescriptorSet>, // TODO: use a smallvec here
    used_resources: ResourcesInUseByGPU,
}

impl<'a> DescriptorSetWriter<'a> {
    pub(crate) fn new(descriptor_set: &'a DescriptorSet) -> Self {
        Self {
            /*device: descriptor_set
            .get_parent_descriptor_pool()
            .get_parent_device()
            .clone(),*/
            descriptor_set,
            writer: vec![],
            //binder: ash::vk::WriteDescriptorSet::builder(),
            used_resources: ResourcesInUseByGPU::create(),
        }
    }

    pub fn bind_uniform_buffer<T>(
        &mut self,
        first_layout_id: u32,
        buffers: &[Arc<dyn BufferTrait>],
        offset: Option<u64>,
        size: Option<u64>,
    ) where
        T: Send + Sync + MemoryAllocator,
    {
        let descriptors: Vec<ash::vk::DescriptorBufferInfo> = buffers.iter().map(|buffer| {
            // TODO: assert usage has VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT bit set
            
            self.used_resources.register_buffer_usage(buffer.clone());
            
            ash::vk::DescriptorBufferInfo::builder()
                .range(match size {
                    Option::Some(sz) => sz,
                    Option::None => buffer.size(),
                })
                .buffer(ash::vk::Buffer::from_raw(buffer.native_handle()))
                .offset(offset.unwrap_or(0))
                .build()
            }
        ).collect();

        let descriptor_writes = ash::vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set.ash_handle())
            .dst_binding(first_layout_id)
            .descriptor_type(ash::vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(descriptors.as_slice())
            .build();

        self.writer.push(descriptor_writes);
    }

    pub fn bind_storage_buffers<T>(
        &mut self,
        first_layout_id: u32,
        buffers: &[Arc<dyn BufferTrait>],
        offset: Option<u64>,
        size: Option<u64>,
    ) where
        T: Send + Sync + MemoryAllocator,
    {
        let descriptors: Vec<ash::vk::DescriptorBufferInfo> = buffers.iter().map(|buffer| {
            // TODO: assert usage has VK_BUFFER_USAGE_STORAGE_BUFFER_BIT bit set

            self.used_resources.register_buffer_usage(buffer.clone());
            
            ash::vk::DescriptorBufferInfo::builder()
                .range(match size {
                    Option::Some(sz) => sz,
                    Option::None => buffer.size(),
                })
                .buffer(ash::vk::Buffer::from_raw(buffer.native_handle()))
                .offset(offset.unwrap_or(0))
                .build()
            }
        ).collect();

        let descriptor_writes = ash::vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set.ash_handle())
            .dst_binding(first_layout_id)
            .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(descriptors.as_slice())
            .build();

        self.writer.push(descriptor_writes);
    }

    pub fn bind_storage_images(
        &mut self,
        first_layout_id: u32,
        images: &[(ImageLayout, Arc<ImageView>)],
    )
    {
        let descriptors: Vec<ash::vk::DescriptorImageInfo> = images.iter().map(|(layout, image)| {
            // TODO: assert usage has VK_BUFFER_USAGE_STORAGE_BUFFER_BIT bit set

            self.used_resources.register_image_view_usage(image.clone());
            
            ash::vk::DescriptorImageInfo::builder()
                .image_layout(layout.ash_layout())
                .image_view(image.ash_handle())
                .build()
            }
        ).collect();

        let descriptor_writes = ash::vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set.ash_handle())
            .dst_binding(first_layout_id)
            .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
            .image_info(descriptors.as_slice())
            .build();

        self.writer.push(descriptor_writes);
    }
}

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
            device
                .ash_handle()
                .free_descriptor_sets(self.pool.ash_handle(), [self.descriptor_set].as_slice())
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

    pub fn bind_resources<F>(&self, f: F) -> ResourcesInUseByGPU
    where
        F: Fn(&mut DescriptorSetWriter),
    {
        let mut writer = DescriptorSetWriter::new(self);

        f(&mut writer);

        unsafe {
            self.get_parent_descriptor_pool()
                .get_parent_device()
                .ash_handle()
                .update_descriptor_sets(writer.writer.as_slice(), &[])
        };

        writer.used_resources
    }

    pub fn new(
        pool: Arc<DescriptorPool>,
        layout: Arc<DescriptorSetLayout>,
    ) -> VulkanResult<Arc<Self>> {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                layout.get_parent_device().native_handle(),
                pool.get_parent_device().native_handle()
            );
        }

        let create_info = ash::vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool.ash_handle())
            .set_layouts([layout.ash_handle()].as_slice())
            .build();

        match unsafe {
            pool.get_parent_device()
                .ash_handle()
                .allocate_descriptor_sets(&create_info)
        } {
            Ok(descriptor_set) => Ok(Arc::new(Self {
                pool,
                descriptor_set: descriptor_set[0],
                layout,
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
