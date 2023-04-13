use ash::vk::{Extent3D, ImageLayout, ImageType, ImageUsageFlags, SampleCountFlags, SharingMode};

use crate::{
    device::{Device, DeviceOwned},
    image::*,
    instance::{InstanceAPIVersion, InstanceOwned},
    memory_allocator::{AllocationResult, MemoryAllocator},
    memory_heap::MemoryHeapOwned,
    memory_pool::{MemoryPool, MemoryPoolBacked},
    prelude::{VulkanError, VulkanResult},
    queue_family::QueueFamily,
};

use std::sync::Arc;

pub struct ImageView {
    image: Arc<dyn ImageTrait>,
    image_view: ash::vk::ImageView,
}

impl ImageOwned for ImageView {
    fn get_parent_image(&self) -> Arc<dyn ImageTrait> {
        self.image.clone()
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {}
}

impl ImageView {
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.image_view.clone())
    }

    pub fn new(image: Arc<dyn ImageTrait>) -> VulkanResult<Arc<Self>> {
        todo!()
    }
}
