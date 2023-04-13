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

pub enum ImageViewType {
    Image1D,
    Image2D,
    Image3D,
    CubeMap,
    Image1DArray,
    Image2DArray,
    CubeMapArray,
}

#[allow(non_camel_case_types)]
pub enum ImageViewColorMapping {
    rgba_rgba,
    bgra_rgba
}

pub struct ImageView {
    image: Arc<dyn ImageTrait>,
    image_view: ash::vk::ImageView,
    view_type: ImageViewType,
    color_mapping: ImageViewColorMapping,
    subrange_base_mip_level: u32,
    subrange_level_count: u32,
    subrange_base_array_layer: u32,
    subrange_layer_count: u32
}

impl ImageOwned for ImageView {
    fn get_parent_image(&self) -> Arc<dyn ImageTrait> {
        self.image.clone()
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        let device = self.get_parent_image().get_parent_device();

        unsafe {
            device.ash_handle().destroy_image_view(self.image_view.clone(), device.get_parent_instance().get_alloc_callbacks())
        }
    }
}

impl ImageView {
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.image_view.clone())
    }

    pub fn new(
        image: Arc<dyn ImageTrait>,
        format: Option<ImageFormat>,
        view_type: ImageViewType,
        maybe_specified_color_mapping: Option<ImageViewColorMapping>,
        maybe_specified_subrange_base_mip_level: Option<u32>,
        maybe_specified_subrange_level_count: Option<u32>,
        maybe_specified_subrange_base_array_layer: Option<u32>,
        maybe_specified_subrange_layer_count: Option<u32>
    ) -> VulkanResult<Arc<Self>> {
        

        // by default do not swizzle colors
        let color_mapping = maybe_specified_color_mapping.unwrap_or(ImageViewColorMapping::rgba_rgba);
        let subrange_base_mip_level = maybe_specified_subrange_base_mip_level.unwrap_or(0);
        let subrange_level_count = maybe_specified_subrange_level_count.unwrap_or(image.mip_levels_count());
        let subrange_base_array_layer = maybe_specified_subrange_base_array_layer.unwrap_or(0);
        let subrange_layer_count = maybe_specified_subrange_layer_count.unwrap_or(1);

        let device = image.get_parent_device();

        let srr = ash::vk::ImageSubresourceRange {

        };

        let create_info = ash::vk::ImageViewCreateInfo::builder()
            .subresource_range(srr)
            .build();

        match unsafe { device.ash_handle().create_image_view(&create_info, device.get_parent_instance().get_alloc_callbacks()) } {
            Ok(image_view) => {
                Ok(Arc::new(Self {
                    image,
                    image_view,
                    view_type,
                    color_mapping,
                    subrange_base_mip_level,
                    subrange_level_count,
                    subrange_base_array_layer,
                    subrange_layer_count
                }))
            },
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    println!("Error creating the specified image view: {}", err);
                    assert_eq!(true, false)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }
}
