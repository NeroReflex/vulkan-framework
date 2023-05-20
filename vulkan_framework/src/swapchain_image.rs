use std::sync::Arc;

use crate::{
    device::DeviceOwned,
    image::ImageTrait,
    prelude::{VulkanError, VulkanResult},
    swapchain::{SwapchainKHR, SwapchainKHROwned},
};

pub struct ImageSwapchainKHR {
    swapchain: Arc<SwapchainKHR>,
    image: ash::vk::Image,
}

impl DeviceOwned for ImageSwapchainKHR {
    fn get_parent_device(&self) -> std::sync::Arc<crate::device::Device> {
        self.swapchain.get_parent_device()
    }
}

impl SwapchainKHROwned for ImageSwapchainKHR {
    fn get_parent_swapchain(&self) -> std::sync::Arc<crate::swapchain::SwapchainKHR> {
        self.swapchain.clone()
    }
}

impl ImageTrait for ImageSwapchainKHR {
    fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.image)
    }

    fn format(&self) -> crate::image::ImageFormat {
        self.swapchain.images_format()
    }

    fn dimensions(&self) -> crate::image::ImageDimensions {
        crate::image::ImageDimensions::Image2D {
            extent: self.swapchain.images_extent(),
        }
    }

    fn layers_count(&self) -> u32 {
        self.swapchain.images_layers_count()
    }

    fn mip_levels_count(&self) -> u32 {
        self.swapchain.images_layers_count()
    }
}

impl ImageSwapchainKHR {
    pub fn extract(
        swapchain: &Arc<SwapchainKHR>,
    ) -> VulkanResult<smallvec::SmallVec<[Arc<Self>; 8]>> {
        match swapchain.get_parent_device().ash_ext_swapchain_khr() {
            Option::Some(ext) => {
                match unsafe { ext.get_swapchain_images(swapchain.ash_handle()) } {
                    Ok(images) => Ok(images
                        .into_iter()
                        .map(|swapchain_image| {
                            Arc::new(Self {
                                swapchain: swapchain.clone(),
                                image: swapchain_image,
                            })
                        })
                        .collect::<smallvec::SmallVec<[Arc<Self>; 8]>>()),
                    Err(_err) => return Err(VulkanError::Unspecified),
                }
            }
            Option::None => {
                return Err(VulkanError::MissingExtension(String::from(
                    "VK_KHR_swapchain",
                )))
            }
        }
    }
}
