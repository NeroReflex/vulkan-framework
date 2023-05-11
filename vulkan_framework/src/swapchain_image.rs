use std::sync::Arc;

use crate::{image::ImageTrait, device::DeviceOwned, swapchain::{SwapchainKHROwned, SwapchainKHR}};

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
        crate::image::ImageDimensions::Image2D { extent: self.swapchain.images_extent() }
    }

    fn layers_count(&self) -> u32 {
        self.swapchain.images_layers_count()
    }

    fn mip_levels_count(&self) -> u32 {
        self.swapchain.images_layers_count()
    }
}

impl ImageSwapchainKHR {

}