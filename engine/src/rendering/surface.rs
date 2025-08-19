use vulkan_framework::{
    image::{CommonImageFormat, ImageFormat},
    swapchain::{DeviceSurfaceInfo, PresentModeSwapchainKHR, SurfaceColorspaceSwapchainKHR},
};

use crate::rendering::RenderingResult;

pub struct SurfaceHelper {
    info: DeviceSurfaceInfo,

    images_count: u32,
    final_format: ImageFormat,
    color_space: SurfaceColorspaceSwapchainKHR,
}

impl SurfaceHelper {
    #[inline]
    pub fn device_swapchain_info(&self) -> &DeviceSurfaceInfo {
        &self.info
    }

    #[inline]
    pub fn final_format(&self) -> ImageFormat {
        self.final_format.to_owned()
    }

    #[inline]
    pub fn color_space(&self) -> SurfaceColorspaceSwapchainKHR {
        self.color_space.to_owned()
    }

    #[inline]
    pub fn images_count(&self) -> u32 {
        self.images_count.to_owned()
    }

    /**
     * Return (frames_in_flight, swapchain_images) so that at least
     * two swapchains can exists at any point in time.
     */
    pub fn frames_in_flight(
        preferred_frames_in_flight: u32,
        device_swapchain_info: &DeviceSurfaceInfo,
    ) -> Option<(u32, u32)> {
        let mut frames_in_flight = preferred_frames_in_flight;
        let mut swapchain_images_count = preferred_frames_in_flight + 1;
        let max_images = device_swapchain_info.max_image_count();

        while frames_in_flight >= 1 {
            match max_images == 0 {
                true => {
                    if swapchain_images_count >= frames_in_flight {
                        break;
                    }
                }
                false => {
                    if max_images >= swapchain_images_count {
                        break;
                    }
                }
            }

            // decrease the amount of needed resources
            swapchain_images_count -= 1;
            frames_in_flight -= 1;
        }

        if frames_in_flight < 1 {
            return None;
        }

        if !device_swapchain_info.image_count_supported(swapchain_images_count) {
            println!(
                "Image count {} not supported (max: {}, min: {})",
                swapchain_images_count,
                device_swapchain_info.max_image_count(),
                device_swapchain_info.min_image_count()
            );
            swapchain_images_count = device_swapchain_info.min_image_count();
            frames_in_flight = swapchain_images_count;
        }

        Some((frames_in_flight, swapchain_images_count))
    }

    pub fn best_format(
        device_swapchain_info: &DeviceSurfaceInfo,
    ) -> (ImageFormat, SurfaceColorspaceSwapchainKHR) {
        if !device_swapchain_info.present_mode_supported(&PresentModeSwapchainKHR::FIFO) {
            panic!("Device does not support the most common present mode. LOL.");
        }

        let final_format = ImageFormat::from(CommonImageFormat::b8g8r8a8_srgb);
        let color_space = SurfaceColorspaceSwapchainKHR::SRGBNonlinear;

        if !device_swapchain_info.format_supported(&color_space, &final_format) {
            panic!("Device does not support the most common format. LOL.");
        }

        (final_format, color_space)
    }

    pub fn new(images_count: u32, info: DeviceSurfaceInfo) -> RenderingResult<Self> {
        let (final_format, color_space) = Self::best_format(&info);

        Ok(Self {
            info,
            images_count,
            final_format,
            color_space,
        })
    }
}
