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

    /// Choose frame slots and a minimum image count, reserving an image for
    /// presentation when the surface permits it.
    pub fn frames_in_flight(
        preferred_frames_in_flight: u32,
        device_swapchain_info: &DeviceSurfaceInfo,
    ) -> Option<(u32, u32)> {
        Self::choose_frame_counts(
            preferred_frames_in_flight,
            device_swapchain_info.min_image_count(),
            device_swapchain_info.max_image_count(),
        )
    }

    pub(crate) fn choose_frame_counts(preferred: u32, min_images: u32, max_images: u32) -> Option<(u32, u32)> {
        if preferred == 0 || min_images == 0 || (max_images != 0 && max_images < min_images) {
            return None;
        }
        let mut images = preferred.saturating_add(1).max(min_images);
        if max_images != 0 {
            images = images.min(max_images);
        }
        let frames = preferred.min(images.saturating_sub(1).max(1));
        Some((frames, images))
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
