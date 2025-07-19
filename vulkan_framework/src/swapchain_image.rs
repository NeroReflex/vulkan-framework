use std::{
    borrow::Borrow,
    ops::Deref,
    sync::{Arc, Mutex},
};

use crate::{
    device::{Device, DeviceOwned},
    image::{Image2DDimensions, ImageDimensions, ImageFlags, ImageFormat, ImageTrait, ImageUsage},
};

pub struct ImageSwapchainKHR {
    device: Arc<Device>,
    flags: ImageFlags,
    usage: ImageUsage,
    format: ImageFormat,
    dimensions: ImageDimensions,
    layers_count: u32,
    mip_levels_count: u32,
    image: ash::vk::Image,
}

impl DeviceOwned for ImageSwapchainKHR {
    fn get_parent_device(&self) -> std::sync::Arc<Device> {
        self.device.clone()
    }
}

impl ImageTrait for ImageSwapchainKHR {
    #[inline]
    fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.image)
    }

    #[inline]
    fn flags(&self) -> ImageFlags {
        self.flags.to_owned()
    }

    #[inline]
    fn usage(&self) -> ImageUsage {
        self.usage.to_owned()
    }

    #[inline]
    fn format(&self) -> ImageFormat {
        self.format.to_owned()
    }

    #[inline]
    fn dimensions(&self) -> ImageDimensions {
        self.dimensions.to_owned()
    }

    #[inline]
    fn layers_count(&self) -> u32 {
        self.layers_count.to_owned()
    }

    #[inline]
    fn mip_levels_count(&self) -> u32 {
        self.mip_levels_count.to_owned()
    }
}

impl ImageSwapchainKHR {
    pub(crate) fn new(
        device: Arc<Device>,
        flags: ImageFlags,
        usage: ImageUsage,
        format: ImageFormat,
        extent: Image2DDimensions,
        layers_count: u32,
        mip_levels_count: u32,
        image: ash::vk::Image,
    ) -> Self {
        Self {
            device,
            image,
            flags,
            usage,
            format,
            dimensions: ImageDimensions::Image2D { extent },
            layers_count: layers_count,
            mip_levels_count,
        }
    }
}
