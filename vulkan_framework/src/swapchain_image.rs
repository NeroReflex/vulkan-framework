use crate::{image::ImageTrait, device::DeviceOwned};

pub struct SwapchainKHRImage {

}

impl DeviceOwned for SwapchainKHRImage {
    fn get_parent_device(&self) -> std::sync::Arc<crate::device::Device> {
        todo!()
    }
}

impl ImageTrait for SwapchainKHRImage {
    fn native_handle(&self) -> u64 {
        todo!()
    }

    fn format(&self) -> crate::image::ImageFormat {
        todo!()
    }

    fn dimensions(&self) -> crate::image::ImageDimensions {
        todo!()
    }

    fn layers_count(&self) -> u32 {
        todo!()
    }

    fn mip_levels_count(&self) -> u32 {
        todo!()
    }
}