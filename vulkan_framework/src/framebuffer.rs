use std::sync::Arc;

use crate::{
    device::DeviceOwned,
    image::{Image1DTrait, Image2DDimensions, Image2DTrait},
    image_view::ImageView,
    instance::InstanceOwned,
    prelude::VulkanResult,
    renderpass::{RenderPass, RenderPassCompatible},
};

pub trait FramebufferTrait: RenderPassCompatible {
    fn native_handle(&self) -> u64;

    fn dimensions(&self) -> Image2DDimensions;

    fn layers(&self) -> u32;
}

pub struct Framebuffer {
    renderpass: Arc<RenderPass>,
    framebuffer: ash::vk::Framebuffer,
    imageviews: smallvec::SmallVec<[Arc<ImageView>; 16]>,
    dimensions: Image2DDimensions,
    layers: u32,
}

impl RenderPassCompatible for Framebuffer {
    fn get_parent_renderpass(&self) -> Arc<RenderPass> {
        self.renderpass.clone()
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        let device = self.renderpass.get_parent_device();

        unsafe {
            device.ash_handle().destroy_framebuffer(
                self.framebuffer,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

impl Framebuffer {
    pub fn new(
        renderpass: Arc<RenderPass>,
        imageviews: &[Arc<ImageView>],
        dimensions: Image2DDimensions,
        layers: u32,
    ) -> VulkanResult<Arc<Self>> {
        let attachments = imageviews
            .iter()
            .map(|iv| iv.ash_handle())
            .collect::<smallvec::SmallVec<[ash::vk::ImageView; 16]>>();

        let create_info = ash::vk::FramebufferCreateInfo::builder()
            .flags(ash::vk::FramebufferCreateFlags::empty())
            .render_pass(renderpass.ash_handle())
            .attachments(attachments.as_slice())
            .width(dimensions.width())
            .height(dimensions.height())
            .layers(layers)
            .build();

        match unsafe {
            renderpass
                .get_parent_device()
                .ash_handle()
                .create_framebuffer(
                    &create_info,
                    renderpass
                        .get_parent_device()
                        .get_parent_instance()
                        .get_alloc_callbacks(),
                )
        } {
            Ok(framebuffer) => Ok(Arc::new(Self {
                renderpass,
                framebuffer,
                imageviews: imageviews
                    .iter()
                    .cloned()
                    .collect::<smallvec::SmallVec<[Arc<ImageView>; 16]>>(),
                dimensions,
                layers,
            })),
            Err(_err) => {
                todo!()
            }
        }
    }
}

impl FramebufferTrait for Framebuffer {
    fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.framebuffer)
    }

    fn dimensions(&self) -> Image2DDimensions {
        self.dimensions
    }

    fn layers(&self) -> u32 {
        self.layers
    }
}
