use std::sync::Arc;

use crate::{
    device::DeviceOwned,
    image::{Image1DTrait, Image2DDimensions, Image2DTrait, ImageFlags, ImageFormat, ImageUsage},
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

        let create_info = ash::vk::FramebufferCreateInfo::default()
            .flags(ash::vk::FramebufferCreateFlags::empty())
            .render_pass(renderpass.ash_handle())
            .attachments(attachments.as_slice())
            .width(dimensions.width())
            .height(dimensions.height())
            .layers(layers);

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

#[derive(Clone, PartialEq, Eq)]
pub struct ImagelessFramebufferAttachmentImageInfo {
    img_usage: ImageUsage,
    img_flags: ImageFlags,

    width: u32,
    height: u32,
    layer_count: u32,
    view_formats: smallvec::SmallVec<[ImageFormat; 2]>,
}

impl ImagelessFramebufferAttachmentImageInfo {
    pub fn new(
        img_usage: ImageUsage,
        img_flags: ImageFlags,
        width: u32,
        height: u32,
        layer_count: u32,
        view_formats: &[ImageFormat],
    ) -> Self {
        Self {
            img_usage,
            img_flags,
            width,
            height,
            layer_count,
            view_formats: view_formats
                .iter()
                .copied()
                .collect::<smallvec::SmallVec<[ImageFormat; 2]>>(),
        }
    }

    #[inline]
    pub fn image_format(&self, index: usize) -> ImageFormat {
        self.view_formats[index]
    }

    #[inline]
    pub fn ash_view_formats(&self) -> smallvec::SmallVec<[ash::vk::Format; 2]> {
        self.view_formats.iter().map(|f| f.ash_format()).collect()
    }

    #[inline]
    pub(crate) fn ash_flags(&self) -> ash::vk::ImageCreateFlags {
        self.img_flags.ash_flags()
    }

    #[inline]
    pub(crate) fn ash_usage(&self) -> ash::vk::ImageUsageFlags {
        self.img_usage.into()
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    pub fn layer_count(&self) -> u32 {
        self.layer_count
    }
}

pub struct ImagelessFramebuffer {
    renderpass: Arc<RenderPass>,
    framebuffer: ash::vk::Framebuffer,
    dimensions: Image2DDimensions,
    attachments_descriptors: smallvec::SmallVec<[ImagelessFramebufferAttachmentImageInfo; 8]>,
    layers: u32,
}

impl RenderPassCompatible for ImagelessFramebuffer {
    fn get_parent_renderpass(&self) -> Arc<RenderPass> {
        self.renderpass.clone()
    }
}

impl Drop for ImagelessFramebuffer {
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

impl ImagelessFramebuffer {
    pub fn new(
        renderpass: Arc<RenderPass>,
        dimensions: Image2DDimensions,
        attachments: &[ImagelessFramebufferAttachmentImageInfo],
        layers: u32,
    ) -> VulkanResult<Arc<Self>> {
        let attachment_formats = attachments
            .iter()
            .map(|ia| ia.ash_view_formats())
            .collect::<smallvec::SmallVec<[smallvec::SmallVec<[ash::vk::Format; 2]>; 8]>>(
        );

        let attachment_image_infos = attachments
            .iter()
            .enumerate()
            .map(|(idx, at)| {
                ash::vk::FramebufferAttachmentImageInfo::default()
                    .flags(at.ash_flags())
                    .usage(at.ash_usage())
                    .width(at.width())
                    .height(at.height())
                    .layer_count(at.layer_count())
                    .view_formats(attachment_formats[idx].as_slice())
            })
            .collect::<Vec<ash::vk::FramebufferAttachmentImageInfoKHR>>/* ::<smallvec::SmallVec<[ash::vk::FramebufferAttachmentImageInfoKHR; 8]>>*/();

        let mut attachments_create_info = ash::vk::FramebufferAttachmentsCreateInfo::default()
            .attachment_image_infos(attachment_image_infos.as_slice());

        let create_info = ash::vk::FramebufferCreateInfo::default()
            .push_next(&mut attachments_create_info)
            .flags(ash::vk::FramebufferCreateFlags::IMAGELESS_KHR)
            .render_pass(renderpass.ash_handle())
            .width(dimensions.width())
            .height(dimensions.height())
            .layers(layers)
            .attachment_count(attachment_image_infos.len() as u32);

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
                dimensions,
                layers,
                attachments_descriptors: attachments
                    .iter()
                    .cloned()
                    .collect::<smallvec::SmallVec<[ImagelessFramebufferAttachmentImageInfo; 8]>>(),
            })),
            Err(_err) => {
                todo!()
            }
        }
    }
}

impl FramebufferTrait for ImagelessFramebuffer {
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
