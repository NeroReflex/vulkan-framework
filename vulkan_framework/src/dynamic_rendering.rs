use std::sync::Arc;

use crate::{
    command_buffer::ClearValues,
    image::{ImageFormat, ImageLayout},
    image_view::ImageView,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AttachmentLoadOp {
    Load,
    Clear,
    DontCare,
}

impl Into<crate::ash::vk::AttachmentLoadOp> for &AttachmentLoadOp {
    fn into(self) -> ash::vk::AttachmentLoadOp {
        match self {
            AttachmentLoadOp::Load => ash::vk::AttachmentLoadOp::LOAD,
            AttachmentLoadOp::DontCare => ash::vk::AttachmentLoadOp::DONT_CARE,
            AttachmentLoadOp::Clear => ash::vk::AttachmentLoadOp::CLEAR,
        }
    }
}

impl Into<crate::ash::vk::AttachmentLoadOp> for AttachmentLoadOp {
    fn into(self) -> ash::vk::AttachmentLoadOp {
        (&self).into()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AttachmentStoreOp {
    Store,
    DontCare,
}

impl Into<crate::ash::vk::AttachmentStoreOp> for &AttachmentStoreOp {
    fn into(self) -> crate::ash::vk::AttachmentStoreOp {
        match self {
            AttachmentStoreOp::Store => ash::vk::AttachmentStoreOp::STORE,
            AttachmentStoreOp::DontCare => ash::vk::AttachmentStoreOp::DONT_CARE,
        }
    }
}

impl Into<crate::ash::vk::AttachmentStoreOp> for AttachmentStoreOp {
    fn into(self) -> crate::ash::vk::AttachmentStoreOp {
        (&self).into()
    }
}

#[derive(Default, Clone, PartialEq, Eq)]
pub struct DynamicRendering {
    pub(crate) color_attachments: smallvec::SmallVec<[ImageFormat; 8]>,
    pub(crate) depth_attachment: Option<ImageFormat>,
    pub(crate) stencil_attachment: Option<ImageFormat>,
}

impl DynamicRendering {
    pub fn new(
        color_attachments: &[ImageFormat],
        depth_attachment: Option<ImageFormat>,
        stencil_attachment: Option<ImageFormat>,
    ) -> Self {
        let color_attachments = color_attachments.iter().cloned().collect();
        Self {
            color_attachments,
            depth_attachment,
            stencil_attachment,
        }
    }
}

#[derive(Clone)]
pub struct DynamicRenderingAttachment(
    Arc<ImageView>,
    ImageLayout,
    ClearValues,
    AttachmentLoadOp,
    AttachmentStoreOp,
);

impl DynamicRenderingAttachment {
    #[inline]
    pub fn image_view(&self) -> Arc<ImageView> {
        self.0.clone()
    }

    pub fn image_layout(&self) -> &ImageLayout {
        &self.1
    }

    #[inline]
    pub fn clear_value(&self) -> &ClearValues {
        &self.2
    }

    #[inline]
    pub fn load_op(&self) -> &AttachmentLoadOp {
        &self.3
    }

    #[inline]
    pub fn store_op(&self) -> &AttachmentStoreOp {
        &self.4
    }

    pub fn new(
        image_view: Arc<ImageView>,
        image_layout: ImageLayout,
        clear_value: ClearValues,
        load_op: AttachmentLoadOp,
        store_op: AttachmentStoreOp,
    ) -> Self {
        Self(image_view, image_layout, clear_value, load_op, store_op)
    }
}
