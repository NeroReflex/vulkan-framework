use std::sync::Arc;

use ash::vk::Handle;

use crate::{
    clear_values::{ColorClearValues, DepthClearValues, StencilClearValues},
    image::ImageFormat,
    image_view::ImageView,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AttachmentLoadOp {
    Load,
    Clear,
    DontCare,
}

impl From<&AttachmentLoadOp> for crate::ash::vk::AttachmentLoadOp {
    fn from(val: &AttachmentLoadOp) -> Self {
        match val {
            AttachmentLoadOp::Load => ash::vk::AttachmentLoadOp::LOAD,
            AttachmentLoadOp::DontCare => ash::vk::AttachmentLoadOp::DONT_CARE,
            AttachmentLoadOp::Clear => ash::vk::AttachmentLoadOp::CLEAR,
        }
    }
}

impl From<AttachmentLoadOp> for crate::ash::vk::AttachmentLoadOp {
    fn from(val: AttachmentLoadOp) -> Self {
        (&val).into()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AttachmentStoreOp {
    Store,
    DontCare,
}

impl From<&AttachmentStoreOp> for crate::ash::vk::AttachmentStoreOp {
    fn from(val: &AttachmentStoreOp) -> Self {
        match val {
            AttachmentStoreOp::Store => ash::vk::AttachmentStoreOp::STORE,
            AttachmentStoreOp::DontCare => ash::vk::AttachmentStoreOp::DONT_CARE,
        }
    }
}

impl From<AttachmentStoreOp> for crate::ash::vk::AttachmentStoreOp {
    fn from(val: AttachmentStoreOp) -> Self {
        (&val).into()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct DynamicRenderingColorDefinition {
    format: ImageFormat,
}

impl DynamicRenderingColorDefinition {
    pub fn new(format: ImageFormat) -> Self {
        Self { format }
    }

    pub fn format(&self) -> ImageFormat {
        self.format
    }
}

impl Into<crate::ash::vk::Format> for &DynamicRenderingColorDefinition {
    fn into(self) -> crate::ash::vk::Format {
        self.format.into()
    }
}

impl Into<crate::ash::vk::PipelineColorBlendAttachmentState> for &DynamicRenderingColorDefinition {
    fn into(self) -> crate::ash::vk::PipelineColorBlendAttachmentState {
        ash::vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(ash::vk::ColorComponentFlags::RGBA)
            .blend_enable(false)
            .src_color_blend_factor(ash::vk::BlendFactor::ONE)
            .dst_color_blend_factor(ash::vk::BlendFactor::ZERO)
            .color_blend_op(ash::vk::BlendOp::ADD)
            .src_alpha_blend_factor(ash::vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(ash::vk::BlendFactor::ONE)
            .alpha_blend_op(ash::vk::BlendOp::ADD)
    }
}

#[derive(Default, Clone, PartialEq, Eq)]
pub struct DynamicRendering {
    pub(crate) color_attachments: smallvec::SmallVec<[DynamicRenderingColorDefinition; 8]>,
    pub(crate) depth_attachment: Option<ImageFormat>,
    pub(crate) stencil_attachment: Option<ImageFormat>,
}

impl DynamicRendering {
    pub fn new(
        color_attachments: &[DynamicRenderingColorDefinition],
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

/// When beginning a rendering operation an attachment can either have:
///   - its initial state set to the previous state of the attachment (recycle previous data)
///   - its initial state set to being cleared with either a specific color or any color (as in you don't care)
#[derive(Debug, Clone)]
pub enum RenderingAttachmentSetup<T>
where
    T: Into<crate::ash::vk::ClearValue>,
{
    Load,
    Clear(Option<T>),
}

impl<T> RenderingAttachmentSetup<T>
where
    T: Into<crate::ash::vk::ClearValue>,
{
    pub fn load() -> Self {
        Self::Load
    }

    pub fn clear(value: T) -> Self {
        Self::Clear(Some(value))
    }

    pub fn dont_care() -> Self {
        Self::Clear(None)
    }
}

impl<T> Into<crate::ash::vk::AttachmentLoadOp> for RenderingAttachmentSetup<T>
where
    T: Into<crate::ash::vk::ClearValue>,
{
    fn into(self) -> crate::ash::vk::AttachmentLoadOp {
        match self {
            RenderingAttachmentSetup::Load => crate::ash::vk::AttachmentLoadOp::LOAD,
            RenderingAttachmentSetup::Clear(maybe_clear) => match maybe_clear {
                Some(_) => crate::ash::vk::AttachmentLoadOp::CLEAR,
                None => crate::ash::vk::AttachmentLoadOp::DONT_CARE,
            },
        }
    }
}

impl<T> Into<crate::ash::vk::ClearValue> for RenderingAttachmentSetup<T>
where
    T: Into<crate::ash::vk::ClearValue>,
{
    fn into(self) -> crate::ash::vk::ClearValue {
        match self {
            RenderingAttachmentSetup::Load => crate::ash::vk::ClearValue::default(),
            RenderingAttachmentSetup::Clear(maybe_clear) => match maybe_clear {
                Some(clear_value) => clear_value.into(),
                None => crate::ash::vk::ClearValue::default(),
            },
        }
    }
}

#[derive(Clone)]
pub struct DynamicRenderingColorAttachment(
    Arc<ImageView>,
    RenderingAttachmentSetup<ColorClearValues>,
    AttachmentStoreOp,
);

impl<'a> Into<crate::ash::vk::RenderingAttachmentInfo<'a>> for DynamicRenderingColorAttachment {
    fn into(self) -> crate::ash::vk::RenderingAttachmentInfo<'a> {
        ash::vk::RenderingAttachmentInfo::default()
            .image_layout(crate::ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image_view(ash::vk::ImageView::from_raw(self.0.native_handle()))
            .store_op(self.2.into())
            .load_op(self.1.clone().into())
            .clear_value(self.1.into())
    }
}

impl DynamicRenderingColorAttachment {
    #[inline]
    pub fn image_view(&self) -> Arc<ImageView> {
        self.0.clone()
    }

    #[inline]
    pub fn clear_value(&self) -> &RenderingAttachmentSetup<ColorClearValues> {
        &self.1
    }

    #[inline]
    pub fn store_op(&self) -> &AttachmentStoreOp {
        &self.2
    }

    pub fn new(
        image_view: Arc<ImageView>,
        load_op: RenderingAttachmentSetup<ColorClearValues>,
        store_op: AttachmentStoreOp,
    ) -> Self {
        Self(image_view, load_op, store_op)
    }
}

#[derive(Clone)]
pub struct DynamicRenderingDepthAttachment(
    Arc<ImageView>,
    RenderingAttachmentSetup<DepthClearValues>,
    AttachmentStoreOp,
);

impl<'a> Into<crate::ash::vk::RenderingAttachmentInfo<'a>> for DynamicRenderingDepthAttachment {
    fn into(self) -> crate::ash::vk::RenderingAttachmentInfo<'a> {
        ash::vk::RenderingAttachmentInfo::default()
            .image_layout(crate::ash::vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .image_view(ash::vk::ImageView::from_raw(self.0.native_handle()))
            .store_op(self.2.into())
            .load_op(self.1.clone().into())
            .clear_value(self.1.into())
    }
}

impl DynamicRenderingDepthAttachment {
    #[inline]
    pub fn image_view(&self) -> Arc<ImageView> {
        self.0.clone()
    }

    #[inline]
    pub fn clear_value(&self) -> &RenderingAttachmentSetup<DepthClearValues> {
        &self.1
    }

    #[inline]
    pub fn store_op(&self) -> &AttachmentStoreOp {
        &self.2
    }

    pub fn new(
        image_view: Arc<ImageView>,
        load_op: RenderingAttachmentSetup<DepthClearValues>,
        store_op: AttachmentStoreOp,
    ) -> Self {
        Self(image_view, load_op, store_op)
    }
}

#[derive(Clone)]
pub struct DynamicRenderingStencilAttachment(
    Arc<ImageView>,
    RenderingAttachmentSetup<StencilClearValues>,
    AttachmentStoreOp,
);

impl<'a> Into<crate::ash::vk::RenderingAttachmentInfo<'a>> for DynamicRenderingStencilAttachment {
    fn into(self) -> crate::ash::vk::RenderingAttachmentInfo<'a> {
        ash::vk::RenderingAttachmentInfo::default()
            .image_layout(crate::ash::vk::ImageLayout::STENCIL_ATTACHMENT_OPTIMAL)
            .image_view(ash::vk::ImageView::from_raw(self.0.native_handle()))
            .store_op(self.2.into())
            .load_op(self.1.clone().into())
            .clear_value(self.1.into())
    }
}

impl DynamicRenderingStencilAttachment {
    #[inline]
    pub fn image_view(&self) -> Arc<ImageView> {
        self.0.clone()
    }

    #[inline]
    pub fn clear_value(&self) -> &RenderingAttachmentSetup<StencilClearValues> {
        &self.1
    }

    #[inline]
    pub fn store_op(&self) -> &AttachmentStoreOp {
        &self.2
    }

    pub fn new(
        image_view: Arc<ImageView>,
        load_op: RenderingAttachmentSetup<StencilClearValues>,
        store_op: AttachmentStoreOp,
    ) -> Self {
        Self(image_view, load_op, store_op)
    }
}
