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
    alpha_blend: bool,
}

impl DynamicRenderingColorDefinition {
    pub fn new(format: ImageFormat) -> Self {
        Self {
            format,
            alpha_blend: false,
        }
    }

    pub fn new_with_alpha_blend(format: ImageFormat) -> Self {
        Self {
            format,
            alpha_blend: true,
        }
    }

    pub fn format(&self) -> ImageFormat {
        self.format
    }
}

impl From<&DynamicRenderingColorDefinition> for crate::ash::vk::Format {
    fn from(val: &DynamicRenderingColorDefinition) -> Self {
        val.format.into()
    }
}

impl From<&DynamicRenderingColorDefinition> for crate::ash::vk::PipelineColorBlendAttachmentState {
    fn from(val: &DynamicRenderingColorDefinition) -> Self {
        let mut color_blend = ash::vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(ash::vk::ColorComponentFlags::RGBA)
            .blend_enable(false)
            .src_color_blend_factor(ash::vk::BlendFactor::ONE)
            .dst_color_blend_factor(ash::vk::BlendFactor::ZERO)
            .color_blend_op(ash::vk::BlendOp::ADD)
            .src_alpha_blend_factor(ash::vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(ash::vk::BlendFactor::ONE)
            .alpha_blend_op(ash::vk::BlendOp::ADD);
        
        if val.alpha_blend {
            color_blend.blend_enable = 1u32;
            color_blend.src_color_blend_factor = ash::vk::BlendFactor::SRC_ALPHA;
            color_blend.dst_color_blend_factor = ash::vk::BlendFactor::ONE_MINUS_SRC_ALPHA;
            color_blend.color_blend_op = ash::vk::BlendOp::ADD;
            color_blend.src_alpha_blend_factor = ash::vk::BlendFactor::ONE;
            color_blend.dst_alpha_blend_factor = ash::vk::BlendFactor::ONE_MINUS_SRC_ALPHA;
            color_blend.alpha_blend_op = ash::vk::BlendOp::ADD;
        }

        color_blend
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

impl<T> From<RenderingAttachmentSetup<T>> for crate::ash::vk::AttachmentLoadOp
where
    T: Into<crate::ash::vk::ClearValue>,
{
    fn from(val: RenderingAttachmentSetup<T>) -> Self {
        match val {
            RenderingAttachmentSetup::Load => crate::ash::vk::AttachmentLoadOp::LOAD,
            RenderingAttachmentSetup::Clear(maybe_clear) => match maybe_clear {
                Some(_) => crate::ash::vk::AttachmentLoadOp::CLEAR,
                None => crate::ash::vk::AttachmentLoadOp::DONT_CARE,
            },
        }
    }
}

impl<T> From<RenderingAttachmentSetup<T>> for crate::ash::vk::ClearValue
where
    T: Into<crate::ash::vk::ClearValue>,
{
    fn from(val: RenderingAttachmentSetup<T>) -> Self {
        match val {
            RenderingAttachmentSetup::Load => crate::ash::vk::ClearValue::default(),
            RenderingAttachmentSetup::Clear(maybe_clear) => match maybe_clear {
                Some(clear_value) => clear_value.into(),
                None => crate::ash::vk::ClearValue::default(),
            },
        }
    }
}

fn attachment_info<'a, T: Into<ash::vk::ClearValue> + Clone>(
    image_view: ash::vk::ImageView,
    image_layout: ash::vk::ImageLayout,
    setup: &RenderingAttachmentSetup<T>,
    store_op: AttachmentStoreOp,
) -> ash::vk::RenderingAttachmentInfo<'a> {
    ash::vk::RenderingAttachmentInfo::default()
        .image_view(image_view)
        .image_layout(image_layout)
        .store_op(store_op.into())
        .load_op(setup.clone().into())
        .clear_value(setup.clone().into())
}

fn depth_stencil_attachment_info<'a, T: Into<ash::vk::ClearValue> + Clone>(
    image_view: ash::vk::ImageView,
    setup: &RenderingAttachmentSetup<T>,
    store_op: AttachmentStoreOp,
) -> ash::vk::RenderingAttachmentInfo<'a> {
    // Match the framework's combined depth/stencil layout without requiring
    // the separateDepthStencilLayouts feature.
    attachment_info(
        image_view,
        ash::vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        setup,
        store_op,
    )
}

pub(crate) fn rendering_info<'a>(
    extent: ash::vk::Extent2D,
    colors: &'a [ash::vk::RenderingAttachmentInfo<'a>],
    depth: Option<&'a ash::vk::RenderingAttachmentInfo<'a>>,
    stencil: Option<&'a ash::vk::RenderingAttachmentInfo<'a>>,
) -> ash::vk::RenderingInfo<'a> {
    if let (Some(depth), Some(stencil)) = (depth, stencil) {
        if depth.image_view != ash::vk::ImageView::null()
            && stencil.image_view != ash::vk::ImageView::null()
        {
            assert_eq!(
                depth.image_view, stencil.image_view,
                "Depth and stencil attachments must use the same image view"
            );
        }
    }

    let mut info = ash::vk::RenderingInfo::default()
        .color_attachments(colors)
        .layer_count(1)
        .render_area(ash::vk::Rect2D::default().extent(extent));
    if let Some(depth) = depth {
        info = info.depth_attachment(depth);
    }
    if let Some(stencil) = stencil {
        info = info.stencil_attachment(stencil);
    }
    info
}

#[derive(Clone)]
pub struct DynamicRenderingColorAttachment(
    Arc<ImageView>,
    RenderingAttachmentSetup<ColorClearValues>,
    AttachmentStoreOp,
);

impl<'a> From<&DynamicRenderingColorAttachment> for crate::ash::vk::RenderingAttachmentInfo<'a> {
    fn from(val: &DynamicRenderingColorAttachment) -> Self {
        attachment_info(
            ash::vk::ImageView::from_raw(val.0.native_handle()),
            ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            &val.1,
            val.2,
        )
    }
}

impl<'a> From<DynamicRenderingColorAttachment> for crate::ash::vk::RenderingAttachmentInfo<'a> {
    fn from(val: DynamicRenderingColorAttachment) -> Self {
        (&val).into()
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

impl<'a> From<&DynamicRenderingDepthAttachment> for crate::ash::vk::RenderingAttachmentInfo<'a> {
    fn from(val: &DynamicRenderingDepthAttachment) -> Self {
        depth_stencil_attachment_info(
            ash::vk::ImageView::from_raw(val.0.native_handle()),
            &val.1,
            val.2,
        )
    }
}

impl<'a> From<DynamicRenderingDepthAttachment> for crate::ash::vk::RenderingAttachmentInfo<'a> {
    fn from(val: DynamicRenderingDepthAttachment) -> Self {
        (&val).into()
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

impl<'a> From<&DynamicRenderingStencilAttachment> for crate::ash::vk::RenderingAttachmentInfo<'a> {
    fn from(val: &DynamicRenderingStencilAttachment) -> Self {
        depth_stencil_attachment_info(
            ash::vk::ImageView::from_raw(val.0.native_handle()),
            &val.1,
            val.2,
        )
    }
}

impl<'a> From<DynamicRenderingStencilAttachment> for crate::ash::vk::RenderingAttachmentInfo<'a> {
    fn from(val: DynamicRenderingStencilAttachment) -> Self {
        (&val).into()
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

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;

    #[test]
    fn borrowed_color_definition_preserves_format_and_blend_state() {
        for format in [vk::Format::R8G8B8A8_UNORM, vk::Format::R32_UINT] {
            let definition = DynamicRenderingColorDefinition::new(format.into());
            assert_eq!(vk::Format::from(&definition), format);
            let blend = vk::PipelineColorBlendAttachmentState::from(&definition);
            assert_eq!(blend.blend_enable, vk::FALSE);
            assert_eq!(blend.color_write_mask, vk::ColorComponentFlags::RGBA);
        }
    }

    #[test]
    fn attachment_metadata_preserves_layout_load_store_and_clear_values() {
        let view = vk::ImageView::from_raw(1);
        let color = attachment_info(
            view,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            &RenderingAttachmentSetup::clear(ColorClearValues::UVec4(1, 2, 3, 4)),
            AttachmentStoreOp::Store,
        );
        assert_eq!(color.image_view, view);
        assert_eq!(
            color.image_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
        assert_eq!(color.load_op, vk::AttachmentLoadOp::CLEAR);
        assert_eq!(color.store_op, vk::AttachmentStoreOp::STORE);
        assert_eq!(unsafe { color.clear_value.color.uint32 }, [1, 2, 3, 4]);

        let depth = depth_stencil_attachment_info(
            view,
            &RenderingAttachmentSetup::clear(DepthClearValues::new(0.5)),
            AttachmentStoreOp::DontCare,
        );
        assert_eq!(
            depth.image_layout,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        );
        assert_eq!(depth.load_op, vk::AttachmentLoadOp::CLEAR);
        assert_eq!(depth.store_op, vk::AttachmentStoreOp::DONT_CARE);
        assert_eq!(unsafe { depth.clear_value.depth_stencil.depth }, 0.5);

        let stencil = depth_stencil_attachment_info(
            view,
            &RenderingAttachmentSetup::clear(StencilClearValues::new(7)),
            AttachmentStoreOp::Store,
        );
        assert_eq!(
            stencil.image_layout,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        );
        assert_eq!(unsafe { stencil.clear_value.depth_stencil.stencil }, 7);

        for (setup, expected) in [
            (
                RenderingAttachmentSetup::<DepthClearValues>::load(),
                vk::AttachmentLoadOp::LOAD,
            ),
            (
                RenderingAttachmentSetup::<DepthClearValues>::dont_care(),
                vk::AttachmentLoadOp::DONT_CARE,
            ),
        ] {
            let info = depth_stencil_attachment_info(view, &setup, AttachmentStoreOp::Store);
            assert_eq!(info.load_op, expected);
        }
    }

    #[test]
    fn rendering_info_keeps_independent_depth_and_stencil_pointers() {
        let colors =
            [vk::RenderingAttachmentInfo::default().image_view(vk::ImageView::from_raw(1))];
        let depth = vk::RenderingAttachmentInfo::default().image_view(vk::ImageView::from_raw(2));
        let stencil = vk::RenderingAttachmentInfo::default().image_view(vk::ImageView::from_raw(2));
        let extent = vk::Extent2D {
            width: 640,
            height: 480,
        };
        for has_colors in [false, true] {
            for has_depth in [false, true] {
                for has_stencil in [false, true] {
                    let colors = if has_colors { &colors[..] } else { &[] };
                    let info = rendering_info(
                        extent,
                        colors,
                        has_depth.then_some(&depth),
                        has_stencil.then_some(&stencil),
                    );
                    assert_eq!(info.color_attachment_count as usize, colors.len());
                    assert_eq!(info.p_color_attachments, colors.as_ptr());
                    assert_eq!(
                        info.p_depth_attachment,
                        if has_depth { &depth } else { std::ptr::null() }
                    );
                    assert_eq!(
                        info.p_stencil_attachment,
                        if has_stencil {
                            &stencil
                        } else {
                            std::ptr::null()
                        }
                    );
                    if has_depth && has_stencil {
                        assert_ne!(info.p_depth_attachment, info.p_stencil_attachment);
                    }
                    assert_eq!(info.render_area.extent, extent);
                    assert_eq!(info.render_area.offset, vk::Offset2D::default());
                    assert_eq!(info.layer_count, 1);
                    assert_eq!(info.view_mask, 0);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "Depth and stencil attachments must use the same image view")]
    fn rendering_info_rejects_different_depth_stencil_views() {
        let depth = vk::RenderingAttachmentInfo::default().image_view(vk::ImageView::from_raw(1));
        let stencil = vk::RenderingAttachmentInfo::default().image_view(vk::ImageView::from_raw(2));
        let _ = rendering_info(vk::Extent2D::default(), &[], Some(&depth), Some(&stencil));
    }
}
