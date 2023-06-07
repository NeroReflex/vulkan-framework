use std::sync::Arc;

use crate::{
    device::{Device, DeviceOwned},
    image::{ImageFormat, ImageLayout, ImageMultisampling},
    instance::InstanceOwned,
    prelude::{VulkanError, VulkanResult},
};

const MAX_NUMBER_OF_SUBPASSES_NOT_REQUIRING_HEAP_ALLOC: usize = 8;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum AttachmentLoadOp {
    Load,
    Clear,
    DontCare,
}

impl AttachmentLoadOp {
    pub(crate) fn ash_op(&self) -> ash::vk::AttachmentLoadOp {
        match self {
            AttachmentLoadOp::Load => ash::vk::AttachmentLoadOp::LOAD,
            AttachmentLoadOp::DontCare => ash::vk::AttachmentLoadOp::DONT_CARE,
            AttachmentLoadOp::Clear => ash::vk::AttachmentLoadOp::CLEAR,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum AttachmentStoreOp {
    Store,
    DontCare,
}

impl AttachmentStoreOp {
    pub(crate) fn ash_op(&self) -> ash::vk::AttachmentStoreOp {
        match self {
            AttachmentStoreOp::Store => ash::vk::AttachmentStoreOp::STORE,
            AttachmentStoreOp::DontCare => ash::vk::AttachmentStoreOp::DONT_CARE,
        }
    }
}

pub struct AttachmentDescription {
    format: ImageFormat,
    samples: ImageMultisampling,
    initial_layout: ImageLayout,
    final_layout: ImageLayout,
    load_op: AttachmentLoadOp,
    store_op: AttachmentStoreOp,
    stencil_load_op: AttachmentLoadOp,
    stencil_store_op: AttachmentStoreOp,
}

impl AttachmentDescription {
    pub(crate) fn ash_description(&self) -> ash::vk::AttachmentDescription {
        ash::vk::AttachmentDescription::builder()
            .format(self.format.ash_format())
            .samples(self.samples.ash_samples())
            .initial_layout(self.initial_layout.ash_layout())
            .final_layout(self.final_layout.ash_layout())
            .load_op(self.load_op.ash_op())
            .store_op(self.store_op.ash_op())
            .stencil_load_op(self.stencil_load_op.ash_op())
            .stencil_store_op(self.stencil_store_op.ash_op())
            .build()
    }

    pub fn new(
        format: ImageFormat,
        samples: ImageMultisampling,
        initial_layout: ImageLayout,
        final_layout: ImageLayout,
        load_op: AttachmentLoadOp,
        store_op: AttachmentStoreOp,
        stencil_load_op: AttachmentLoadOp,
        stencil_store_op: AttachmentStoreOp,
    ) -> Self {
        Self {
            format,
            samples,
            initial_layout,
            final_layout,
            load_op,
            store_op,
            stencil_load_op,
            stencil_store_op,
        }
    }
}

pub struct RenderSubPass<'a> {
    pub(crate) input_color_attachment_indeces: &'a [u32],
    pub(crate) output_color_attachment_indeces: &'a [u32],
    pub(crate) output_depth_stencil_attachment_index: Option<u32>,
}

impl<'a> RenderSubPass<'a> {
    pub fn from(
        input_color_attachment_indeces: &'a [u32],
        output_color_attachment_indeces: &'a [u32],
        output_depth_stencil_attachment_index: Option<u32>,
    ) -> Self {
        Self {
            input_color_attachment_indeces,
            output_color_attachment_indeces,
            output_depth_stencil_attachment_index,
        }
    }
}

pub struct RenderPass {
    device: Arc<Device>,
    renderpass: ash::vk::RenderPass,
}

impl DeviceOwned for RenderPass {
    fn get_parent_device(&self) -> Arc<crate::device::Device> {
        self.device.clone()
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_render_pass(
                self.renderpass,
                self.device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

impl RenderPass {
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.renderpass)
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::RenderPass {
        self.renderpass
    }

    pub fn new<'a>(
        device: Arc<Device>,
        attachments: &'a [AttachmentDescription],
        subpasses: &[RenderSubPass<'a>],
        // TODO: subpass dependencies!!!
    ) -> VulkanResult<Arc<Self>> {
        let attachment_descriptors = attachments
            .iter()
            .map(|a| a.ash_description())
            .collect::<smallvec::SmallVec<[ash::vk::AttachmentDescription; 32]>>(
        );
        let mut subpass_definitions = smallvec::SmallVec::<
            [ash::vk::SubpassDescription; MAX_NUMBER_OF_SUBPASSES_NOT_REQUIRING_HEAP_ALLOC],
        >::with_capacity(subpasses.len());
        let mut input_attachment_references_by_subpass = smallvec::SmallVec::<
            [smallvec::SmallVec<[ash::vk::AttachmentReference; 8]>;
                MAX_NUMBER_OF_SUBPASSES_NOT_REQUIRING_HEAP_ALLOC],
        >::with_capacity(subpasses.len());
        let mut output_color_attachment_references_by_subpass =
            smallvec::SmallVec::<
                [smallvec::SmallVec<[ash::vk::AttachmentReference; 8]>;
                    MAX_NUMBER_OF_SUBPASSES_NOT_REQUIRING_HEAP_ALLOC],
            >::with_capacity(subpasses.len());
        let mut output_depth_stencil_attachment_references_by_subpass =
            smallvec::SmallVec::<
                [ash::vk::AttachmentReference; MAX_NUMBER_OF_SUBPASSES_NOT_REQUIRING_HEAP_ALLOC],
            >::with_capacity(subpasses.len());

        for subpass in subpasses.iter() {
            let mut color_attachment_of_subpass: smallvec::SmallVec<
                [ash::vk::AttachmentReference; 8],
            > = smallvec::smallvec![];
            for color_attachment in subpass.output_color_attachment_indeces {
                if *color_attachment as usize >= attachment_descriptors.len() {
                    return Err(VulkanError::Unspecified);
                }

                color_attachment_of_subpass.push(
                    ash::vk::AttachmentReference::builder()
                        .attachment(*color_attachment)
                        .layout(ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .build(),
                )
            }
            output_color_attachment_references_by_subpass.push(color_attachment_of_subpass);

            let mut input_attachment_of_subpass: smallvec::SmallVec<
                [ash::vk::AttachmentReference; 8],
            > = smallvec::smallvec![];
            for input_attachment in subpass.input_color_attachment_indeces {
                if *input_attachment as usize >= attachment_descriptors.len() {
                    return Err(VulkanError::Unspecified);
                }

                input_attachment_of_subpass.push(
                    ash::vk::AttachmentReference::builder()
                        .attachment(*input_attachment)
                        .layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .build(),
                )
            }
            input_attachment_references_by_subpass.push(input_attachment_of_subpass);

            let mut subpass_uses_depth_stencil_attachment = false;
            if let Some(depth_stencil_attachment_index) =
                subpass.output_depth_stencil_attachment_index
            {
                assert!((depth_stencil_attachment_index as usize) < attachment_descriptors.len());

                subpass_uses_depth_stencil_attachment = true;

                output_depth_stencil_attachment_references_by_subpass.push(
                    ash::vk::AttachmentReference::builder()
                        .attachment(depth_stencil_attachment_index)
                        .layout(ash::vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .build(),
                )
            } else {
                output_depth_stencil_attachment_references_by_subpass.push(
                    ash::vk::AttachmentReference::builder()
                        .attachment(0)
                        .layout(ash::vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .build(),
                )
            }

            let mut subpass_definition = ash::vk::SubpassDescription::builder()
                .pipeline_bind_point(ash::vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(
                    output_color_attachment_references_by_subpass
                        [output_color_attachment_references_by_subpass.len() - 1]
                        .as_slice(),
                )
                .input_attachments(
                    input_attachment_references_by_subpass
                        [input_attachment_references_by_subpass.len() - 1]
                        .as_slice(),
                );
            if subpass_uses_depth_stencil_attachment {
                subpass_definition = subpass_definition.depth_stencil_attachment(
                    &output_depth_stencil_attachment_references_by_subpass
                        [output_depth_stencil_attachment_references_by_subpass.len() - 1],
                );
            }

            subpass_definitions.push(subpass_definition.build())
        }

        let create_info = ash::vk::RenderPassCreateInfo::builder()
            .attachments(attachment_descriptors.as_slice())
            .subpasses(subpass_definitions.as_slice())
            .build();

        match unsafe {
            device.ash_handle().create_render_pass(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        } {
            Ok(renderpass) => Ok(Arc::new(Self { device, renderpass })),
            Err(err) => Err(VulkanError::Unspecified),
        }
    }
}
