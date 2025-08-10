use std::sync::Arc;

use ash::vk::Handle;

use crate::{
    buffer::BufferSubresourceRange,
    image::{ImageLayout, ImageSubresourceRange},
    pipeline_stage::PipelineStages,
    queue_family::QueueFamily,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MemoryAccessAs {
    IndirectCommandRead,
    IndexRead,
    VertexAttribureRead,
    UniformRead,
    InputAttachmentRead,
    ShaderRead,
    ShaderWrite,
    ColorAttachmentRead,
    ColorAttachmentWrite,
    DepthStencilAttachmentRead,
    DepthStencilAttachmentWrite,
    TransferRead,
    TransferWrite,
    HostRead,
    HostWrite,
    MemoryRead,
    MemoryWrite,
}

impl From<MemoryAccessAs> for ash::vk::AccessFlags2 {
    fn from(val: MemoryAccessAs) -> Self {
        type AshFlags = ash::vk::AccessFlags2;
        match val {
            MemoryAccessAs::IndirectCommandRead => AshFlags::INDIRECT_COMMAND_READ,
            MemoryAccessAs::IndexRead => AshFlags::INDEX_READ,
            MemoryAccessAs::VertexAttribureRead => AshFlags::VERTEX_ATTRIBUTE_READ,
            MemoryAccessAs::UniformRead => AshFlags::UNIFORM_READ,
            MemoryAccessAs::InputAttachmentRead => AshFlags::INPUT_ATTACHMENT_READ,
            MemoryAccessAs::ShaderRead => AshFlags::SHADER_READ,
            MemoryAccessAs::ShaderWrite => AshFlags::SHADER_WRITE,
            MemoryAccessAs::ColorAttachmentRead => AshFlags::COLOR_ATTACHMENT_READ,
            MemoryAccessAs::ColorAttachmentWrite => AshFlags::COLOR_ATTACHMENT_WRITE,
            MemoryAccessAs::DepthStencilAttachmentRead => AshFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            MemoryAccessAs::DepthStencilAttachmentWrite => AshFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            MemoryAccessAs::TransferRead => AshFlags::TRANSFER_READ,
            MemoryAccessAs::TransferWrite => AshFlags::TRANSFER_WRITE,
            MemoryAccessAs::HostRead => AshFlags::HOST_READ,
            MemoryAccessAs::HostWrite => AshFlags::HOST_WRITE,
            MemoryAccessAs::MemoryRead => AshFlags::MEMORY_READ,
            MemoryAccessAs::MemoryWrite => AshFlags::MEMORY_WRITE,
        }
    }
}

/// Represents AccessFlags2
///
/// When the contained value is None it is the same as having VK_ACCESS_2_NONE
/// otherwise at least one access flag is present.
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub struct MemoryAccess(Option<ash::vk::AccessFlags2>);

impl From<&[MemoryAccessAs]> for MemoryAccess {
    fn from(value: &[MemoryAccessAs]) -> Self {
        let mut flags = ash::vk::AccessFlags2::empty();
        for v in value.iter() {
            flags |= v.to_owned().into();
        }

        Self(match flags {
            ash::vk::AccessFlags2::NONE => None,
            flags => Some(flags),
        })
    }
}

impl From<ash::vk::AccessFlags2> for MemoryAccess {
    fn from(value: ash::vk::AccessFlags2) -> Self {
        Self(match value {
            ash::vk::AccessFlags2::NONE => None,
            value => Some(value),
        })
    }
}

impl From<MemoryAccess> for ash::vk::AccessFlags2 {
    fn from(val: MemoryAccess) -> Self {
        match &val.0 {
            Some(flags) => flags.to_owned(),
            None => ash::vk::AccessFlags2::NONE,
        }
    }
}

pub struct BufferMemoryBarrier {
    subresource_range: BufferSubresourceRange,
    src_stages: PipelineStages,
    src_access: MemoryAccess,
    dst_stages: PipelineStages,
    dst_access: MemoryAccess,
    src_queue_family: Arc<QueueFamily>,
    dst_queue_family: Arc<QueueFamily>,
}

impl BufferMemoryBarrier {
    #[inline]
    pub fn subresource_range(&self) -> &BufferSubresourceRange {
        &self.subresource_range
    }

    #[inline]
    pub fn new(
        src_stages: PipelineStages,
        src_access: MemoryAccess,
        dst_stages: PipelineStages,
        dst_access: MemoryAccess,
        subresource_range: BufferSubresourceRange,
        src_queue_family: Arc<QueueFamily>,
        dst_queue_family: Arc<QueueFamily>,
    ) -> Self {
        Self {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            subresource_range,
            src_queue_family,
            dst_queue_family,
        }
    }
}

impl<'a> From<&BufferMemoryBarrier> for crate::ash::vk::BufferMemoryBarrier2<'a> {
    fn from(val: &BufferMemoryBarrier) -> Self {
        ash::vk::BufferMemoryBarrier2::default()
            .buffer(ash::vk::Buffer::from_raw(
                val.subresource_range().buffer().native_handle(),
            ))
            .src_access_mask(val.src_access.into())
            .src_stage_mask(val.src_stages.into())
            .dst_access_mask(val.dst_access.into())
            .dst_stage_mask(val.dst_stages.into())
            .src_queue_family_index(val.src_queue_family.get_family_index())
            .dst_queue_family_index(val.dst_queue_family.get_family_index())
            .offset(val.subresource_range().offset())
            .size(val.subresource_range().size())
    }
}

pub struct ImageMemoryBarrier {
    src_stages: PipelineStages,
    src_access: MemoryAccess,
    dst_stages: PipelineStages,
    dst_access: MemoryAccess,
    subresource_range: ImageSubresourceRange,
    old_layout: ImageLayout,
    new_layout: ImageLayout,
    src_queue_family: Arc<QueueFamily>,
    dst_queue_family: Arc<QueueFamily>,
}

impl ImageMemoryBarrier {
    #[inline]
    pub fn subresource_range(&self) -> &ImageSubresourceRange {
        &self.subresource_range
    }

    #[inline]
    pub fn new(
        src_stages: PipelineStages,
        src_access: MemoryAccess,
        dst_stages: PipelineStages,
        dst_access: MemoryAccess,
        subresource_range: ImageSubresourceRange,
        old_layout: ImageLayout,
        new_layout: ImageLayout,
        src_queue_family: Arc<QueueFamily>,
        dst_queue_family: Arc<QueueFamily>,
    ) -> Self {
        Self {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            subresource_range,
            old_layout,
            new_layout,
            src_queue_family,
            dst_queue_family,
        }
    }
}

impl<'a> From<&ImageMemoryBarrier> for ash::vk::ImageMemoryBarrier2<'a> {
    fn from(val: &ImageMemoryBarrier) -> Self {
        ash::vk::ImageMemoryBarrier2::default()
            .image(ash::vk::Image::from_raw(
                val.subresource_range.image().native_handle(),
            ))
            .src_access_mask(val.src_access.into())
            .src_stage_mask(val.src_stages.into())
            .dst_access_mask(val.dst_access.into())
            .dst_stage_mask(val.dst_stages.into())
            .old_layout(val.old_layout.into())
            .new_layout(val.new_layout.into())
            .src_queue_family_index(val.src_queue_family.get_family_index())
            .dst_queue_family_index(val.dst_queue_family.get_family_index())
            .subresource_range((&val.subresource_range).into())
    }
}

impl<'a> From<ImageMemoryBarrier> for ash::vk::ImageMemoryBarrier2<'a> {
    fn from(val: ImageMemoryBarrier) -> Self {
        (&val).into()
    }
}
