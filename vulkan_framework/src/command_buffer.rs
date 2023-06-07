use std::{
    collections::HashSet,
    hash::Hash,
    sync::{Arc, Mutex},
};

use ash::vk::Handle;

use crate::{
    command_pool::{CommandPool, CommandPoolOwned},
    device::DeviceOwned,
    framebuffer::{Framebuffer, FramebufferTrait},
    graphics_pipeline::GraphicsPipeline,
    image::{
        Image1DTrait, Image2DTrait, ImageAspects, ImageDimensions, ImageLayout,
        ImageSubresourceLayers, ImageSubresourceRange, ImageTrait,
    },
    instance::InstanceOwned,
    pipeline_layout::PipelineLayout,
    pipeline_stage::PipelineStages,
    prelude::{VulkanError, VulkanResult},
    queue_family::QueueFamily,
    renderpass::RenderPassCompatible,
};
use crate::{
    compute_pipeline::ComputePipeline, descriptor_set::DescriptorSet, device::Device,
    queue_family::QueueFamilyOwned,
};

enum CommandBufferReferencedResource {
    ComputePipeline(Arc<ComputePipeline>),
    GraphicsPipeline(Arc<GraphicsPipeline>),
    DescriptorSet(Arc<DescriptorSet>),
    PipelineLayout(Arc<PipelineLayout>),
    Framebuffer(Arc<dyn FramebufferTrait>),
    Image(Arc<dyn ImageTrait>),
}

impl Eq for CommandBufferReferencedResource {}

impl CommandBufferReferencedResource {
    pub fn hash(&self) -> u128 {
        match self {
            Self::ComputePipeline(l0) => l0.native_handle() as u128,
            Self::DescriptorSet(l0) => (0b0001u128 << 124u128) | (l0.native_handle() as u128),
            Self::PipelineLayout(l0) => (0b0010u128 << 124u128) | (l0.native_handle() as u128),
            Self::Image(l0) => (0b0011u128 << 124u128) | (l0.native_handle() as u128),
            Self::Framebuffer(l0) => (0b0100u128 << 124u128) | (l0.native_handle() as u128),
            Self::GraphicsPipeline(l0) => (0b0101u128 << 124u128) | (l0.native_handle() as u128),
        }
    }
}

impl PartialEq for CommandBufferReferencedResource {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ComputePipeline(l0), Self::ComputePipeline(r0)) => {
                l0.native_handle() == r0.native_handle()
            }
            (Self::DescriptorSet(l0), Self::DescriptorSet(r0)) => {
                l0.native_handle() == r0.native_handle()
            }
            (Self::PipelineLayout(l0), Self::PipelineLayout(r0)) => {
                l0.native_handle() == r0.native_handle()
            }
            (Self::GraphicsPipeline(l0), Self::GraphicsPipeline(r0)) => {
                l0.native_handle() == r0.native_handle()
            }
            (Self::Framebuffer(l0), Self::Framebuffer(r0)) => {
                l0.native_handle() == r0.native_handle()
            }
            (Self::Image(l0), Self::Image(r0)) => l0.native_handle() == r0.native_handle(),
            _ => false,
        }
    }
}

impl Hash for CommandBufferReferencedResource {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u128(self.hash())
    }
}

/*
 * Provided by VK_KHR_acceleration_structure
 */
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum AccessFlagAccelerationStructureKHR {
    AccelerationStructureRead = 0x00200000u32,
    AccelerationStructureWrite = 0x00400000u32,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum AccessFlag {
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

/*
 * Provided by VK_KHR_acceleration_structure
 */
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct AccessFlagsAccelerationStructureKHR {
    acceleration_structure_read: bool,
    acceleration_structure_write: bool,
}

impl AccessFlagsAccelerationStructureKHR {
    pub fn empty() -> Self {
        Self {
            acceleration_structure_read: false,
            acceleration_structure_write: false,
        }
    }

    pub fn from(flags: &[AccessFlagAccelerationStructureKHR]) -> Self {
        Self {
            acceleration_structure_read: flags
                .contains(&AccessFlagAccelerationStructureKHR::AccelerationStructureRead),
            acceleration_structure_write: flags
                .contains(&AccessFlagAccelerationStructureKHR::AccelerationStructureWrite),
        }
    }

    pub fn new(acceleration_structure_read: bool, acceleration_structure_write: bool) -> Self {
        Self {
            acceleration_structure_read,
            acceleration_structure_write,
        }
    }

    pub(crate) fn ash_flags(&self) -> ash::vk::AccessFlags {
        (match self.acceleration_structure_read {
            true => ash::vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
            false => ash::vk::AccessFlags::empty(),
        }) | (match self.acceleration_structure_write {
            true => ash::vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            false => ash::vk::AccessFlags::empty(),
        })
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct AccessFlagsSpecifier {
    indirect_command_read: bool,
    index_read: bool,
    vertex_attribure_read: bool,
    uniform_read: bool,
    input_attachment_read: bool,
    shader_read: bool,
    shader_write: bool,
    color_attachment_read: bool,
    color_attachment_write: bool,
    depth_stencil_attachment_read: bool,
    depth_stencil_attachment_write: bool,
    transfer_read: bool,
    transfer_write: bool,
    host_read: bool,
    host_write: bool,
    memory_read: bool,
    memory_write: bool,
    acceleration_structure: AccessFlagsAccelerationStructureKHR,
}

impl AccessFlagsSpecifier {
    pub fn new(
        indirect_command_read: bool,
        index_read: bool,
        vertex_attribure_read: bool,
        uniform_read: bool,
        input_attachment_read: bool,
        shader_read: bool,
        shader_write: bool,
        color_attachment_read: bool,
        color_attachment_write: bool,
        depth_stencil_attachment_read: bool,
        depth_stencil_attachment_write: bool,
        transfer_read: bool,
        transfer_write: bool,
        host_read: bool,
        host_write: bool,
        memory_read: bool,
        memory_write: bool,
        maybe_acceleration_structure: Option<AccessFlagsAccelerationStructureKHR>,
    ) -> Self {
        Self {
            indirect_command_read,
            index_read,
            vertex_attribure_read,
            uniform_read,
            input_attachment_read,
            shader_read,
            shader_write,
            color_attachment_read,
            color_attachment_write,
            depth_stencil_attachment_read,
            depth_stencil_attachment_write,
            transfer_read,
            transfer_write,
            host_read,
            host_write,
            memory_read,
            memory_write,
            acceleration_structure: match maybe_acceleration_structure {
                Some(access_accel_structure_khr) => access_accel_structure_khr,
                None => AccessFlagsAccelerationStructureKHR::empty(),
            },
        }
    }

    pub fn from(
        flags: &[AccessFlag],
        acceleration_structure_flags: Option<&[AccessFlagAccelerationStructureKHR]>,
    ) -> Self {
        Self::new(
            flags.contains(&AccessFlag::IndirectCommandRead),
            flags.contains(&AccessFlag::IndexRead),
            flags.contains(&AccessFlag::VertexAttribureRead),
            flags.contains(&AccessFlag::UniformRead),
            flags.contains(&AccessFlag::InputAttachmentRead),
            flags.contains(&AccessFlag::ShaderRead),
            flags.contains(&AccessFlag::ShaderWrite),
            flags.contains(&AccessFlag::ColorAttachmentRead),
            flags.contains(&AccessFlag::ColorAttachmentWrite),
            flags.contains(&AccessFlag::DepthStencilAttachmentRead),
            flags.contains(&AccessFlag::DepthStencilAttachmentWrite),
            flags.contains(&AccessFlag::TransferRead),
            flags.contains(&AccessFlag::TransferWrite),
            flags.contains(&AccessFlag::HostRead),
            flags.contains(&AccessFlag::HostWrite),
            flags.contains(&AccessFlag::MemoryRead),
            flags.contains(&AccessFlag::MemoryWrite),
            acceleration_structure_flags.map(AccessFlagsAccelerationStructureKHR::from),
        )
    }

    pub(crate) fn ash_flags(&self) -> ash::vk::AccessFlags {
        (ash::vk::AccessFlags::empty())
            | (match self.indirect_command_read {
                true => ash::vk::AccessFlags::INDIRECT_COMMAND_READ,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.index_read {
                true => ash::vk::AccessFlags::INDEX_READ,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.vertex_attribure_read {
                true => ash::vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.uniform_read {
                true => ash::vk::AccessFlags::UNIFORM_READ,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.input_attachment_read {
                true => ash::vk::AccessFlags::INPUT_ATTACHMENT_READ,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.shader_read {
                true => ash::vk::AccessFlags::SHADER_READ,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.shader_write {
                true => ash::vk::AccessFlags::SHADER_WRITE,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.color_attachment_read {
                true => ash::vk::AccessFlags::COLOR_ATTACHMENT_READ,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.color_attachment_write {
                true => ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.depth_stencil_attachment_read {
                true => ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.depth_stencil_attachment_write {
                true => ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.transfer_read {
                true => ash::vk::AccessFlags::TRANSFER_READ,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.transfer_write {
                true => ash::vk::AccessFlags::TRANSFER_WRITE,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.host_read {
                true => ash::vk::AccessFlags::HOST_READ,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.host_write {
                true => ash::vk::AccessFlags::HOST_WRITE,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.memory_read {
                true => ash::vk::AccessFlags::MEMORY_READ,
                false => ash::vk::AccessFlags::empty(),
            })
            | (match self.memory_write {
                true => ash::vk::AccessFlags::MEMORY_WRITE,
                false => ash::vk::AccessFlags::empty(),
            })
            | (self.acceleration_structure.ash_flags())
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum AccessFlags {
    Managed(AccessFlagsSpecifier),
    Unmanaged(u32),
}

impl AccessFlags {
    pub fn empty() -> Self {
        Self::Unmanaged(0)
    }

    pub fn from_raw(flags: u32) -> Self {
        Self::Unmanaged(flags)
    }

    pub fn from(flags: AccessFlagsSpecifier) -> Self {
        Self::Managed(flags)
    }

    pub(crate) fn ash_flags(&self) -> ash::vk::AccessFlags {
        match self {
            Self::Managed(spec) => spec.ash_flags(),
            Self::Unmanaged(raw) => ash::vk::AccessFlags::from_raw(*raw),
        }
    }
}

pub struct ImageMemoryBarrier {
    src_stages: PipelineStages,
    src_access: AccessFlags,
    dst_stages: PipelineStages,
    dst_access: AccessFlags,
    image: Arc<dyn ImageTrait>,
    srr: ImageSubresourceRange,
    old_layout: ImageLayout,
    new_layout: ImageLayout,
    src_queue_family: Arc<QueueFamily>,
    dst_queue_family: Arc<QueueFamily>,
}

impl ImageMemoryBarrier {
    pub(crate) fn ash_src_access_mask_flags(&self) -> ash::vk::AccessFlags {
        self.src_access.ash_flags()
    }

    pub(crate) fn ash_dst_access_mask_flags(&self) -> ash::vk::AccessFlags {
        self.dst_access.ash_flags()
    }

    pub(crate) fn ash_src_queue_family(&self) -> u32 {
        self.src_queue_family.get_family_index()
    }

    pub(crate) fn ash_dst_queue_family(&self) -> u32 {
        self.dst_queue_family.get_family_index()
    }

    pub(crate) fn ash_image_handle(&self) -> ash::vk::Image {
        ash::vk::Image::from_raw(self.image.native_handle())
    }

    pub(crate) fn ash_src_flags(&self) -> ash::vk::PipelineStageFlags {
        self.src_stages.ash_flags()
    }

    pub(crate) fn ash_dst_flags(&self) -> ash::vk::PipelineStageFlags {
        self.dst_stages.ash_flags()
    }

    pub(crate) fn ash_subresource_range(&self) -> ash::vk::ImageSubresourceRange {
        self.srr.ash_subresource_range()
    }

    pub fn from_subnresource_range(
        src_stages: PipelineStages,
        src_access: AccessFlags,
        dst_stages: PipelineStages,
        dst_access: AccessFlags,
        image: Arc<dyn ImageTrait>,
        srr: ImageSubresourceRange,
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
            image,
            srr,
            old_layout,
            new_layout,
            src_queue_family,
            dst_queue_family,
        }
    }

    pub fn new(
        src_stages: PipelineStages,
        src_access: AccessFlags,
        dst_stages: PipelineStages,
        dst_access: AccessFlags,
        image: Arc<dyn ImageTrait>,
        maybe_image_aspect: Option<ImageAspects>,
        maybe_base_mip_level: Option<u32>,
        maybe_mip_levels_count: Option<u32>,
        maybe_base_array_layer: Option<u32>,
        maybe_array_layers_count: Option<u32>,
        old_layout: ImageLayout,
        new_layout: ImageLayout,
        src_queue_family: Arc<QueueFamily>,
        dst_queue_family: Arc<QueueFamily>,
    ) -> Self {
        let srr = ImageSubresourceRange::from(
            image.clone(),
            maybe_image_aspect,
            maybe_base_mip_level,
            maybe_mip_levels_count,
            maybe_base_array_layer,
            maybe_array_layers_count,
        );

        Self {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            image,
            srr,
            old_layout,
            new_layout,
            src_queue_family,
            dst_queue_family,
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum ColorClearValues {
    Vec4(f32, f32, f32, f32),
    IVec4(i32, i32, i32, i32),
    UVec4(u32, u32, u32, u32),
}

#[derive(Copy, Clone, PartialEq)]
pub struct ClearValues {
    color: Option<ColorClearValues>,
}

impl ClearValues {
    pub fn new(color: Option<ColorClearValues>) -> Self {
        Self { color }
    }

    pub(crate) fn ash_clear(&self) -> ash::vk::ClearValue {
        let mut result = ash::vk::ClearValue::default();

        match self.color {
            Some(color) => match color {
                ColorClearValues::Vec4(r, g, b, a) => {
                    result.color.float32 = [r, g, b, a];
                }
                ColorClearValues::IVec4(r, g, b, a) => {
                    result.color.int32 = [r, g, b, a];
                }
                ColorClearValues::UVec4(r, g, b, a) => {
                    result.color.uint32 = [r, g, b, a];
                }
            },
            None => {}
        }

        result
    }
}

pub struct CommandBufferRecorder<'a> {
    device: Arc<Device>, // this field is repeated to speed-up execution, otherwise a ton of Arc<>.clone() will be performed
    command_buffer: &'a dyn CommandBufferCrateTrait,

    used_resources: HashSet<CommandBufferReferencedResource>,
}

impl<'a> CommandBufferRecorder<'a> {
    pub fn draw(
        &mut self,
        first_vertex_index: u32,
        vertex_count: u32,
        first_instance_index: u32,
        instance_count: u32,
    ) {
        unsafe {
            self.device.ash_handle().cmd_draw(
                self.command_buffer.ash_handle(),
                vertex_count,
                instance_count,
                first_vertex_index,
                first_instance_index,
            )
        }
    }

    pub fn bind_graphics_pipeline(&mut self, graphics_pipeline: Arc<GraphicsPipeline>) {
        unsafe {
            self.device.ash_handle().cmd_bind_pipeline(
                self.command_buffer.ash_handle(),
                ash::vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline.ash_handle(),
            )
        }

        self.used_resources
            .insert(CommandBufferReferencedResource::GraphicsPipeline(
                graphics_pipeline,
            ));
    }

    pub fn end_renderpass(&mut self) {
        unsafe {
            self.device
                .ash_handle()
                .cmd_end_render_pass(self.command_buffer.ash_handle())
        }
    }

    /**
     * Corresponds to vkCmdBeginRenderpass, where the framebuffer is the provided one,
     * the renderpass is the RenderPass that Framebuffer was created from and the render area
     * is the whole area identified by the framebuffer
     */
    pub fn begin_renderpass(
        &mut self,
        framebuffer: Arc<Framebuffer>,
        clear_values: &[ClearValues],
    ) {
        let ash_clear_values: smallvec::SmallVec<[ash::vk::ClearValue; 32]> =
            clear_values.iter().map(|cv| cv.ash_clear()).collect();

        let render_pass_begin_info = ash::vk::RenderPassBeginInfo::builder()
            .clear_values(ash_clear_values.as_slice())
            .framebuffer(ash::vk::Framebuffer::from_raw(framebuffer.native_handle()))
            .render_pass(framebuffer.get_parent_renderpass().ash_handle())
            .render_area(
                ash::vk::Rect2D::builder()
                    .offset(ash::vk::Offset2D::builder().x(0).y(0).build())
                    .extent(
                        ash::vk::Extent2D::builder()
                            .width(framebuffer.dimensions().width())
                            .height(framebuffer.dimensions().height())
                            .build(),
                    )
                    .build(),
            )
            .build();

        self.used_resources
            .insert(CommandBufferReferencedResource::Framebuffer(framebuffer));

        unsafe {
            self.device.ash_handle().cmd_begin_render_pass(
                self.command_buffer.ash_handle(),
                &render_pass_begin_info,
                ash::vk::SubpassContents::INLINE,
            )
        }
    }

    pub fn copy_image(
        &mut self,
        src_layout: ImageLayout,
        src_subresource: ImageSubresourceLayers,
        src: Arc<dyn ImageTrait>,
        dst_layout: ImageLayout,
        dst_subresource: ImageSubresourceLayers,
        dst: Arc<dyn ImageTrait>,
        extent: ImageDimensions,
        //srr: ImageSubresourceRange,
    ) {
        let src_offset = ash::vk::Offset3D::builder().x(0).y(0).z(0).build();

        let dst_offset = ash::vk::Offset3D::builder().x(0).y(0).z(0).build();

        let regions = ash::vk::ImageCopy::builder()
            .extent(extent.ash_extent_3d())
            .dst_subresource(dst_subresource.ash_subresource_layers())
            .src_subresource(src_subresource.ash_subresource_layers())
            .dst_offset(dst_offset)
            .src_offset(src_offset)
            .build();

        self.used_resources
            .insert(CommandBufferReferencedResource::Image(src.clone()));

        self.used_resources
            .insert(CommandBufferReferencedResource::Image(dst.clone()));

        unsafe {
            self.device.ash_handle().cmd_copy_image(
                self.command_buffer.ash_handle(),
                ash::vk::Image::from_raw(src.native_handle()),
                src_layout.ash_layout(),
                ash::vk::Image::from_raw(dst.native_handle()),
                dst_layout.ash_layout(),
                &[regions],
            );
        }
    }

    pub fn image_barrier(&mut self, dependency_info: ImageMemoryBarrier) {
        // TODO: check every resource is from the same device

        let image_memory_barrier = ash::vk::ImageMemoryBarrier::builder()
            .image(dependency_info.ash_image_handle())
            .old_layout(dependency_info.old_layout.ash_layout())
            .new_layout(dependency_info.new_layout.ash_layout())
            .src_queue_family_index(dependency_info.ash_src_queue_family())
            .dst_queue_family_index(dependency_info.ash_dst_queue_family())
            .src_access_mask(dependency_info.ash_src_access_mask_flags())
            .dst_access_mask(dependency_info.ash_dst_access_mask_flags())
            .subresource_range(dependency_info.ash_subresource_range())
            .build();

        self.used_resources
            .insert(CommandBufferReferencedResource::Image(
                dependency_info.image.clone(),
            ));

        unsafe {
            self.device.ash_handle().cmd_pipeline_barrier(
                self.command_buffer.ash_handle(),
                dependency_info.ash_src_flags(),
                dependency_info.ash_dst_flags(),
                ash::vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_memory_barrier],
            );
        }
    }

    pub fn bind_compute_pipeline(&mut self, compute_pipeline: Arc<ComputePipeline>) {
        unsafe {
            self.device.ash_handle().cmd_bind_pipeline(
                self.command_buffer.ash_handle(),
                ash::vk::PipelineBindPoint::COMPUTE,
                compute_pipeline.ash_handle(),
            )
        }

        self.used_resources
            .insert(CommandBufferReferencedResource::ComputePipeline(
                compute_pipeline,
            ));
    }

    pub fn bind_descriptor_sets(
        &mut self,
        pipeline_layout: Arc<PipelineLayout>,
        offset: u32,
        descriptor_sets: &[Arc<DescriptorSet>],
    ) {
        let mut sets = Vec::<ash::vk::DescriptorSet>::new();

        for ds in descriptor_sets.iter() {
            self.used_resources
                .insert(CommandBufferReferencedResource::DescriptorSet(ds.clone()));

            sets.push(ds.ash_handle());
        }

        unsafe {
            self.device.ash_handle().cmd_bind_descriptor_sets(
                self.command_buffer.ash_handle(),
                ash::vk::PipelineBindPoint::COMPUTE,
                pipeline_layout.ash_handle(),
                offset,
                sets.as_slice(),
                &[],
            )
        }

        self.used_resources
            .insert(CommandBufferReferencedResource::PipelineLayout(
                pipeline_layout,
            ));
    }

    pub fn push_constant_for_compute_shader(
        &mut self,
        pipeline_layout: Arc<PipelineLayout>,
        offset: u32,
        data: &[u8],
    ) {
        unsafe {
            self.device.ash_handle().cmd_push_constants(
                self.command_buffer.ash_handle(),
                pipeline_layout.ash_handle(),
                ash::vk::ShaderStageFlags::COMPUTE,
                offset,
                data,
            )
        }

        self.used_resources
            .insert(CommandBufferReferencedResource::PipelineLayout(
                pipeline_layout,
            ));
    }

    pub fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            self.device.ash_handle().cmd_dispatch(
                self.command_buffer.ash_handle(),
                group_count_x,
                group_count_y,
                group_count_z,
            )
        }
    }
}

pub trait CommandBufferTrait: CommandPoolOwned {
    fn native_handle(&self) -> u64;

    fn flag_execution_as_finished(&self);
}

pub(crate) trait CommandBufferCrateTrait: CommandBufferTrait {
    fn ash_handle(&self) -> ash::vk::CommandBuffer;
}

pub struct PrimaryCommandBuffer {
    command_pool: Arc<CommandPool>,
    command_buffer: ash::vk::CommandBuffer,
    resources_in_use: Mutex<HashSet<CommandBufferReferencedResource>>,
}

impl Drop for PrimaryCommandBuffer {
    fn drop(&mut self) {
        // Command buffers will be automatically freed when their command pool is destroyed, so we don't need explicit cleanup.
    }
}

impl CommandPoolOwned for PrimaryCommandBuffer {
    fn get_parent_command_pool(&self) -> Arc<CommandPool> {
        self.command_pool.clone()
    }
}

impl CommandBufferTrait for PrimaryCommandBuffer {
    fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.command_buffer)
    }

    fn flag_execution_as_finished(&self) {
        // TODO: if the record was a one-time-submit clear the list of used resources
        //self.recording_status.store(0, Ordering::Release);
    }
}

impl CommandBufferCrateTrait for PrimaryCommandBuffer {
    fn ash_handle(&self) -> ash::vk::CommandBuffer {
        self.command_buffer
    }
}

impl PrimaryCommandBuffer {
    pub fn record_commands<F>(&self, commands_writer_fn: F) -> VulkanResult<()>
    where
        F: Fn(&mut CommandBufferRecorder) + Sized,
    {
        let device = self
            .get_parent_command_pool()
            .get_parent_queue_family()
            .get_parent_device();

        match self.resources_in_use.lock() {
            Ok(mut resources_lck) => {
                let begin_info = ash::vk::CommandBufferBeginInfo::builder()
                    .flags(ash::vk::CommandBufferUsageFlags::empty() /*ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT*/)
                    .build();

                match unsafe {
                    device
                        .ash_handle()
                        .begin_command_buffer(self.ash_handle(), &begin_info)
                } {
                    Ok(_) => {
                        let mut recorder = CommandBufferRecorder {
                            device: device.clone(),
                            command_buffer: self,
                            used_resources: HashSet::new(),
                        };

                        commands_writer_fn(&mut recorder);

                        match unsafe { device.ash_handle().end_command_buffer(self.ash_handle()) } {
                            Ok(()) => {
                                *resources_lck = recorder.used_resources;

                                Ok(())
                            }
                            Err(_err) => {
                                #[cfg(debug_assertions)]
                                {
                                    panic!("Error creating the command buffer recorder: the command buffer already is in recording state!")
                                }

                                Err(VulkanError::Unspecified)
                            }
                        }
                    }
                    Err(err) => {
                        // TODO: the command buffer is in the previous state... A good one unless the error is DEVICE_LOST

                        #[cfg(debug_assertions)]
                        {
                            panic!("Error opening the command buffer for writing: {}", err);
                        }

                        Err(VulkanError::Unspecified)
                    }
                }
            }
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    panic!(
                        "Error opening the command buffer for writing: Error acquiring mutex: {}",
                        err
                    );
                }

                Err(VulkanError::Unspecified)
            }
        }
    }

    pub fn new(
        command_pool: Arc<CommandPool>,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = command_pool.get_parent_queue_family().get_parent_device();

        let create_info = ash::vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool.ash_handle())
            .level(ash::vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1)
            .build();

        match unsafe { device.ash_handle().allocate_command_buffers(&create_info) } {
            Ok(command_buffers) => {
                let command_buffer = command_buffers[0];

                let mut obj_name_bytes = vec![];
                if let Some(ext) = device.get_parent_instance().get_debug_ext_extension() {
                    if let Some(name) = debug_name {
                        for name_ch in name.as_bytes().iter() {
                            obj_name_bytes.push(*name_ch);
                        }
                        obj_name_bytes.push(0x00);

                        unsafe {
                            let object_name = std::ffi::CStr::from_bytes_with_nul_unchecked(
                                obj_name_bytes.as_slice(),
                            );
                            // set device name for debugging
                            let dbg_info = ash::vk::DebugUtilsObjectNameInfoEXT::builder()
                                .object_type(ash::vk::ObjectType::COMMAND_BUFFER)
                                .object_handle(ash::vk::Handle::as_raw(command_buffer))
                                .object_name(object_name)
                                .build();

                            match ext.set_debug_utils_object_name(
                                device.ash_handle().handle(),
                                &dbg_info,
                            ) {
                                Ok(_) => {
                                    #[cfg(debug_assertions)]
                                    {
                                        println!("Command Pool Debug object name changed");
                                    }
                                }
                                Err(err) => {
                                    #[cfg(debug_assertions)]
                                    {
                                        panic!("Error setting the Debug name for the newly created Command Pool, will use handle. Error: {}", err)
                                    }
                                }
                            }
                        }
                    }
                }

                Ok(Arc::new(Self {
                    command_buffer,
                    command_pool,
                    resources_in_use: Mutex::new(HashSet::new()),
                }))
            }
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error creating the command buffer: {}", err)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }
}
