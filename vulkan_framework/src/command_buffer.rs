use std::{collections::HashSet, hash::Hash, sync::Arc};

#[cfg(feature = "better_mutex")]
use parking_lot::{const_mutex, Mutex};

#[cfg(not(feature = "better_mutex"))]
use std::sync::Mutex;

use ash::vk::{Handle, Offset2D};

use crate::{
    acceleration_structure::{
        bottom_level::BottomLevelAccelerationStructure, top_level::TopLevelAccelerationStructure,
        AllowedBuildingDevice,
    },
    binding_tables::RaytracingBindingTables,
    buffer::BufferTrait,
    command_pool::{CommandPool, CommandPoolOwned},
    device::DeviceOwned,
    framebuffer::{Framebuffer, FramebufferTrait, ImagelessFramebuffer},
    graphics_pipeline::{GraphicsPipeline, Scissor, Viewport},
    image::{
        Image1DTrait, Image2DTrait, Image3DDimensions, Image3DTrait, ImageDimensions, ImageLayout,
        ImageSubresourceLayers, ImageSubresourceRange, ImageTrait,
    },
    image_view::ImageView,
    pipeline_layout::PipelineLayout,
    pipeline_stage::PipelineStages,
    prelude::{FrameworkError, VulkanError, VulkanResult},
    queue_family::QueueFamily,
    raytracing_pipeline::RaytracingPipeline,
    renderpass::RenderPassCompatible,
};
use crate::{
    compute_pipeline::ComputePipeline, descriptor_set::DescriptorSet, device::Device,
    queue_family::QueueFamilyOwned,
};

enum CommandBufferReferencedResource {
    ComputePipeline(Arc<ComputePipeline>),
    GraphicsPipeline(Arc<GraphicsPipeline>),
    RaytracingPipeline(Arc<RaytracingPipeline>),
    DescriptorSet(Arc<DescriptorSet>),
    PipelineLayout(Arc<PipelineLayout>),
    Framebuffer(Arc<dyn FramebufferTrait>),
    Image(Arc<dyn ImageTrait>),
    Buffer(Arc<dyn BufferTrait>),
    //ImageView(Arc<ImageView>),
}

impl Eq for CommandBufferReferencedResource {}

impl CommandBufferReferencedResource {
    #[inline]
    pub fn hash(&self) -> u128 {
        match self {
            Self::ComputePipeline(l0) => l0.native_handle() as u128,
            Self::DescriptorSet(l0) => (0b0001u128 << 124u128) | (l0.native_handle() as u128),
            Self::PipelineLayout(l0) => (0b0010u128 << 124u128) | (l0.native_handle() as u128),
            Self::Image(l0) => (0b0011u128 << 124u128) | (l0.native_handle() as u128),
            Self::Framebuffer(l0) => (0b0100u128 << 124u128) | (l0.native_handle() as u128),
            Self::GraphicsPipeline(l0) => (0b0101u128 << 124u128) | (l0.native_handle() as u128),
            Self::RaytracingPipeline(l0) => (0b0110u128 << 124u128) | (l0.native_handle() as u128),
            Self::Buffer(l0) => (0b0111u128 << 124u128) | (l0.native_handle() as u128),
            //Self::ImageView(l0) => (0b0111u128 << 124u128) | (l0.native_handle() as u128),
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
            (Self::RaytracingPipeline(l0), Self::RaytracingPipeline(r0)) => {
                l0.native_handle() == r0.native_handle()
            }
            (Self::Framebuffer(l0), Self::Framebuffer(r0)) => {
                l0.native_handle() == r0.native_handle()
            }
            (Self::Image(l0), Self::Image(r0)) => l0.native_handle() == r0.native_handle(),
            (Self::Buffer(l0), Self::Buffer(r0)) => l0.native_handle() == r0.native_handle(),
            //(Self::ImageView(l0), Self::Image(r0)) => l0.native_handle() == r0.native_handle(),
            _ => false,
        }
    }
}

impl Hash for CommandBufferReferencedResource {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u128(self.hash())
    }
}

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

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct MemoryAccess(ash::vk::AccessFlags2);

impl From<&[MemoryAccessAs]> for MemoryAccess {
    fn from(value: &[MemoryAccessAs]) -> Self {
        let mut flags = ash::vk::AccessFlags2::empty();
        for v in value.iter() {
            flags |= v.to_owned().into();
        }

        Self(flags)
    }
}

impl From<ash::vk::AccessFlags2> for MemoryAccess {
    fn from(value: ash::vk::AccessFlags2) -> Self {
        Self(value)
    }
}

impl From<MemoryAccess> for ash::vk::AccessFlags2 {
    fn from(val: MemoryAccess) -> Self {
        val.0
    }
}

pub struct ImageMemoryBarrier {
    image: Arc<dyn ImageTrait>,
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
    pub fn image(&self) -> Arc<dyn ImageTrait> {
        self.image.clone()
    }

    #[inline]
    pub fn new(
        image: Arc<dyn ImageTrait>,
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
            image,
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

impl<'a> From<ImageMemoryBarrier> for ash::vk::ImageMemoryBarrier2<'a> {
    fn from(val: ImageMemoryBarrier) -> Self {
        ash::vk::ImageMemoryBarrier2::default()
            .image(ash::vk::Image::from_raw(val.image.native_handle()))
            .src_access_mask(val.src_access.into())
            .src_stage_mask(val.src_stages.into())
            .dst_access_mask(val.dst_access.into())
            .dst_stage_mask(val.dst_stages.into())
            .old_layout(val.old_layout.ash_layout())
            .new_layout(val.new_layout.ash_layout())
            .src_queue_family_index(val.src_queue_family.get_family_index())
            .dst_queue_family_index(val.dst_queue_family.get_family_index())
            .subresource_range(val.subresource_range.ash_subresource_range())
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ColorClearValues {
    Vec4(f32, f32, f32, f32),
    IVec4(i32, i32, i32, i32),
    UVec4(u32, u32, u32, u32),
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ClearValues {
    color: Option<ColorClearValues>,
}

impl ClearValues {
    pub fn new(color: Option<ColorClearValues>) -> Self {
        Self { color }
    }

    pub(crate) fn ash_clear(&self) -> ash::vk::ClearValue {
        let mut result = ash::vk::ClearValue::default();

        if let Some(color) = self.color {
            match color {
                ColorClearValues::Vec4(r, g, b, a) => {
                    result.color.float32 = [r, g, b, a];
                }
                ColorClearValues::IVec4(r, g, b, a) => {
                    result.color.int32 = [r, g, b, a];
                }
                ColorClearValues::UVec4(r, g, b, a) => {
                    result.color.uint32 = [r, g, b, a];
                }
            }
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
    /*
     * This function instructs the GPU to build the Top Level Acceleration Structure.
     *
     * Before calling this function the TLAS instance buffer MUST be filled with references
     * to instances of Bottom Level Acceleration Structure(s) that the user wants to include in the
     * acceleration structure to be built.
     *
     * @param tlas Top Level Acceleration Structure to build
     * @param primitive_offset the number of consecutives BLAS instances to skip (on the instance buffer)
     * @param primitive_count the number of BLAS instances to include on the TLAS
     */
    pub fn build_tlas(
        &mut self,
        tlas: Arc<TopLevelAccelerationStructure>,
        primitive_offset: u32,
        primitive_count: u32,
    ) {
        assert!(tlas.allowed_building_devices() != AllowedBuildingDevice::HostOnly);

        let tlas_max_instances = tlas.max_instances() as u64;
        let selected_instances_max_index =
            (primitive_offset.to_owned() as u64) + (primitive_count.to_owned() as u64);
        assert!(tlas_max_instances >= selected_instances_max_index);

        let (geometries, range_infos) = tlas
            .ash_build_info(primitive_offset, primitive_count)
            .unwrap();

        let ranges_collection: smallvec::SmallVec<
            [&[ash::vk::AccelerationStructureBuildRangeInfoKHR]; 1],
        > = range_infos.iter().map(|r| r.as_slice()).collect();

        // See https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetAccelerationStructureBuildSizesKHR.html
        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(geometries.as_slice())
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .mode(ash::vk::BuildAccelerationStructureModeKHR::BUILD)
            .src_acceleration_structure(ash::vk::AccelerationStructureKHR::null())
            .dst_acceleration_structure(tlas.ash_handle())
            .scratch_data(tlas.device_build_scratch_buffer().addr());

        // check if ray_tracing extension is enabled
        let Some(rt_ext) = self.device.ash_ext_acceleration_structure_khr() else {
            panic!("Ray tracing pipeline is not enabled!");
        };

        unsafe {
            rt_ext.cmd_build_acceleration_structures(
                self.command_buffer.ash_handle(),
                &[geometry_info],
                ranges_collection.as_slice(),
            )
        }
    }

    /*
     * This function instructs the GPU to build the Bottom Level Acceleration Structure.
     *
     * Before calling this function BLAS buffer(s) MUST be filled, this includes:
     *   - transform buffer: TODO
     *   - index_buffer: the list of index to vertices stored in the vertex_buffer
     *   - vertex_buffer: the list of vertices that are referenced from the index_buffer
     *
     * @param blas Bottom Level Acceleration Structure to build
     * @param primitive_offset the number of consecutives BLAS instances to skip (on the instance buffer)
     * @param primitive_count the number of BLAS instances to include on the TLAS
     */
    pub fn build_blas(
        &mut self,
        blas: Arc<BottomLevelAccelerationStructure>,
        primitive_offset: u32,
        primitive_count: u32,
        first_vertex: u32,
        transform_offset: u32,
    ) {
        assert!(blas.allowed_building_devices() != AllowedBuildingDevice::HostOnly);

        // TODO: assert from same device

        let (geometries, range_infos) = blas
            .ash_build_info(
                primitive_offset,
                primitive_count,
                first_vertex,
                transform_offset,
            )
            .unwrap();

        let ranges_collection: Vec<&[ash::vk::AccelerationStructureBuildRangeInfoKHR]> =
            range_infos.iter().map(|r| r.as_slice()).collect();

        // From vulkan specs: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetAccelerationStructureBuildSizesKHR.html
        // The srcAccelerationStructure, dstAccelerationStructure, and mode members of pBuildInfo are ignored.
        // Any VkDeviceOrHostAddressKHR members of pBuildInfo are ignored by this command,
        // except that the hostAddress member of VkAccelerationStructureGeometryTrianglesDataKHR::transformData
        // will be examined to check if it is NULL.
        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(geometries.as_slice())
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .ty(ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .mode(ash::vk::BuildAccelerationStructureModeKHR::BUILD)
            .src_acceleration_structure(ash::vk::AccelerationStructureKHR::null())
            .dst_acceleration_structure(blas.ash_handle())
            .scratch_data(blas.device_build_scratch_buffer().addr());

        // check if ray_tracing extension is enabled
        let Some(rt_ext) = self.device.ash_ext_acceleration_structure_khr() else {
            panic!("Ray tracing pipeline is not enabled!");
        };

        unsafe {
            rt_ext.cmd_build_acceleration_structures(
                self.command_buffer.ash_handle(),
                &[geometry_info],
                ranges_collection.as_slice(),
            )
        }
    }

    pub fn trace_rays(
        &mut self,
        binding_tables: Arc<RaytracingBindingTables>,
        dimensions: Image3DDimensions,
    ) {
        // check if ray_tracing extension is enabled
        match self.device.ash_ext_raytracing_pipeline_khr() {
            Some(rt_ext) => {
                let raygen_shader_binding_tables = binding_tables.ash_raygen_strided();
                let miss_shader_binding_tables = binding_tables.ash_miss_strided();
                let hit_shader_binding_tables = binding_tables.ash_closesthit_strided();
                let callable_shader_binding_tables = binding_tables.ash_callable_strided();

                unsafe {
                    rt_ext.cmd_trace_rays(
                        self.command_buffer.ash_handle(),
                        &raygen_shader_binding_tables,
                        &miss_shader_binding_tables,
                        &hit_shader_binding_tables,
                        &callable_shader_binding_tables,
                        dimensions.width(),
                        dimensions.height(),
                        dimensions.depth(),
                    )
                }
            }
            None => {
                println!("Ray tracing pipeline is not enabled, nothing will happend.");
            }
        }
    }

    pub fn bind_ray_tracing_pipeline(&mut self, raytracing_pipeline: Arc<RaytracingPipeline>) {
        unsafe {
            self.device.ash_handle().cmd_bind_pipeline(
                self.command_buffer.ash_handle(),
                ash::vk::PipelineBindPoint::RAY_TRACING_KHR,
                raytracing_pipeline.ash_handle(),
            )
        }

        self.used_resources
            .insert(CommandBufferReferencedResource::RaytracingPipeline(
                raytracing_pipeline,
            ));
    }

    pub fn bind_descriptor_sets_for_ray_tracing_pipeline(
        &mut self,
        pipeline_layout: Arc<PipelineLayout>,
        offset: u32,
        descriptor_sets: &[Arc<DescriptorSet>],
    ) {
        // TODO: check if ray_tracing extension is enabled

        let mut sets = Vec::<ash::vk::DescriptorSet>::new();

        for ds in descriptor_sets.iter() {
            self.used_resources
                .insert(CommandBufferReferencedResource::DescriptorSet(ds.clone()));

            sets.push(ds.ash_handle());
        }

        unsafe {
            self.device.ash_handle().cmd_bind_descriptor_sets(
                self.command_buffer.ash_handle(),
                ash::vk::PipelineBindPoint::RAY_TRACING_KHR,
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

    pub fn bind_descriptor_sets_for_graphics_pipeline(
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
                ash::vk::PipelineBindPoint::GRAPHICS,
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

    pub fn bind_graphics_pipeline(
        &mut self,
        graphics_pipeline: Arc<GraphicsPipeline>,
        viewport: Option<Viewport>,
        scissor: Option<Scissor>,
    ) {
        unsafe {
            self.device.ash_handle().cmd_bind_pipeline(
                self.command_buffer.ash_handle(),
                ash::vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline.ash_handle(),
            );

            match viewport {
                Some(viewport) => {
                    assert!(graphics_pipeline.is_viewport_dynamic());

                    let viewports = [ash::vk::Viewport::default()
                        .x(viewport.top_left_x())
                        .y(viewport.top_left_y())
                        .width(viewport.width())
                        .height(viewport.height())
                        .min_depth(viewport.min_depth())
                        .max_depth(viewport.max_depth())];

                    self.device.ash_handle().cmd_set_viewport(
                        self.command_buffer.ash_handle(),
                        0,
                        viewports.as_slice(),
                    );
                }
                None => {
                    assert!(!graphics_pipeline.is_viewport_dynamic());
                }
            }

            match scissor {
                Some(scissor) => {
                    assert!(graphics_pipeline.is_scissor_dynamic());

                    let dimensions = scissor.dimensions();

                    let scissors = [ash::vk::Rect2D::default()
                        .offset(
                            Offset2D::default()
                                .x(scissor.offset_x())
                                .y(scissor.offset_y()),
                        )
                        .extent(
                            ash::vk::Extent2D::default()
                                .width(dimensions.width())
                                .height(dimensions.height()),
                        )];

                    self.device.ash_handle().cmd_set_scissor(
                        self.command_buffer.ash_handle(),
                        0,
                        scissors.as_slice(),
                    );
                }
                None => {
                    assert!(!graphics_pipeline.is_scissor_dynamic());
                }
            }
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

    pub fn begin_renderpass_with_imageless_framebuffer(
        &mut self,
        framebuffer: Arc<ImagelessFramebuffer>,
        imageviews: &[Arc<ImageView>],
        clear_values: &[ClearValues],
    ) {
        let ash_clear_values: smallvec::SmallVec<[ash::vk::ClearValue; 32]> =
            clear_values.iter().map(|cv| cv.ash_clear()).collect();

        let ash_imageviews/*: smallvec::SmallVec<[ash::vk::ImageView; 32]>*/ = imageviews
            .iter()
            .map(|iv: &Arc<ImageView>| iv.ash_handle())
            .collect::<Vec<ash::vk::ImageView>>();

        let mut attachment_begin_info = ash::vk::RenderPassAttachmentBeginInfo::default()
            .attachments(ash_imageviews.as_slice());

        let render_pass_begin_info = ash::vk::RenderPassBeginInfo::default()
            .push_next(&mut attachment_begin_info)
            .clear_values(ash_clear_values.as_slice())
            .framebuffer(ash::vk::Framebuffer::from_raw(framebuffer.native_handle()))
            .render_pass(framebuffer.get_parent_renderpass().ash_handle())
            .render_area(
                ash::vk::Rect2D::default()
                    .offset(ash::vk::Offset2D::default().x(0).y(0))
                    .extent(
                        ash::vk::Extent2D::default()
                            .width(framebuffer.dimensions().width())
                            .height(framebuffer.dimensions().height()),
                    ),
            );

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

        let render_pass_begin_info = ash::vk::RenderPassBeginInfo::default()
            .clear_values(ash_clear_values.as_slice())
            .framebuffer(ash::vk::Framebuffer::from_raw(framebuffer.native_handle()))
            .render_pass(framebuffer.get_parent_renderpass().ash_handle())
            .render_area(
                ash::vk::Rect2D::default()
                    .offset(ash::vk::Offset2D::default().x(0).y(0))
                    .extent(
                        ash::vk::Extent2D::default()
                            .width(framebuffer.dimensions().width())
                            .height(framebuffer.dimensions().height()),
                    ),
            );

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

    pub fn copy_buffer_to_image(
        &mut self,
        src: Arc<dyn BufferTrait>,
        //src_subresource: ImageSubresourceLayers,
        dst_layout: ImageLayout,
        dst_subresource: ImageSubresourceLayers,
        dst: Arc<dyn ImageTrait>,
        extent: ImageDimensions,
    ) {
        let dst_offset = ash::vk::Offset3D::default().x(0).y(0).z(0);

        let regions = ash::vk::BufferImageCopy::default()
            .buffer_offset(0u64)
            .image_offset(dst_offset)
            .image_extent(extent.ash_extent_3d())
            .image_subresource(dst_subresource.ash_subresource_layers());

        self.used_resources
            .insert(CommandBufferReferencedResource::Buffer(src.clone()));

        self.used_resources
            .insert(CommandBufferReferencedResource::Image(dst.clone()));

        unsafe {
            self.device.ash_handle().cmd_copy_buffer_to_image(
                self.command_buffer.ash_handle(),
                ash::vk::Buffer::from_raw(src.native_handle()),
                ash::vk::Image::from_raw(dst.native_handle()),
                dst_layout.ash_layout(),
                &[regions],
            );
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
        let src_offset = ash::vk::Offset3D::default().x(0).y(0).z(0);

        let dst_offset = ash::vk::Offset3D::default().x(0).y(0).z(0);

        let regions = ash::vk::ImageCopy::default()
            .extent(extent.ash_extent_3d())
            .dst_subresource(dst_subresource.ash_subresource_layers())
            .src_subresource(src_subresource.ash_subresource_layers())
            .dst_offset(dst_offset)
            .src_offset(src_offset);

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

    pub fn image_barrier(&mut self, image_mem_barrier: ImageMemoryBarrier) {
        // TODO: check every resource is from the same device

        self.used_resources
            .insert(CommandBufferReferencedResource::Image(
                image_mem_barrier.image(),
            ));

        let image_memory_barriers: [ash::vk::ImageMemoryBarrier2; 1] = [image_mem_barrier.into()];

        let dependency_info = ash::vk::DependencyInfo::default()
            .image_memory_barriers(image_memory_barriers.as_slice());

        unsafe {
            self.device
                .ash_handle()
                .cmd_pipeline_barrier2(self.command_buffer.ash_handle(), &dependency_info);
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

    pub fn bind_descriptor_sets_for_compute_pipeline(
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
        F: FnOnce(&mut CommandBufferRecorder) + Sized,
    {
        let device = self
            .get_parent_command_pool()
            .get_parent_queue_family()
            .get_parent_device();

        #[cfg(feature = "better_mutex")]
        let mut resources_lck = self.resources_in_use.lock();

        #[cfg(not(feature = "better_mutex"))]
        let mut resources_lck = match self.resources_in_use.lock() {
            Ok(lock) => lock,
            Err(err) => {
                return Err(VulkanError::Framework(FrameworkError::MutexError(format!(
                    "{err}"
                ))))
            }
        };

        let begin_info = ash::vk::CommandBufferBeginInfo::default()
            .flags(ash::vk::CommandBufferUsageFlags::empty() /*ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT*/);

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
                    Err(err) => Err(VulkanError::Vulkan(
                        err.as_raw(),
                        Some(format!("Error updating the command buffer: {}", err)),
                    )),
                }
            }
            Err(err) =>
            // the command buffer is in the previous state... A good one unless the error is DEVICE_LOST
            {
                Err(VulkanError::Vulkan(
                    err.as_raw(),
                    Some(format!(
                        "Error opening the command buffer for writing: {}",
                        err
                    )),
                ))
            }
        }
    }

    pub fn new(
        command_pool: Arc<CommandPool>,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = command_pool.get_parent_queue_family().get_parent_device();

        let create_info = ash::vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool.ash_handle())
            .level(ash::vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        match unsafe { device.ash_handle().allocate_command_buffers(&create_info) } {
            Ok(command_buffers) => {
                let command_buffer = command_buffers[0];

                let mut obj_name_bytes = vec![];
                if let Some(ext) = device.ash_ext_debug_utils_ext() {
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
                            let dbg_info = ash::vk::DebugUtilsObjectNameInfoEXT::default()
                                .object_handle(command_buffer)
                                .object_name(object_name);

                            if let Err(err) = ext.set_debug_utils_object_name(&dbg_info) {
                                #[cfg(debug_assertions)]
                                {
                                    println!("Error setting the Debug name for the newly created Command Pool, will use handle. Error: {}", err)
                                }
                            }
                        }
                    }
                }

                #[cfg(feature = "better_mutex")]
                let resources_in_use = const_mutex(HashSet::new());

                #[cfg(not(feature = "better_mutex"))]
                let resources_in_use: Mutex<
                    HashSet<CommandBufferReferencedResource>,
                > = Mutex::new(HashSet::new());

                Ok(Arc::new(Self {
                    command_buffer,
                    command_pool,
                    resources_in_use,
                }))
            }
            Err(err) => Err(VulkanError::Vulkan(
                err.as_raw(),
                Some(format!("Error creating the command buffer: {}", err)),
            )),
        }
    }
}
