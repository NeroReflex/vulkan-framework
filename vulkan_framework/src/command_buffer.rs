use std::{
    collections::HashSet,
    hash::Hash,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};

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
    clear_values::{ColorClearValues, DepthClearValues},
    command_pool::{CommandPool, CommandPoolOwned},
    device::DeviceOwned,
    dynamic_rendering::{
        DynamicRenderingColorAttachment, DynamicRenderingDepthAttachment,
        DynamicRenderingStencilAttachment,
    },
    graphics_pipeline::{GraphicsPipeline, IndexType, Scissor, Viewport},
    image::{
        Image1DTrait, Image2DDimensions, Image2DTrait, Image3DDimensions, Image3DTrait,
        ImageDimensions, ImageLayout, ImageSubresourceLayers, ImageSubresourceRange, ImageTrait,
    },
    image_view::ImageView,
    memory_barriers::{BufferMemoryBarrier, ImageMemoryBarrier, PipelineBarrier},
    pipeline_layout::PipelineLayout,
    prelude::{FrameworkError, VulkanError, VulkanResult},
    raytracing_pipeline::RaytracingPipeline,
    shader_stage_access::ShaderStagesAccess,
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
    Image(Arc<dyn ImageTrait>),
    Buffer(Arc<dyn BufferTrait>),
    ImageView(Arc<ImageView>),
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
            Self::GraphicsPipeline(l0) => (0b0101u128 << 124u128) | (l0.native_handle() as u128),
            Self::RaytracingPipeline(l0) => (0b0110u128 << 124u128) | (l0.native_handle() as u128),
            Self::Buffer(l0) => (0b0111u128 << 124u128) | (l0.native_handle() as u128),
            Self::ImageView(l0) => (0b0111u128 << 124u128) | (l0.native_handle() as u128),
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
            (Self::Image(l0), Self::Image(r0)) => l0.native_handle() == r0.native_handle(),
            (Self::Buffer(l0), Self::Buffer(r0)) => l0.native_handle() == r0.native_handle(),
            (Self::ImageView(l0), Self::Image(r0)) => l0.native_handle() == r0.native_handle(),
            _ => false,
        }
    }
}

impl Hash for CommandBufferReferencedResource {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u128(self.hash())
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

        let (geometries, range_infos) = tlas
            .ash_build_info(primitive_offset, primitive_count)
            .unwrap();

        assert!(!geometries.is_empty());
        assert!(!range_infos.is_empty());

        let ranges_collection: smallvec::SmallVec<
            [&[ash::vk::AccelerationStructureBuildRangeInfoKHR]; 1],
        > = range_infos.iter().map(|r| r.as_slice()).collect();

        assert!(!ranges_collection.is_empty());

        // See https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetAccelerationStructureBuildSizesKHR.html
        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(geometries.as_slice())
            .flags(tlas.build_flags())
            .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .mode(ash::vk::BuildAccelerationStructureModeKHR::BUILD)
            .src_acceleration_structure(ash::vk::AccelerationStructureKHR::null())
            .dst_acceleration_structure(tlas.ash_handle())
            .scratch_data(tlas.device_build_scratch_buffer().addr());

        // check if ray_tracing extension is enabled
        let Some(rt_ext) = self.device.ash_ext_acceleration_structure_khr() else {
            panic!("Ray tracing pipeline is not enabled!");
        };

        assert_eq!(ranges_collection.len(), 1_usize);

        unsafe {
            rt_ext.cmd_build_acceleration_structures(
                self.command_buffer.ash_handle(),
                [geometry_info].as_slice(),
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

    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            self.device.ash_handle().cmd_draw_indexed(
                self.command_buffer.ash_handle(),
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
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

    /// Implemented using dynamic rendering: roughly equivalent to using a renderpass,
    /// except ImageViews are used directly and no framebuffer is required.
    ///
    /// You are free to use less attachments than what the graphics pipeline supports,
    /// and/or exclude the depth or stencil buffer.
    ///
    /// Color attachments will be transitioned to `ImageLayout::ColorAttachmentOptimal`
    /// and depth/stencil attachments will be transitioned to `ImageLayout::DepthStencilAttachmentOptimal`.
    pub fn graphics_rendering<T>(
        &mut self,
        render_extent: Image2DDimensions,
        color_attachments: &[DynamicRenderingColorAttachment],
        depth_attachment: Option<&DynamicRenderingDepthAttachment>,
        stencil_attachment: Option<&DynamicRenderingStencilAttachment>,
        fun: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let mut ash_color_attachments: smallvec::SmallVec<[ash::vk::RenderingAttachmentInfo; 8]> =
            smallvec::smallvec![];
        for attachment in color_attachments.iter().cloned() {
            // TODO: check for dimensions to fit into the render_extent

            self.used_resources
                .insert(CommandBufferReferencedResource::ImageView(
                    attachment.image_view().clone(),
                ));

            ash_color_attachments.push(attachment.into());
        }

        let render_area = ash::vk::Rect2D::default()
            .offset(ash::vk::Offset2D::default().x(0).y(0))
            .extent(
                ash::vk::Extent2D::default()
                    .width(render_extent.width())
                    .height(render_extent.height()),
            );

        let mut render_info = ash::vk::RenderingInfo::default()
            .color_attachments(ash_color_attachments.as_slice())
            .layer_count(1)
            .render_area(render_area);

        let mut d_attachment = ash::vk::RenderingAttachmentInfo::default();
        render_info = match depth_attachment {
            Some(attachment) => {
                // TODO: check for dimensions to fit into the render_extent

                self.used_resources
                    .insert(CommandBufferReferencedResource::ImageView(
                        attachment.image_view().clone(),
                    ));
                d_attachment = attachment.clone().into();
                render_info.depth_attachment(&d_attachment)
            }
            None => render_info,
        };

        let mut s_attachment = ash::vk::RenderingAttachmentInfo::default();
        render_info = match stencil_attachment {
            Some(attachment) => {
                // TODO: check for dimensions to fit into the render_extent

                self.used_resources
                    .insert(CommandBufferReferencedResource::ImageView(
                        attachment.image_view().clone(),
                    ));
                s_attachment = attachment.clone().into();
                render_info.stencil_attachment(&s_attachment)
            }
            None => render_info,
        };

        unsafe {
            self.device
                .ash_handle()
                .cmd_begin_rendering(self.command_buffer.ash_handle(), &render_info)
        }

        let result = fun(self);

        unsafe {
            self.device
                .ash_handle()
                .cmd_end_rendering(self.command_buffer.ash_handle())
        }

        result
    }

    pub fn update_buffer<T>(&mut self, dst: Arc<dyn BufferTrait>, offset: u64, src: &[T])
    where
        T: Sized,
    {
        self.used_resources
            .insert(CommandBufferReferencedResource::Buffer(dst.clone()));

        let bytes = std::mem::size_of_val(src);
        if bytes as u64 % 4 != 0 {
            panic!("Size not multiple of 4 given!");
        }

        if (offset % 4) != 0 {
            panic!("Not aligned offset given!");
        }

        let ptr = src.as_ptr() as *const std::ffi::c_void;
        if ptr as u64 % 4 != 0 {
            panic!("Unaligned pointer given!");
        }

        unsafe {
            self.device.ash_handle().cmd_update_buffer(
                self.command_buffer.ash_handle(),
                ash::vk::Buffer::from_raw(dst.native_handle()),
                offset,
                std::slice::from_raw_parts(ptr as *const u8, bytes),
            )
        };
    }

    pub fn copy_buffer(
        &mut self,
        src: Arc<dyn BufferTrait>,
        dst: Arc<dyn BufferTrait>,
        regions: &[(u64, u64, u64)],
    ) {
        self.used_resources
            .insert(CommandBufferReferencedResource::Buffer(src.clone()));

        self.used_resources
            .insert(CommandBufferReferencedResource::Buffer(dst.clone()));

        let regions: smallvec::SmallVec<[ash::vk::BufferCopy2; 4]> = regions
            .iter()
            .map(|(src_offset, dst_offset, size)| {
                ash::vk::BufferCopy2::default()
                    .src_offset(src_offset.to_owned())
                    .dst_offset(dst_offset.to_owned())
                    .size(size.to_owned())
            })
            .collect();

        let copy_info = ash::vk::CopyBufferInfo2::default()
            .src_buffer(ash::vk::Buffer::from_raw(src.native_handle()))
            .dst_buffer(ash::vk::Buffer::from_raw(dst.native_handle()))
            .regions(regions.as_slice());

        unsafe {
            self.device
                .ash_handle()
                .cmd_copy_buffer2(self.command_buffer.ash_handle(), &copy_info);
        }
    }

    pub fn copy_buffer_to_image(
        &mut self,
        src: Arc<dyn BufferTrait>,
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
                dst_layout.into(),
                &[regions],
            );
        }
    }

    /// Place a command to clear the specified (color) image and returns the `ImageSubresourceRange`
    /// that has to be used for image barriers.
    ///
    /// WARNING: The image layout MUST be VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    /// at the time of the transfer operation
    pub fn clear_color_image(&mut self, value: ColorClearValues, image_srr: ImageSubresourceRange) {
        let aspects: crate::ash::vk::ImageAspectFlags = image_srr.aspects().clone().into();
        assert!(aspects.contains(ash::vk::ImageAspectFlags::COLOR));

        self.used_resources
            .insert(CommandBufferReferencedResource::Image(image_srr.image()));

        let clear_color = value.into();
        unsafe {
            self.device.ash_handle().cmd_clear_color_image(
                self.command_buffer.ash_handle(),
                ash::vk::Image::from_raw(image_srr.image().native_handle()),
                crate::ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_color,
                &[image_srr.clone().into()],
            );
        }
    }

    /// Place a command to clear the specified (depth) image and returns the `ImageSubresourceRange`
    /// that has to be used for image barriers.
    ///
    /// WARNING: The image layout MUST be VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    /// at the time of the transfer operation
    pub fn clear_depth_image(&mut self, value: DepthClearValues, image_srr: ImageSubresourceRange) {
        let aspects: crate::ash::vk::ImageAspectFlags = image_srr.aspects().clone().into();
        assert!(aspects.contains(ash::vk::ImageAspectFlags::DEPTH));

        self.used_resources
            .insert(CommandBufferReferencedResource::Image(image_srr.image()));

        let clear_color = value.into();
        unsafe {
            self.device.ash_handle().cmd_clear_depth_stencil_image(
                self.command_buffer.ash_handle(),
                ash::vk::Image::from_raw(image_srr.image().native_handle()),
                crate::ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_color,
                &[image_srr.clone().into()],
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
                src_layout.into(),
                ash::vk::Image::from_raw(dst.native_handle()),
                dst_layout.into(),
                &[regions],
            );
        }
    }

    /// Place a pipeline buffer into the command buffer.
    pub fn pipeline_barriers(&mut self, barriers: impl IntoIterator<Item = PipelineBarrier>) {
        // TODO: check every resource is from the same device

        let mut image_memory_barriers =
            smallvec::SmallVec::<[ash::vk::ImageMemoryBarrier2; 8]>::new();
        let mut buffer_memory_barriers =
            smallvec::SmallVec::<[ash::vk::BufferMemoryBarrier2; 4]>::new();
        let mut memory_barriers = smallvec::SmallVec::<[ash::vk::MemoryBarrier2; 2]>::new();
        for barrier in barriers.into_iter() {
            match barrier {
                PipelineBarrier::Buffer(buffer_memory_barrier) => {
                    assert_eq!(
                        self.device.native_handle(),
                        buffer_memory_barrier
                            .subresource_range()
                            .buffer()
                            .get_parent_device()
                            .native_handle()
                    );

                    self.used_resources
                        .insert(CommandBufferReferencedResource::Buffer(
                            buffer_memory_barrier.subresource_range().buffer(),
                        ));

                    buffer_memory_barriers.push(buffer_memory_barrier.into())
                }
                PipelineBarrier::Image(image_mem_barrier) => {
                    assert_eq!(
                        self.device.native_handle(),
                        image_mem_barrier
                            .subresource_range()
                            .image()
                            .get_parent_device()
                            .native_handle()
                    );

                    self.used_resources
                        .insert(CommandBufferReferencedResource::Image(
                            image_mem_barrier.subresource_range().image(),
                        ));
                    image_memory_barriers.push(image_mem_barrier.into());
                }
                PipelineBarrier::Global(memory_barrier) => {
                    memory_barriers.push(memory_barrier.into());
                }
            }
        }

        let dependency_info = ash::vk::DependencyInfo::default()
            .memory_barriers(memory_barriers.as_slice())
            .buffer_memory_barriers(buffer_memory_barriers.as_slice())
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

    pub fn push_constant(
        &mut self,
        pipeline_layout: Arc<PipelineLayout>,
        access: ShaderStagesAccess,
        offset: u32,
        data: &[u8],
    ) {
        unsafe {
            self.device.ash_handle().cmd_push_constants(
                self.command_buffer.ash_handle(),
                pipeline_layout.ash_handle(),
                access.into(),
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

    pub fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        vertex_buffers: &[(u64, Arc<dyn BufferTrait>)],
    ) {
        let mut buffers: smallvec::SmallVec<[ash::vk::Buffer; 8]> = smallvec::smallvec![];
        let mut offsets: smallvec::SmallVec<[ash::vk::DeviceSize; 8]> = smallvec::smallvec![];
        for (offset, vb) in vertex_buffers.iter() {
            self.used_resources
                .insert(CommandBufferReferencedResource::Buffer(vb.clone()));

            offsets.push(offset.to_owned() as ash::vk::DeviceSize);

            buffers.push(ash::vk::Buffer::from_raw(vb.native_handle()));
        }

        unsafe {
            self.device.ash_handle().cmd_bind_vertex_buffers(
                self.command_buffer.ash_handle(),
                first_binding,
                buffers.as_slice(),
                offsets.as_slice(),
            )
        }
    }

    pub fn bind_index_buffer(
        &mut self,
        offset: u64,
        index_buffer: Arc<dyn BufferTrait>,
        index_type: IndexType,
    ) {
        self.used_resources
            .insert(CommandBufferReferencedResource::Buffer(
                index_buffer.clone(),
            ));

        unsafe {
            self.device.ash_handle().cmd_bind_index_buffer(
                self.command_buffer.ash_handle(),
                ash::vk::Buffer::from_raw(index_buffer.native_handle()),
                offset,
                index_type.into(),
            )
        }
    }
}

pub(crate) trait SubmittableCommandBufferTrait {
    fn mark_execution_begin(&self) -> VulkanResult<()>;

    fn mark_execution_complete(&self) -> VulkanResult<()>;
}

pub trait CommandBufferTrait: SubmittableCommandBufferTrait + CommandPoolOwned {
    fn native_handle(&self) -> u64;
}

pub(crate) trait CommandBufferCrateTrait: CommandBufferTrait {
    fn ash_handle(&self) -> ash::vk::CommandBuffer;
}

const PRIMARY_COMMAND_BUFFER_STATUS_NO_COMMANDS: u32 = 0;
const PRIMARY_COMMAND_BUFFER_STATUS_RECORDING: u32 = 1;
const PRIMARY_COMMAND_BUFFER_STATUS_READY: u32 = 2;
const PRIMARY_COMMAND_BUFFER_STATUS_READY_ONE_TIME: u32 = 3;
const PRIMARY_COMMAND_BUFFER_STATUS_RUNNING: u32 = 4;
const PRIMARY_COMMAND_BUFFER_STATUS_RUNNING_ONE_TIME: u32 = 5;

pub struct PrimaryCommandBuffer {
    command_pool: Arc<CommandPool>,
    command_buffer: ash::vk::CommandBuffer,
    processing: AtomicU32,
    resources_in_use: Mutex<HashSet<CommandBufferReferencedResource>>,
}

impl Drop for PrimaryCommandBuffer {
    fn drop(&mut self) {
        // Command buffers will be automatically freed when their command pool is destroyed,
        // so we don't need any explicit cleanup here.
    }
}

impl CommandPoolOwned for PrimaryCommandBuffer {
    fn get_parent_command_pool(&self) -> Arc<CommandPool> {
        self.command_pool.clone()
    }
}

impl SubmittableCommandBufferTrait for PrimaryCommandBuffer {
    fn mark_execution_begin(&self) -> VulkanResult<()> {
        let cb_status: u32 = self.processing.fetch_min(u32::MAX, Ordering::SeqCst);
        let new_status = match cb_status {
            PRIMARY_COMMAND_BUFFER_STATUS_NO_COMMANDS => {
                return Err(VulkanError::Framework(
                    FrameworkError::CommandBufferSubmitNoCommands,
                ))
            }
            PRIMARY_COMMAND_BUFFER_STATUS_RECORDING => {
                return Err(VulkanError::Framework(
                    FrameworkError::CommandBufferSubmitRecording,
                ))
            }
            PRIMARY_COMMAND_BUFFER_STATUS_READY => PRIMARY_COMMAND_BUFFER_STATUS_RUNNING,
            PRIMARY_COMMAND_BUFFER_STATUS_READY_ONE_TIME => {
                PRIMARY_COMMAND_BUFFER_STATUS_RUNNING_ONE_TIME
            }
            PRIMARY_COMMAND_BUFFER_STATUS_RUNNING => {
                return Err(VulkanError::Framework(
                    FrameworkError::CommandBufferSubmitAlreadyRunning,
                ))
            }
            PRIMARY_COMMAND_BUFFER_STATUS_RUNNING_ONE_TIME => {
                return Err(VulkanError::Framework(
                    FrameworkError::CommandBufferSubmitAlreadyRunning,
                ))
            }
            _ => {
                return Err(VulkanError::Framework(
                    FrameworkError::CommandBufferInvalidState,
                ))
            }
        };

        self.processing
            .compare_exchange(cb_status, new_status, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|cb_status| match cb_status {
                PRIMARY_COMMAND_BUFFER_STATUS_NO_COMMANDS => {
                    VulkanError::Framework(FrameworkError::CommandBufferSubmitNoCommands)
                }
                PRIMARY_COMMAND_BUFFER_STATUS_RECORDING => {
                    VulkanError::Framework(FrameworkError::CommandBufferSubmitRecording)
                }
                PRIMARY_COMMAND_BUFFER_STATUS_READY => {
                    VulkanError::Framework(FrameworkError::CommandBufferInternalError(cb_status))
                }
                PRIMARY_COMMAND_BUFFER_STATUS_READY_ONE_TIME => {
                    VulkanError::Framework(FrameworkError::CommandBufferInternalError(cb_status))
                }
                PRIMARY_COMMAND_BUFFER_STATUS_RUNNING => {
                    VulkanError::Framework(FrameworkError::CommandBufferSubmitAlreadyRunning)
                }
                PRIMARY_COMMAND_BUFFER_STATUS_RUNNING_ONE_TIME => {
                    VulkanError::Framework(FrameworkError::CommandBufferSubmitAlreadyRunning)
                }
                _ => VulkanError::Framework(FrameworkError::CommandBufferInvalidState),
            })?;

        Ok(())
    }

    fn mark_execution_complete(&self) -> VulkanResult<()> {
        let cb_status: u32 = self.processing.fetch_min(u32::MAX, Ordering::SeqCst);
        let new_status = match cb_status {
            PRIMARY_COMMAND_BUFFER_STATUS_NO_COMMANDS => {
                return Err(VulkanError::Framework(
                    FrameworkError::CommandBufferInternalError(cb_status),
                ))
            }
            PRIMARY_COMMAND_BUFFER_STATUS_RECORDING => {
                return Err(VulkanError::Framework(
                    FrameworkError::CommandBufferInternalError(cb_status),
                ))
            }
            PRIMARY_COMMAND_BUFFER_STATUS_READY => {
                return Err(VulkanError::Framework(
                    FrameworkError::CommandBufferInternalError(cb_status),
                ))
            }
            PRIMARY_COMMAND_BUFFER_STATUS_READY_ONE_TIME => {
                return Err(VulkanError::Framework(
                    FrameworkError::CommandBufferInternalError(cb_status),
                ))
            }
            PRIMARY_COMMAND_BUFFER_STATUS_RUNNING => PRIMARY_COMMAND_BUFFER_STATUS_READY,
            PRIMARY_COMMAND_BUFFER_STATUS_RUNNING_ONE_TIME => {
                match self.resources_in_use.lock() {
                    Ok(mut lock) => lock.clear(),
                    Err(err) => {
                        return Err(VulkanError::Framework(FrameworkError::MutexError(format!(
                            "{err}"
                        ))))
                    }
                }

                PRIMARY_COMMAND_BUFFER_STATUS_NO_COMMANDS
            }
            _ => {
                return Err(VulkanError::Framework(
                    FrameworkError::CommandBufferInvalidState,
                ))
            }
        };

        self.processing
            .compare_exchange(cb_status, new_status, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| VulkanError::Framework(FrameworkError::CommandBufferInvalidState))?;

        Ok(())
    }
}

impl CommandBufferTrait for PrimaryCommandBuffer {
    fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.command_buffer)
    }
}

impl CommandBufferCrateTrait for PrimaryCommandBuffer {
    fn ash_handle(&self) -> ash::vk::CommandBuffer {
        self.command_buffer
    }
}

impl PrimaryCommandBuffer {
    pub(crate) fn record_commands_raw<F, T>(
        &self,
        commands_writer_fn: F,
        flags: crate::ash::vk::CommandBufferUsageFlags,
    ) -> VulkanResult<T>
    where
        F: FnOnce(&mut CommandBufferRecorder) -> T + Sized,
    {
        let device = self
            .get_parent_command_pool()
            .get_parent_queue_family()
            .get_parent_device();

        #[cfg(feature = "better_mutex")]
        let mut resources_lck = self.resources_in_use.lock();

        let begin_info = ash::vk::CommandBufferBeginInfo::default().flags(flags);

        // transition to recording command so no other thread can record at the same time
        {
            let cb_status: u32 = self.processing.fetch_min(u32::MAX, Ordering::SeqCst);
            let new_status = match cb_status {
                PRIMARY_COMMAND_BUFFER_STATUS_NO_COMMANDS => {
                    PRIMARY_COMMAND_BUFFER_STATUS_RECORDING
                }
                PRIMARY_COMMAND_BUFFER_STATUS_RECORDING => {
                    return Err(VulkanError::Framework(
                        FrameworkError::CommandBufferRecordRecording,
                    ))
                }
                PRIMARY_COMMAND_BUFFER_STATUS_READY => PRIMARY_COMMAND_BUFFER_STATUS_RECORDING,
                PRIMARY_COMMAND_BUFFER_STATUS_READY_ONE_TIME => {
                    PRIMARY_COMMAND_BUFFER_STATUS_RECORDING
                }
                PRIMARY_COMMAND_BUFFER_STATUS_RUNNING => {
                    return Err(VulkanError::Framework(
                        FrameworkError::CommandBufferRecordRunning,
                    ))
                }
                PRIMARY_COMMAND_BUFFER_STATUS_RUNNING_ONE_TIME => {
                    return Err(VulkanError::Framework(
                        FrameworkError::CommandBufferRecordRunning,
                    ))
                }
                _ => {
                    return Err(VulkanError::Framework(
                        FrameworkError::CommandBufferInvalidState,
                    ))
                }
            };

            self.processing
                .compare_exchange(cb_status, new_status, Ordering::SeqCst, Ordering::SeqCst)
                .map_err(|_| VulkanError::Framework(FrameworkError::CommandBufferInvalidState))?;
        }

        #[cfg(not(feature = "better_mutex"))]
        let mut resources_lck = match self.resources_in_use.lock() {
            Ok(lock) => lock,
            Err(err) => {
                return Err(VulkanError::Framework(FrameworkError::MutexError(format!(
                    "{err}"
                ))))
            }
        };

        unsafe {
            device
                .ash_handle()
                .begin_command_buffer(self.ash_handle(), &begin_info)
        }?;

        let mut recorder = CommandBufferRecorder {
            device: device.clone(),
            command_buffer: self,
            used_resources: HashSet::new(),
        };

        let result = commands_writer_fn(&mut recorder);

        unsafe { device.ash_handle().end_command_buffer(self.ash_handle()) }?;
        *resources_lck = recorder.used_resources;
        drop(resources_lck);

        // transition to recording command so no other thread can record at the same time
        {
            let cb_status: u32 = self.processing.fetch_min(u32::MAX, Ordering::SeqCst);
            let new_status = match cb_status {
                PRIMARY_COMMAND_BUFFER_STATUS_RECORDING => {
                    match flags.contains(crate::ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT) {
                        true => PRIMARY_COMMAND_BUFFER_STATUS_READY_ONE_TIME,
                        false => PRIMARY_COMMAND_BUFFER_STATUS_READY,
                    }
                }
                _ => {
                    return Err(VulkanError::Framework(
                        FrameworkError::CommandBufferInvalidState,
                    ))
                }
            };

            self.processing
                .compare_exchange(cb_status, new_status, Ordering::SeqCst, Ordering::SeqCst)
                .map_err(|_| VulkanError::Framework(FrameworkError::CommandBufferInvalidState))?;
        }

        Ok(result)
    }

    pub fn record_one_time_submit<F, T>(&self, commands_writer_fn: F) -> VulkanResult<T>
    where
        F: FnOnce(&mut CommandBufferRecorder) -> T + Sized,
    {
        self.record_commands_raw(
            commands_writer_fn,
            crate::ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        )
    }

    pub fn record<F, T>(&self, commands_writer_fn: F) -> VulkanResult<T>
    where
        F: FnOnce(&mut CommandBufferRecorder) -> T + Sized,
    {
        self.record_commands_raw(
            commands_writer_fn,
            crate::ash::vk::CommandBufferUsageFlags::empty(),
        )
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

        let command_buffers =
            unsafe { device.ash_handle().allocate_command_buffers(&create_info) }?;
        let command_buffer = command_buffers[0];

        let mut obj_name_bytes = vec![];
        if let Some(ext) = device.ash_ext_debug_utils_ext() {
            if let Some(name) = debug_name {
                for name_ch in name.as_bytes().iter() {
                    obj_name_bytes.push(*name_ch);
                }
                obj_name_bytes.push(0x00);

                unsafe {
                    let object_name =
                        std::ffi::CStr::from_bytes_with_nul_unchecked(obj_name_bytes.as_slice());
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
        let resources_in_use = Mutex::new(HashSet::new());

        Ok(Arc::new(Self {
            command_buffer,
            command_pool,
            processing: AtomicU32::new(PRIMARY_COMMAND_BUFFER_STATUS_NO_COMMANDS),
            resources_in_use,
        }))
    }
}
