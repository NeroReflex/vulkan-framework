use std::sync::{Arc, Mutex};

use inline_spirv::*;

use crate::rendering::{
    MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingResult, rendering_dimensions::RenderingDimensions,
    resources::object::Manager,
};

use vulkan_framework::{
    command_buffer::{ClearValues, ColorClearValues, CommandBufferRecorder},
    device::Device,
    framebuffer::Framebuffer,
    graphics_pipeline::{
        AttributeType, CullMode, DepthCompareOp, DepthConfiguration, FrontFace, GraphicsPipeline,
        PolygonMode, Rasterizer, Scissor, VertexInputAttribute, VertexInputBinding,
        VertexInputRate, Viewport,
    },
    image::{
        AllocatedImage, CommonImageFormat, ConcreteImageDescriptor, Image, Image1DTrait,
        Image2DDimensions, Image2DTrait, ImageDimensions, ImageFlags, ImageFormat, ImageLayout,
        ImageMultisampling, ImageSubresourceRange, ImageTiling, ImageTrait, ImageUsage, ImageUseAs,
    },
    image_view::{ImageView, ImageViewType},
    memory_allocator::{DefaultAllocator, MemoryAllocator},
    memory_heap::{ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryRequirements, MemoryType},
    memory_pool::{MemoryPool, MemoryPoolFeatures},
    memory_requiring::MemoryRequiring,
    pipeline_layout::PipelineLayout,
    renderpass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, RenderPass,
        RenderSubPassDescription,
    },
    shaders::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
};

const FINAL_RENDERING_VERTEX_SPV: &[u32] = inline_spirv!(
    r#"
#version 450 core

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
"#,
    vert
);

const FINAL_RENDERING_FRAGMENT_SPV: &[u32] = inline_spirv!(
    r#"
#version 450 core

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(1.0, 0.0, 0.0, 1.0);
}
"#,
    frag
);

/// Represents the stage of the pipeline responsible for drawing static meshes.
pub struct MeshRendering {
    _memory_pool: Arc<MemoryPool>,
    image_dimensions: Image2DDimensions,
    gbuffer_depth_stencil_images:
        smallvec::SmallVec<[Arc<AllocatedImage>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    gbuffer_depth_stencil_image_views:
        smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    gbuffer_position_images:
        smallvec::SmallVec<[Arc<AllocatedImage>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    gbuffer_position_image_views:
        smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    gbuffer_normal_images:
        smallvec::SmallVec<[Arc<AllocatedImage>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    gbuffer_normal_image_views:
        smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    gbuffer_texture_images:
        smallvec::SmallVec<[Arc<AllocatedImage>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    gbuffer_texture_image_views:
        smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    framebuffers: smallvec::SmallVec<[Arc<Framebuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    graphics_pipeline: Arc<GraphicsPipeline>,
}

impl MeshRendering {
    fn output_image_color_layout() -> ImageLayout {
        ImageLayout::ShaderReadOnlyOptimal
    }

    fn output_image_depth_stencil_layout() -> ImageLayout {
        ImageLayout::DepthStencilReadOnlyOptimal
    }

    fn output_image_color_format() -> ImageFormat {
        ImageFormat::from(CommonImageFormat::r32g32b32a32_sfloat)
    }

    fn output_image_depth_stencil_format() -> ImageFormat {
        ImageFormat::from(CommonImageFormat::d32_sfloat_s8_uint)
    }

    pub fn new(
        device: Arc<Device>,
        render_area: &RenderingDimensions,
        frames_in_flight: u32,
    ) -> RenderingResult<Self> {
        let image_dimensions = render_area.into();

        let mut gbuffer_depth_stencil_image_handles: smallvec::SmallVec<
            [_; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        for index in 0..(frames_in_flight as usize) {
            let image_name = format!("mesh_rendering.gbuffer_depth_stencil_images[{index}]");
            let image = Image::new(
                device.clone(),
                ConcreteImageDescriptor::new(
                    ImageDimensions::Image2D {
                        extent: image_dimensions,
                    },
                    ImageUsage::from(
                        [ImageUseAs::Sampled, ImageUseAs::DepthStencilAttachment].as_slice(),
                    ),
                    ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    Self::output_image_depth_stencil_format(),
                    ImageFlags::empty(),
                    ImageTiling::Optimal,
                ),
                None,
                Some(image_name.as_str()),
            )?;

            gbuffer_depth_stencil_image_handles.push(image);
        }

        let mut gbuffer_position_image_handles: smallvec::SmallVec<
            [_; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        for index in 0..(frames_in_flight as usize) {
            let image_name = format!("mesh_rendering.gbuffer_position_images[{index}]");
            let image = Image::new(
                device.clone(),
                ConcreteImageDescriptor::new(
                    ImageDimensions::Image2D {
                        extent: image_dimensions,
                    },
                    ImageUsage::from([ImageUseAs::Sampled, ImageUseAs::ColorAttachment].as_slice()),
                    ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    Self::output_image_color_format(),
                    ImageFlags::empty(),
                    ImageTiling::Optimal,
                ),
                None,
                Some(image_name.as_str()),
            )?;

            gbuffer_position_image_handles.push(image);
        }

        let mut gbuffer_normal_image_handles: smallvec::SmallVec<
            [_; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        for index in 0..(frames_in_flight as usize) {
            let image_name = format!("mesh_rendering.gbuffer_normal_images[{index}]");
            let image = Image::new(
                device.clone(),
                ConcreteImageDescriptor::new(
                    ImageDimensions::Image2D {
                        extent: image_dimensions,
                    },
                    ImageUsage::from([ImageUseAs::Sampled, ImageUseAs::ColorAttachment].as_slice()),
                    ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    Self::output_image_color_format(),
                    ImageFlags::empty(),
                    ImageTiling::Optimal,
                ),
                None,
                Some(image_name.as_str()),
            )?;

            gbuffer_normal_image_handles.push(image);
        }

        let mut gbuffer_texture_image_handles: smallvec::SmallVec<
            [_; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        for index in 0..(frames_in_flight as usize) {
            let image_name = format!("mesh_rendering.gbuffer_texture_images[{index}]");
            let image = Image::new(
                device.clone(),
                ConcreteImageDescriptor::new(
                    ImageDimensions::Image2D {
                        extent: image_dimensions,
                    },
                    ImageUsage::from([ImageUseAs::Sampled, ImageUseAs::ColorAttachment].as_slice()),
                    ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    Self::output_image_color_format(),
                    ImageFlags::empty(),
                    ImageTiling::Optimal,
                ),
                None,
                Some(image_name.as_str()),
            )?;

            gbuffer_texture_image_handles.push(image);
        }

        let memory_required: u64 = gbuffer_depth_stencil_image_handles
            .iter()
            .map(|obj| obj.memory_requirements().size() + obj.memory_requirements().alignment())
            .chain(gbuffer_position_image_handles.iter().map(|obj| {
                obj.memory_requirements().size() + obj.memory_requirements().alignment()
            }))
            .chain(gbuffer_normal_image_handles.iter().map(|obj| {
                obj.memory_requirements().size() + obj.memory_requirements().alignment()
            }))
            .chain(gbuffer_texture_image_handles.iter().map(|obj| {
                obj.memory_requirements().size() + obj.memory_requirements().alignment()
            }))
            .sum();

        let memory_pool = {
            let image_handles: Vec<&dyn MemoryRequiring> = gbuffer_depth_stencil_image_handles
                .iter()
                .map(|obj| obj as &dyn MemoryRequiring)
                .chain(
                    gbuffer_position_image_handles
                        .iter()
                        .map(|obj| obj as &dyn MemoryRequiring),
                )
                .chain(
                    gbuffer_normal_image_handles
                        .iter()
                        .map(|obj| obj as &dyn MemoryRequiring),
                )
                .chain(
                    gbuffer_texture_image_handles
                        .iter()
                        .map(|obj| obj as &dyn MemoryRequiring),
                )
                .collect();

            // one block for each resource to take care of the alignment,
            // plus the number of blocks needed to have memory_required bytes,
            // plus a few extra blocks
            let allocator = DefaultAllocator::with_blocksize(
                1024u64,
                (4u64 * (frames_in_flight as u64 * 4u64)) + ((memory_required / 1024u64) + 4u64),
            );

            let memory_heap = MemoryHeap::new(
                device.clone(),
                ConcreteMemoryHeapDescriptor::new(
                    MemoryType::DeviceLocal(None),
                    allocator.total_size(),
                ),
                MemoryRequirements::try_from(image_handles.as_slice())?,
            )?;

            MemoryPool::new(
                memory_heap,
                Arc::new(allocator),
                MemoryPoolFeatures::from([].as_slice()),
            )?
        };

        let mut gbuffer_depth_stencil_image_views: smallvec::SmallVec<
            [Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        let mut gbuffer_depth_stencil_images: smallvec::SmallVec<
            [Arc<AllocatedImage>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        for (index, image) in gbuffer_depth_stencil_image_handles.into_iter().enumerate() {
            let allocated_image = AllocatedImage::new(memory_pool.clone(), image)?;
            gbuffer_depth_stencil_images.push(allocated_image.clone());

            let image_view_name =
                format!("mesh_rendering.gbuffer_depth_stencil_image_views[{index}]");
            gbuffer_depth_stencil_image_views.push(ImageView::new(
                allocated_image,
                Some(ImageViewType::Image2D),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(image_view_name.as_str()),
            )?);
        }

        let mut gbuffer_position_image_views: smallvec::SmallVec<
            [Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        let mut gbuffer_position_images: smallvec::SmallVec<
            [Arc<AllocatedImage>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        for (index, image) in gbuffer_position_image_handles.into_iter().enumerate() {
            let allocated_image = AllocatedImage::new(memory_pool.clone(), image)?;
            gbuffer_position_images.push(allocated_image.clone());

            let image_view_name = format!("mesh_rendering.gbuffer_position_image_views[{index}]");
            gbuffer_position_image_views.push(ImageView::new(
                allocated_image,
                Some(ImageViewType::Image2D),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(image_view_name.as_str()),
            )?);
        }

        let mut gbuffer_normal_image_views: smallvec::SmallVec<
            [Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        let mut gbuffer_normal_images: smallvec::SmallVec<
            [Arc<AllocatedImage>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        for (index, image) in gbuffer_normal_image_handles.into_iter().enumerate() {
            let allocated_image = AllocatedImage::new(memory_pool.clone(), image)?;
            gbuffer_normal_images.push(allocated_image.clone());

            let image_view_name = format!("mesh_rendering.gbuffer_normal_image_views[{index}]");
            gbuffer_normal_image_views.push(ImageView::new(
                allocated_image,
                Some(ImageViewType::Image2D),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(image_view_name.as_str()),
            )?);
        }

        let mut gbuffer_texture_image_views: smallvec::SmallVec<
            [Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        let mut gbuffer_texture_images: smallvec::SmallVec<
            [Arc<AllocatedImage>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        for (index, image) in gbuffer_texture_image_handles.into_iter().enumerate() {
            let allocated_image = AllocatedImage::new(memory_pool.clone(), image)?;
            gbuffer_texture_images.push(allocated_image.clone());

            let image_view_name = format!("mesh_rendering.gbuffer_texture_image_views[{index}]");
            gbuffer_texture_image_views.push(ImageView::new(
                allocated_image,
                Some(ImageViewType::Image2D),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(image_view_name.as_str()),
            )?);
        }

        let renderpass = RenderPass::new(
            device.clone(),
            &[
                // depth+stencil attachment
                AttachmentDescription::new(
                    Self::output_image_depth_stencil_format(),
                    ImageMultisampling::SamplesPerPixel1,
                    ImageLayout::Undefined,
                    Self::output_image_depth_stencil_layout(),
                    AttachmentLoadOp::Clear,
                    AttachmentStoreOp::Store,
                    AttachmentLoadOp::Clear,
                    AttachmentStoreOp::Store,
                ),
                // vPosition attachment
                AttachmentDescription::new(
                    Self::output_image_color_format(),
                    ImageMultisampling::SamplesPerPixel1,
                    ImageLayout::Undefined,
                    Self::output_image_color_layout(),
                    AttachmentLoadOp::Clear,
                    AttachmentStoreOp::Store,
                    AttachmentLoadOp::Clear,
                    AttachmentStoreOp::Store,
                ),
                // vNormal attachment
                AttachmentDescription::new(
                    Self::output_image_color_format(),
                    ImageMultisampling::SamplesPerPixel1,
                    ImageLayout::Undefined,
                    Self::output_image_color_layout(),
                    AttachmentLoadOp::Clear,
                    AttachmentStoreOp::Store,
                    AttachmentLoadOp::Clear,
                    AttachmentStoreOp::Store,
                ),
                // vTexture attachment
                AttachmentDescription::new(
                    Self::output_image_color_format(),
                    ImageMultisampling::SamplesPerPixel1,
                    ImageLayout::Undefined,
                    Self::output_image_color_layout(),
                    AttachmentLoadOp::Clear,
                    AttachmentStoreOp::Store,
                    AttachmentLoadOp::Clear,
                    AttachmentStoreOp::Store,
                ),
            ],
            &[RenderSubPassDescription::new(&[], &[1, 2, 3], Some(0))],
        )?;

        let vertex_shader =
            VertexShader::new(device.clone(), &[], &[], FINAL_RENDERING_VERTEX_SPV).unwrap();

        let fragment_shader =
            FragmentShader::new(device.clone(), &[], &[], FINAL_RENDERING_FRAGMENT_SPV).unwrap();

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            &[
                        /*BindingDescriptor::new(
                            shader_access,
                            binding_type,
                            binding_point,
                            binding_count
                        )*/
                    ],
            &[],
            Some("pipeline_layout"),
        )?;

        let graphics_pipeline = GraphicsPipeline::new(
            None,
            renderpass.clone(),
            0,
            ImageMultisampling::SamplesPerPixel1,
            Some(DepthConfiguration::new(
                true,
                DepthCompareOp::Always,
                Some((0.0, 1.0)),
            )),
            None,
            None,
            pipeline_layout,
            [
                // vertex position data
                VertexInputBinding::new(
                    VertexInputRate::PerVertex,
                    0u32,
                    &[VertexInputAttribute::new(0, 0, AttributeType::Vec3)],
                ),
                // vertex normal data
                VertexInputBinding::new(
                    VertexInputRate::PerVertex,
                    0u32,
                    &[VertexInputAttribute::new(
                        1,
                        4u32 * 3u32,
                        AttributeType::Vec3,
                    )],
                ),
                // vertex (diffuse) text coords
                VertexInputBinding::new(
                    VertexInputRate::PerVertex,
                    0u32,
                    &[VertexInputAttribute::new(
                        2,
                        (4u32 * 3u32) + (4u32 * 3u32),
                        AttributeType::Vec2,
                    )],
                ),
            ]
            .as_slice(),
            Rasterizer::new(
                PolygonMode::Fill,
                FrontFace::CounterClockwise,
                CullMode::None,
                None,
            ),
            (vertex_shader, None),
            (fragment_shader, None),
            Some("mesh_rendering.graphics_pipeline"),
        )?;

        let mut framebuffers = smallvec::smallvec![];
        for iw in 0..(frames_in_flight as usize) {
            let framebuffer = Framebuffer::new(
                renderpass.clone(),
                [
                    gbuffer_depth_stencil_image_views[iw].clone(),
                    gbuffer_position_image_views[iw].clone(),
                    gbuffer_normal_image_views[iw].clone(),
                    gbuffer_texture_image_views[iw].clone(),
                ]
                .as_slice(),
                image_dimensions,
                1,
            )?;

            framebuffers.push(framebuffer);
        }

        Ok(Self {
            image_dimensions,
            _memory_pool: memory_pool,
            graphics_pipeline,

            gbuffer_depth_stencil_images,
            gbuffer_depth_stencil_image_views,
            gbuffer_position_images,
            gbuffer_position_image_views,
            gbuffer_normal_images,
            gbuffer_normal_image_views,
            gbuffer_texture_images,
            gbuffer_texture_image_views,

            framebuffers,
        })
    }

    pub fn record_rendering_commands(
        &self,
        current_frame: usize,
        meshes: Arc<Mutex<Manager>>,
        recorder: &mut CommandBufferRecorder,
    ) -> [(Arc<ImageView>, ImageSubresourceRange, ImageLayout); 3] {
        recorder.begin_renderpass(
            self.framebuffers[current_frame].clone(),
            &[ClearValues::new(Some(ColorClearValues::Vec4(
                0.0, 0.0, 0.0, 1.0,
            )))],
        );

        recorder.bind_graphics_pipeline(
            self.graphics_pipeline.clone(),
            Some(Viewport::new(
                0.0f32,
                0.0f32,
                self.image_dimensions.width() as f32,
                self.image_dimensions.height() as f32,
                0.0f32,
                0.0f32,
            )),
            Some(Scissor::new(0, 0, self.image_dimensions)),
        );

        // bind vertex buffer + bind index buffer
        // TODO: vkCmdBindVertexBuffers vkCmdBindIndexBuffer

        // TODO: vkCmdPushConstants

        // TODO: vkCmdDrawIndexed

        recorder.draw(0, 3, 0, 1);
        let meshes_lck = meshes.lock().unwrap();
        meshes_lck.foreach_object(|loaded_obj| {});

        recorder.end_renderpass();

        [
            (
                self.gbuffer_position_image_views[current_frame].clone(),
                ImageSubresourceRange::from(
                    self.gbuffer_position_images[current_frame].clone() as Arc<dyn ImageTrait>
                ),
                Self::output_image_color_layout(),
            ),
            (
                self.gbuffer_normal_image_views[current_frame].clone(),
                ImageSubresourceRange::from(
                    self.gbuffer_position_images[current_frame].clone() as Arc<dyn ImageTrait>
                ),
                Self::output_image_color_layout(),
            ),
            (
                self.gbuffer_texture_image_views[current_frame].clone(),
                ImageSubresourceRange::from(
                    self.gbuffer_position_images[current_frame].clone() as Arc<dyn ImageTrait>
                ),
                Self::output_image_color_layout(),
            ),
        ]
    }
}
