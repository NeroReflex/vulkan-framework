use std::sync::Arc;

use inline_spirv::*;

use crate::rendering::{
    MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingResult, rendering_dimensions::RenderingDimensions,
    resources::object::Manager,
};

use vulkan_framework::{
    command_buffer::{
        ClearValues, ColorClearValues, CommandBufferRecorder, ImageMemoryBarrier, MemoryAccess,
        MemoryAccessAs,
    },
    device::Device,
    dynamic_rendering::{
        AttachmentLoadOp, AttachmentStoreOp, DynamicRendering, DynamicRenderingAttachment,
    },
    graphics_pipeline::{
        AttributeType, CullMode, DepthCompareOp, DepthConfiguration, FrontFace, GraphicsPipeline,
        PolygonMode, Rasterizer, Scissor, VertexInputAttribute, VertexInputBinding,
        VertexInputRate, Viewport,
    },
    image::{
        AllocatedImage, CommonImageFormat, ConcreteImageDescriptor, Image, Image1DTrait,
        Image2DDimensions, Image2DTrait, ImageDimensions, ImageFlags, ImageFormat, ImageLayout,
        ImageMultisampling, ImageSubresourceRange, ImageTiling, ImageUsage, ImageUseAs,
    },
    image_view::{ImageView, ImageViewType},
    memory_allocator::{DefaultAllocator, MemoryAllocator},
    memory_heap::{ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryRequirements, MemoryType},
    memory_pool::{MemoryPool, MemoryPoolFeatures},
    memory_requiring::MemoryRequiring,
    pipeline_layout::{PipelineLayout, PipelineLayoutDependant},
    pipeline_stage::{PipelineStage, PipelineStages},
    push_constant_range::PushConstanRange,
    queue_family::QueueFamily,
    shader_stage_access::{ShaderStageAccessIn, ShaderStagesAccess},
    shaders::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
};

const MESH_RENDERING_VERTEX_SPV: &[u32] = inline_spirv!(
    r#"
#version 450 core

layout (location = 0) in vec3 vertex_position_modelspace;
layout (location = 1) in vec3 vertex_normal_modelspace;
layout (location = 2) in vec2 vertex_texture;


//layout (location = 3) in uint vMaterialIndex;

void main() {
    
}
"#,
    vert
);

const MESH_RENDERING_FRAGMENT_SPV: &[u32] = inline_spirv!(
    r#"
#version 450 core

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform HDR {
    uint material_id;
} hdr;

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

    push_constants_access: ShaderStagesAccess,
    graphics_pipeline: Arc<GraphicsPipeline>,

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
}

impl MeshRendering {
    fn output_image_color_layout() -> ImageLayout {
        ImageLayout::ColorAttachmentOptimal
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

        let vertex_shader =
            VertexShader::new(device.clone(), &[], &[], MESH_RENDERING_VERTEX_SPV).unwrap();

        let fragment_shader =
            FragmentShader::new(device.clone(), &[], &[], MESH_RENDERING_FRAGMENT_SPV).unwrap();

        let push_constants_access = [ShaderStageAccessIn::Fragment].as_slice().into();

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
            &[PushConstanRange::new(0, 4u32, push_constants_access)],
            Some("mesh_rendering.pipeline_layout"),
        )?;

        let graphics_pipeline = GraphicsPipeline::new(
            None,
            DynamicRendering::new(
                [
                    Self::output_image_color_format(),
                    Self::output_image_color_format(),
                    Self::output_image_color_format(),
                ]
                .as_slice(),
                Some(Self::output_image_depth_stencil_format()),
                None,
            ),
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
                    Manager::vertex_buffer_position_stride(),
                    &[VertexInputAttribute::new(0, 0, AttributeType::Vec3)],
                ),
                // vertex normal data
                VertexInputBinding::new(
                    VertexInputRate::PerVertex,
                    Manager::vertex_buffer_normals_stride(),
                    &[VertexInputAttribute::new(1, 0, AttributeType::Vec3)],
                ),
                // vertex text coords
                VertexInputBinding::new(
                    VertexInputRate::PerVertex,
                    Manager::vertex_buffer_texture_uv_stride(),
                    &[VertexInputAttribute::new(2, 0, AttributeType::Vec2)],
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

        Ok(Self {
            image_dimensions,
            _memory_pool: memory_pool,

            push_constants_access,
            graphics_pipeline,

            gbuffer_depth_stencil_images,
            gbuffer_depth_stencil_image_views,
            gbuffer_position_images,
            gbuffer_position_image_views,
            gbuffer_normal_images,
            gbuffer_normal_image_views,
            gbuffer_texture_images,
            gbuffer_texture_image_views,
        })
    }

    pub fn record_rendering_commands<ManagerT>(
        &self,
        queue_family: Arc<QueueFamily>,
        current_frame: usize,
        meshes: ManagerT,
        recorder: &mut CommandBufferRecorder,
    ) -> [(Arc<ImageView>, ImageSubresourceRange, ImageLayout); 4]
    where
        ManagerT: std::ops::Deref<Target = Manager>,
    {
        let position_imageview = self.gbuffer_position_image_views[current_frame].clone();
        let normal_imageview = self.gbuffer_normal_image_views[current_frame].clone();
        let texture_imageview = self.gbuffer_texture_image_views[current_frame].clone();
        let depth_stencil_imageview = self.gbuffer_depth_stencil_image_views[current_frame].clone();

        // Transition the framebuffer images into depth/color attachment optimal layout,
        // so that the graphics pipeline has it in the best format
        recorder.image_barriers(
            [
                ImageMemoryBarrier::new(
                    PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                    MemoryAccess::from([].as_slice()),
                    PipelineStages::from([PipelineStage::AllGraphics].as_slice()),
                    MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                    position_imageview.image().into(),
                    ImageLayout::Undefined,
                    Self::output_image_color_layout(),
                    queue_family.clone(),
                    queue_family.clone(),
                ),
                ImageMemoryBarrier::new(
                    PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                    MemoryAccess::from([].as_slice()),
                    PipelineStages::from([PipelineStage::AllGraphics].as_slice()),
                    MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                    normal_imageview.image().into(),
                    ImageLayout::Undefined,
                    Self::output_image_color_layout(),
                    queue_family.clone(),
                    queue_family.clone(),
                ),
                ImageMemoryBarrier::new(
                    PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                    MemoryAccess::from([].as_slice()),
                    PipelineStages::from([PipelineStage::AllGraphics].as_slice()),
                    MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                    texture_imageview.image().into(),
                    ImageLayout::Undefined,
                    Self::output_image_color_layout(),
                    queue_family.clone(),
                    queue_family.clone(),
                ),
                ImageMemoryBarrier::new(
                    PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                    MemoryAccess::from([].as_slice()),
                    PipelineStages::from([PipelineStage::AllGraphics].as_slice()),
                    MemoryAccess::from([MemoryAccessAs::DepthStencilAttachmentWrite].as_slice()),
                    depth_stencil_imageview.image().into(),
                    ImageLayout::Undefined,
                    Self::output_image_depth_stencil_layout(),
                    queue_family.clone(),
                    queue_family.clone(),
                ),
            ]
            .as_slice(),
        );

        let rendering_color_attachments = [
            DynamicRenderingAttachment::new(
                position_imageview.clone(),
                Self::output_image_color_layout(),
                ClearValues::new(Some(ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0))),
                AttachmentLoadOp::Clear,
                AttachmentStoreOp::Store,
            ),
            DynamicRenderingAttachment::new(
                normal_imageview.clone(),
                Self::output_image_color_layout(),
                ClearValues::new(Some(ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0))),
                AttachmentLoadOp::Clear,
                AttachmentStoreOp::Store,
            ),
            DynamicRenderingAttachment::new(
                texture_imageview.clone(),
                Self::output_image_color_layout(),
                ClearValues::new(Some(ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0))),
                AttachmentLoadOp::Clear,
                AttachmentStoreOp::Store,
            ),
        ];
        let rendering_depth_attachment = DynamicRenderingAttachment::new(
            depth_stencil_imageview.clone(),
            Self::output_image_depth_stencil_layout(),
            ClearValues::new(Some(ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0))),
            AttachmentLoadOp::Clear,
            AttachmentStoreOp::Store,
        );
        recorder.graphics_rendering(
            self.image_dimensions,
            rendering_color_attachments.as_slice(),
            Some(&rendering_depth_attachment),
            None,
            |recorder| {
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

                // performs the actual rendering
                meshes.deref().guided_rendering(
                    recorder,
                    self.graphics_pipeline.get_parent_pipeline_layout(),
                    0,
                    4u32,
                    self.push_constants_access,
                );
            },
        );

        [
            (
                position_imageview.clone(),
                position_imageview.image().into(),
                Self::output_image_color_layout(),
            ),
            (
                normal_imageview.clone(),
                normal_imageview.image().into(),
                Self::output_image_color_layout(),
            ),
            (
                texture_imageview.clone(),
                texture_imageview.image().into(),
                Self::output_image_color_layout(),
            ),
            (
                depth_stencil_imageview.clone(),
                depth_stencil_imageview.image().into(),
                Self::output_image_color_layout(),
            ),
        ]
    }
}
