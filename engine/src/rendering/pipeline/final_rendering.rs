use std::sync::Arc;

use inline_spirv::*;

use crate::rendering::{
    MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingResult, rendering_dimensions::RenderingDimensions,
};

use vulkan_framework::{
    command_buffer::{ClearValues, ColorClearValues, CommandBufferRecorder},
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::Device,
    dynamic_rendering::{
        AttachmentLoadOp, AttachmentStoreOp, DynamicRendering, DynamicRenderingAttachment,
    },
    graphics_pipeline::{
        CullMode, DepthCompareOp, DepthConfiguration, FrontFace, GraphicsPipeline, PolygonMode,
        Rasterizer, Scissor, Viewport,
    },
    image::{
        AllocatedImage, CommonImageFormat, ConcreteImageDescriptor, Image, Image1DTrait,
        Image2DDimensions, Image2DTrait, ImageDimensions, ImageFlags, ImageFormat, ImageLayout,
        ImageMultisampling, ImageSubresourceRange, ImageTiling, ImageTrait, ImageUsage, ImageUseAs,
    },
    image_view::{ImageView, ImageViewType},
    memory_allocator::{DefaultAllocator, MemoryAllocator},
    memory_barriers::{ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    memory_heap::{ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryRequirements, MemoryType},
    memory_pool::{MemoryPool, MemoryPoolFeatures},
    memory_requiring::AllocationRequiring,
    pipeline_layout::{PipelineLayout, PipelineLayoutDependant},
    pipeline_stage::{PipelineStage, PipelineStages},
    queue_family::QueueFamily,
    shaders::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
};

const FINAL_RENDERING_VERTEX_SPV: &[u32] = inline_spirv!(
    r#"
#version 450 core

layout (location = 0) out vec2 out_vTextureUV;

const vec2 vQuadPosition[6] = {
	vec2(-1, -1),
	vec2(+1, -1),
	vec2(-1, +1),
	vec2(-1, +1),
	vec2(+1, +1),
	vec2(+1, -1),
};

const vec2 vUVCoordinates[6] = {
    vec2(0.0, 0.0),
	vec2(1.0, 0.0),
	vec2(0.0, 1.0),
	vec2(0.0, 1.0),
	vec2(1.0, 1.0),
	vec2(1.0, 0.0),
};

void main() {
    out_vTextureUV = vUVCoordinates[gl_VertexIndex];
	gl_Position = vec4(vQuadPosition[gl_VertexIndex], 0.0, 1.0);
}
"#,
    vert
);

const FINAL_RENDERING_FRAGMENT_SPV: &[u32] = inline_spirv!(
    r#"
#version 450 core

layout(location = 0) out vec4 outColor;

layout (location = 0) in vec2 in_vTextureUV;

// gbuffer: 0 for position, 1 for normal, 2 for diffuse texture
layout(set = 0, binding = 0) uniform sampler2D gbuffer[3];

void main() {
    const vec3 in_vPosition_worldspace = texture(gbuffer[0], in_vTextureUV).xyz;
    const vec3 in_vNormal_worldspace = texture(gbuffer[1], in_vTextureUV).xyz;
    const vec4 in_vDiffuseAlbedo = texture(gbuffer[2], in_vTextureUV);

    outColor = vec4(in_vPosition_worldspace.xyz, 1.0);
}
"#,
    frag
);

/// This is the stage of the pipeline that assembles every other results into what will be fed
/// to the HDR stage.
///
/// This is the deferred shading step, basically.
pub struct FinalRendering {
    _memory_pool: Arc<MemoryPool>,

    image_dimensions: Image2DDimensions,
    images: smallvec::SmallVec<[Arc<AllocatedImage>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    image_views: smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    graphics_pipeline: Arc<GraphicsPipeline>,
}

impl FinalRendering {
    fn output_image_layout() -> ImageLayout {
        ImageLayout::ColorAttachmentOptimal
    }

    pub fn output_image_format() -> ImageFormat {
        ImageFormat::from(CommonImageFormat::r32g32b32a32_sfloat)
    }

    pub fn new(
        device: Arc<Device>,
        gbuffer_descriptor_set_layout: Arc<DescriptorSetLayout>,
        render_area: &RenderingDimensions,
        frames_in_flight: u32,
    ) -> RenderingResult<Self> {
        let image_dimensions = render_area.into();

        let mut image_handles: smallvec::SmallVec<[_; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]> =
            smallvec::smallvec![];
        for index in 0..(frames_in_flight as usize) {
            let image_name = format!("final_rendering.image[{index}]");
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
                    Self::output_image_format(),
                    ImageFlags::empty(),
                    ImageTiling::Optimal,
                ),
                None,
                Some(image_name.as_str()),
            )?;

            image_handles.push(image);
        }

        let memory_required: u64 = image_handles
            .iter()
            .map(|obj| {
                obj.allocation_requirements().size() + obj.allocation_requirements().alignment()
            })
            .sum();

        let allocator = DefaultAllocator::with_blocksize(
            1024,
            (frames_in_flight as u64) + (memory_required / 1024u64),
        );
        let memory_heap = MemoryHeap::new(
            device.clone(),
            ConcreteMemoryHeapDescriptor::new(
                MemoryType::DeviceLocal(None),
                allocator.total_size(),
            ),
            MemoryRequirements::try_from(image_handles.as_slice())?,
        )?;

        let memory_pool = MemoryPool::new(
            memory_heap,
            Arc::new(allocator),
            MemoryPoolFeatures::from([].as_slice()),
        )?;

        let mut image_views: smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]> =
            smallvec::smallvec![];
        let mut images: smallvec::SmallVec<[Arc<AllocatedImage>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]> =
            smallvec::smallvec![];
        for (index, image) in image_handles.into_iter().enumerate() {
            let allocated_image = AllocatedImage::new(memory_pool.clone(), image)?;

            images.push(allocated_image.clone());

            let image_view_name = format!("final_rendering.image_view[{index}]");
            let image_view = ImageView::new(
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
            )?;

            image_views.push(image_view);
        }

        let vertex_shader =
            VertexShader::new(device.clone(), &[], &[], FINAL_RENDERING_VERTEX_SPV).unwrap();

        let fragment_shader =
            FragmentShader::new(device.clone(), &[], &[], FINAL_RENDERING_FRAGMENT_SPV).unwrap();

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            &[gbuffer_descriptor_set_layout],
            &[],
            Some("final_rendering.pipeline_layout"),
        )?;

        let graphics_pipeline = GraphicsPipeline::new(
            None,
            DynamicRendering::new([Self::output_image_format()].as_slice(), None, None),
            ImageMultisampling::SamplesPerPixel1,
            Some(DepthConfiguration::new(
                true,
                DepthCompareOp::Always,
                Some((0.0, 1.0)),
            )),
            None,
            None,
            pipeline_layout,
            &[],
            Rasterizer::new(
                PolygonMode::Fill,
                FrontFace::CounterClockwise,
                CullMode::None,
                None,
            ),
            (vertex_shader, None),
            (fragment_shader, None),
            Some("final_rendering.graphics_pipeline"),
        )?;

        Ok(Self {
            _memory_pool: memory_pool,

            image_dimensions,
            image_views,
            images,

            graphics_pipeline,
        })
    }

    pub fn record_rendering_commands(
        &self,
        queue_family: Arc<QueueFamily>,
        gbuffer_descriptor_set: Arc<DescriptorSet>,
        current_frame: usize,
        recorder: &mut CommandBufferRecorder,
    ) -> (Arc<ImageView>, ImageSubresourceRange, ImageLayout) {
        let image_view = self.image_views[current_frame].clone();

        // Transition the framebuffer image into color attachment optimal layout,
        // so that the graphics pipeline has it in the best format
        recorder.image_barriers(
            [ImageMemoryBarrier::new(
                PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                MemoryAccess::from([].as_slice()),
                PipelineStages::from([PipelineStage::AllGraphics].as_slice()),
                MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                image_view.image().into(),
                ImageLayout::Undefined,
                ImageLayout::ColorAttachmentOptimal,
                queue_family.clone(),
                queue_family.clone(),
            )]
            .as_slice(),
        );

        let rendering_color_attachments = [DynamicRenderingAttachment::new(
            image_view.clone(),
            Self::output_image_layout(),
            ClearValues::new(Some(ColorClearValues::Vec4(0.0, 0.0, 0.0, 1.0))),
            AttachmentLoadOp::Clear,
            AttachmentStoreOp::Store,
        )];
        recorder.graphics_rendering(
            self.image_dimensions,
            rendering_color_attachments.as_slice(),
            None,
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

                recorder.bind_descriptor_sets_for_graphics_pipeline(
                    self.graphics_pipeline.get_parent_pipeline_layout(),
                    0,
                    [gbuffer_descriptor_set].as_slice(),
                );

                recorder.draw(0, 6, 0, 1);
            },
        );

        (
            image_view,
            ImageSubresourceRange::from(self.images[current_frame].clone() as Arc<dyn ImageTrait>),
            Self::output_image_layout(),
        )
    }
}
