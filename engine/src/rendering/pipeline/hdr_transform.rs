use std::sync::Arc;

use inline_spirv::*;

use vulkan_framework::{
    command_buffer::{ClearValues, ColorClearValues, CommandBufferRecorder},
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor,
        DescriptorPoolSizesAcceletarionStructureKHR, DescriptorPoolSizesConcreteDescriptor,
    },
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
        AllocatedImage, CommonImageFormat, ConcreteImageDescriptor, Image, Image2DDimensions,
        ImageFlags, ImageFormat, ImageLayout, ImageMultisampling, ImageSubresourceRange,
        ImageTiling, ImageTrait, ImageUseAs,
    },
    image_view::{ImageView, ImageViewType},
    memory_allocator::{DefaultAllocator, MemoryAllocator},
    memory_barriers::{ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    memory_heap::{ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryRequirements, MemoryType},
    memory_pool::{MemoryPool, MemoryPoolFeatures},
    memory_requiring::MemoryRequiring,
    pipeline_layout::PipelineLayout,
    pipeline_stage::{PipelineStage, PipelineStages},
    push_constant_range::PushConstanRange,
    queue_family::QueueFamily,
    sampler::{Filtering, MipmapMode, Sampler},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::{ShaderStageAccessIn, ShaderStagesAccess},
    shaders::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
};

use crate::{
    core::hdr::HDR,
    rendering::{
        MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingResult, rendering_dimensions::RenderingDimensions,
    },
};

const RENDERQUAD_VERTEX_SPV: &[u32] = inline_spirv!(
    r#"
#version 460

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
    glsl,
    vert,
    vulkan1_0,
    entry = "main"
);

const RENDERQUAD_FRAGMENT_SPV: &[u32] = inline_spirv!(
    r#"
#version 460

layout (location = 0) in vec2 in_vTextureUV;

layout(binding = 0, set = 0) uniform sampler2D src;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform HDR {
    float gamma;
    float exposure;
} hdr;

void main() {
    outColor = texture(src, in_vTextureUV) * (hdr.gamma + hdr.exposure);
}
"#,
    glsl,
    frag,
    vulkan1_0,
    entry = "main"
);

pub struct HDRTransform {
    _memory_pool: Arc<MemoryPool>,

    descriptor_sets: smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    push_constant_size: u32,
    push_constant_access: ShaderStagesAccess,
    pipeline_layout: Arc<PipelineLayout>,
    graphics_pipeline: Arc<GraphicsPipeline>,

    image_dimensions: Image2DDimensions,
    images: smallvec::SmallVec<[Arc<AllocatedImage>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    image_views: smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    sampler: Arc<Sampler>,
}

/// Represents the stage of the pipeline that preceeds the final rendering:
/// the one that transform the raw result into something that can be sampled
/// to generate the image to be presented: applies tone mapping and/or hdr.
impl HDRTransform {
    /// Returns the format of the input 2D image MUST be in where the rendering operation starts.
    #[inline]
    pub fn image_input_format() -> ImageLayout {
        ImageLayout::ShaderReadOnlyOptimal
    }

    fn output_image_layout() -> ImageLayout {
        ImageLayout::ColorAttachmentOptimal
    }

    #[inline]
    fn image_output_format() -> ImageFormat {
        CommonImageFormat::r32g32b32a32_sfloat.into()
    }

    pub fn new(
        device: Arc<Device>,
        frames_in_flight: u32,
        render_area: &RenderingDimensions,
    ) -> RenderingResult<Self> {
        let image_dimensions = render_area.into();

        let descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    frames_in_flight,
                    0,
                    frames_in_flight,
                    0,
                    0,
                    0,
                    0,
                    0,
                    Some(DescriptorPoolSizesAcceletarionStructureKHR::new(
                        frames_in_flight,
                    )),
                ),
                frames_in_flight,
            ),
            Some("hdr_transform.descriptor_pool"),
        )?;

        let binding_descriptor = BindingDescriptor::new(
            ShaderStagesAccess::graphics(),
            BindingType::Native(NativeBindingType::CombinedImageSampler),
            0,
            1,
        );

        let binding_descriptors = [binding_descriptor.clone()];

        let descriptor_set_layout =
            DescriptorSetLayout::new(device.clone(), binding_descriptors.as_slice())?;

        let mut descriptor_sets = smallvec::smallvec![];
        for _ in 0..frames_in_flight {
            let descriptor_set =
                DescriptorSet::new(descriptor_pool.clone(), descriptor_set_layout.clone())?;
            descriptor_sets.push(descriptor_set);
        }

        let push_constant_size = 4u32 * 2u32;
        let push_constant_access = [ShaderStageAccessIn::Fragment].as_slice().into();

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            &[descriptor_set_layout.clone()],
            &[PushConstanRange::new(
                0,
                push_constant_size.to_owned(),
                push_constant_access,
            )],
            Some("hdr_transform.pipeline_layout"),
        )?;

        let vertex_shader = VertexShader::new(
            device.clone(),
            &[],
            binding_descriptors.as_slice(),
            RENDERQUAD_VERTEX_SPV,
        )?;

        let fragment_shader = FragmentShader::new(
            device.clone(),
            &[PushConstanRange::new(
                0,
                (std::mem::size_of::<u32>() as u32) * 2u32,
                push_constant_access,
            )],
            binding_descriptors.as_slice(),
            RENDERQUAD_FRAGMENT_SPV,
        )?;

        let graphics_pipeline = GraphicsPipeline::new(
            None,
            DynamicRendering::new([Self::image_output_format()].as_slice(), None, None),
            ImageMultisampling::SamplesPerPixel1,
            Some(DepthConfiguration::new(
                true,
                DepthCompareOp::Always,
                Some((0.0, 1.0)),
            )),
            Some(Viewport::new(
                0.0f32,
                0.0f32,
                render_area.width() as f32,
                render_area.height() as f32,
                0.0f32,
                0.0f32,
            )),
            Some(Scissor::new(0, 0, image_dimensions)),
            pipeline_layout.clone(),
            &[],
            Rasterizer::new(
                PolygonMode::Fill,
                FrontFace::CounterClockwise,
                CullMode::None,
                None,
            ),
            (vertex_shader, None),
            (fragment_shader, None),
            Some("hdr_transform.pipeline"),
        )?;

        let mut image_handles: smallvec::SmallVec<[_; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]> =
            smallvec::smallvec![];
        for index in 0..(frames_in_flight as usize) {
            let image_name = format!("hdr_transform.image[{index}]");
            let image = Image::new(
                device.clone(),
                ConcreteImageDescriptor::new(
                    image_dimensions.into(),
                    [ImageUseAs::Sampled, ImageUseAs::ColorAttachment]
                        .as_slice()
                        .into(),
                    ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    Self::image_output_format(),
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
            .map(|obj| obj.memory_requirements().size() + obj.memory_requirements().alignment())
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

            let image_view_name = format!("hdr_transform.image_view[{index}]");
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

        let sampler = Sampler::new(
            device.clone(),
            Filtering::Nearest,
            Filtering::Nearest,
            MipmapMode::ModeNearest,
            0.0,
        )
        .unwrap();

        Ok(Self {
            _memory_pool: memory_pool,

            descriptor_sets,

            push_constant_size,
            push_constant_access,
            pipeline_layout,
            graphics_pipeline,

            image_dimensions,
            images,
            image_views,

            sampler,
        })
    }

    pub fn record_rendering_commands(
        &self,
        queue_family: Arc<QueueFamily>,
        hdr: &HDR,
        input_image_view: Arc<ImageView>,
        current_frame: usize,
        recorder: &mut CommandBufferRecorder,
    ) -> (Arc<ImageView>, ImageSubresourceRange, ImageLayout) {
        self.descriptor_sets[current_frame]
            .bind_resources(|binder| {
                binder
                    .bind_combined_images_samplers(
                        0,
                        &[(
                            Self::image_input_format(),
                            input_image_view,
                            self.sampler.clone(),
                        )],
                    )
                    .unwrap()
            })
            .unwrap();

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
                Self::output_image_layout(),
                queue_family.clone(),
                queue_family.clone(),
            )]
            .as_slice(),
        );

        let rendering_color_attachments = [DynamicRenderingAttachment::new(
            self.image_views[current_frame].clone(),
            Self::output_image_layout(),
            ClearValues::new(Some(ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0))),
            AttachmentLoadOp::Clear,
            AttachmentStoreOp::Store,
        )];
        recorder.graphics_rendering(
            self.image_dimensions,
            rendering_color_attachments.as_slice(),
            None,
            None,
            |recorder| {
                recorder.bind_graphics_pipeline(self.graphics_pipeline.clone(), None, None);
                recorder.bind_descriptor_sets_for_graphics_pipeline(
                    self.pipeline_layout.clone(),
                    0,
                    &[self.descriptor_sets[current_frame].clone()],
                );

                let push_constant = [hdr.gamma(), hdr.exposure()];
                assert_eq!(
                    std::mem::size_of_val(&push_constant) as usize,
                    self.push_constant_size as usize
                );

                recorder.push_constant(
                    self.pipeline_layout.clone(),
                    self.push_constant_access,
                    0,
                    unsafe {
                        ::core::slice::from_raw_parts(
                            (&push_constant[0] as *const _) as *const u8,
                            self.push_constant_size as usize,
                        )
                    },
                );

                recorder.draw(0, 6, 0, 1);
            },
        );

        (
            self.image_views[current_frame].clone(),
            ImageSubresourceRange::from(self.images[current_frame].clone() as Arc<dyn ImageTrait>),
            Self::output_image_layout(),
        )
    }
}
