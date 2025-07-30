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
    framebuffer::Framebuffer,
    graphics_pipeline::{
        CullMode, DepthCompareOp, DepthConfiguration, FrontFace, GraphicsPipeline, PolygonMode,
        Rasterizer, Scissor, Viewport,
    },
    image::{CommonImageFormat, ImageFormat, ImageLayout, ImageMultisampling},
    image_view::ImageView,
    pipeline_layout::PipelineLayout,
    renderpass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, RenderPass,
        RenderSubPassDescription,
    },
    sampler::{Filtering, MipmapMode, Sampler},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::ShaderStagesAccess,
    shaders::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
};

use crate::rendering::{
    MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingResult, rendering_dimensions::RenderingDimensions,
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

void main() {
    outColor = texture(src, in_vTextureUV);
}
"#,
    glsl,
    frag,
    vulkan1_0,
    entry = "main"
);

pub struct HDRTransform {
    descriptor_sets: smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    pipeline_layout: Arc<PipelineLayout>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    sampler: Arc<Sampler>,
}

/// Represents the stage of the pipeline that preceeds the final rendering:
/// the one that transform the raw result into something that can be sampled
/// to generate the image to be presented: applies tone mapping and/or hdr.
impl HDRTransform {
    /// Returns the layout of the input 2D image MUST be in where the rendering operation starts.
    #[inline]
    pub fn image_input_format() -> ImageLayout {
        ImageLayout::ShaderReadOnlyOptimal
    }

    #[inline]
    pub fn image_output_format() -> ImageFormat {
        CommonImageFormat::r32g32b32a32_sfloat.into()
    }

    #[inline]
    pub fn renderpass(&self) -> Arc<RenderPass> {
        self.graphics_pipeline.renderpass()
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
            Some("renderquad_descriptor_pool"),
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

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            &[descriptor_set_layout.clone()],
            &[],
            Some("pipeline_layout"),
        )?;

        let vertex_shader = VertexShader::new(
            device.clone(),
            &[],
            binding_descriptors.as_slice(),
            RENDERQUAD_VERTEX_SPV,
        )?;

        let fragment_shader = FragmentShader::new(
            device.clone(),
            &[],
            binding_descriptors.as_slice(),
            RENDERQUAD_FRAGMENT_SPV,
        )?;

        let renderpass = RenderPass::new(
            device.clone(),
            &[AttachmentDescription::new(
                Self::image_output_format(),
                ImageMultisampling::SamplesPerPixel1,
                ImageLayout::Undefined,
                ImageLayout::ShaderReadOnlyOptimal,
                AttachmentLoadOp::Clear,
                AttachmentStoreOp::Store,
                AttachmentLoadOp::Clear,
                AttachmentStoreOp::Store,
            )],
            &[RenderSubPassDescription::new(&[], &[0], None)],
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
            Some("renderquad_pipeline"),
        )?;

        let sampler = Sampler::new(
            device.clone(),
            Filtering::Nearest,
            Filtering::Nearest,
            MipmapMode::ModeNearest,
            0.0,
        )
        .unwrap();

        Ok(Self {
            descriptor_sets,
            pipeline_layout,
            graphics_pipeline,
            sampler,
        })
    }

    pub fn record_rendering_commands(
        &self,
        input_image_view: Arc<ImageView>,
        framebuffer: Arc<Framebuffer>,
        current_frame: usize,
        recorder: &mut CommandBufferRecorder,
    ) {
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

        recorder.begin_renderpass(
            framebuffer,
            &[ClearValues::new(Some(ColorClearValues::Vec4(
                1.0, 1.0, 1.0, 1.0,
            )))],
        );
        recorder.bind_graphics_pipeline(self.graphics_pipeline.clone(), None, None);
        recorder.bind_descriptor_sets_for_graphics_pipeline(
            self.pipeline_layout.clone(),
            0,
            &[self.descriptor_sets[current_frame].clone()],
        );
        recorder.draw(0, 6, 0, 1);

        recorder.end_renderpass();
    }
}
