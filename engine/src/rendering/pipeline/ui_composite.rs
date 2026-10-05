use std::sync::Arc;

use inline_spirv::*;

use vulkan_framework::{
    clear_values::ColorClearValues,
    command_buffer::CommandBufferRecorder,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::Device,
    dynamic_rendering::{
        AttachmentStoreOp, DynamicRendering, DynamicRenderingColorAttachment,
        DynamicRenderingColorDefinition, RenderingAttachmentSetup,
    },
    graphics_pipeline::{
        CullMode, FrontFace, GraphicsPipeline, PolygonMode, Rasterizer, Scissor, Viewport,
    },
    image::{
        Image2DDimensions, ImageFormat, ImageLayout, ImageLayoutSwapchainKHR,
        ImageMultisampling,
    },
    image_view::ImageView,
    memory_barriers::{ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    pipeline_layout::PipelineLayout,
    pipeline_stage::{PipelineStage, PipelineStages},
    queue_family::QueueFamily,
    sampler::{Filtering, MipmapMode, Sampler},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::ShaderStagesAccess,
    shaders::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
};

use crate::rendering::RenderingResult;

const VERTEX_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
layout (location = 0) out vec2 out_uv;
const vec2 vQuadPosition[6] = vec2[6](
    vec2(-1, -1), vec2(+1, -1), vec2(-1, +1),
    vec2(-1, +1), vec2(+1, +1), vec2(+1, -1));
const vec2 vUVCoordinates[6] = vec2[6](
    vec2(0,0), vec2(1,0), vec2(0,1),
    vec2(0,1), vec2(1,1), vec2(1,0));
void main() {
    out_uv = vUVCoordinates[gl_VertexIndex];
    gl_Position = vec4(vQuadPosition[gl_VertexIndex], 0.0, 1.0);
}
"#,
    glsl,
    vert,
    vulkan1_0,
    entry = "main"
);

const FRAGMENT_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
layout (location = 0) in vec2 in_uv;
layout(binding = 0, set = 0) uniform sampler2D hdr_tex;
layout(binding = 1, set = 0) uniform sampler2D ui_tex;
layout(location = 0) out vec4 outColor;
void main() {
    vec4 scene = texture(hdr_tex, in_uv);
    vec4 ui = texture(ui_tex, in_uv);
    outColor = vec4(mix(scene.rgb, ui.rgb, ui.a), scene.a);
}
"#,
    glsl,
    frag,
    vulkan1_0,
    entry = "main"
);

pub struct UiComposite {
    pipeline_layout: Arc<PipelineLayout>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    sampler: Arc<Sampler>,
}

impl UiComposite {
    pub fn image_input_layout() -> ImageLayout {
        ImageLayout::ShaderReadOnlyOptimal
    }

    pub fn new(device: Arc<Device>, output_format: ImageFormat, width: u32, height: u32) -> RenderingResult<Self> {
        let bindings = [
            BindingDescriptor::new(
                ShaderStagesAccess::graphics(),
                BindingType::Native(NativeBindingType::CombinedImageSampler),
                0,
                1,
            ),
            BindingDescriptor::new(
                ShaderStagesAccess::graphics(),
                BindingType::Native(NativeBindingType::CombinedImageSampler),
                1,
                1,
            ),
        ];
        let descriptor_set_layout = DescriptorSetLayout::new(device.clone(), bindings.as_slice())?;
        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            &[descriptor_set_layout.clone()],
            &[],
            Some("ui_composite.pipeline_layout"),
        )?;
        let vertex_shader = VertexShader::new(device.clone(), VERTEX_SPV)?;
        let fragment_shader = FragmentShader::new(device.clone(), FRAGMENT_SPV)?;
        let graphics_pipeline = GraphicsPipeline::new(
            None,
            DynamicRendering::new(
                [DynamicRenderingColorDefinition::new(output_format)].as_slice(),
                None,
                None,
            ),
            ImageMultisampling::SamplesPerPixel1,
            None,
            Some(Viewport::new(0.0, 0.0, width as f32, height as f32, 0.0, 1.0)),
            Some(Scissor::new(0, 0, Image2DDimensions::new(width, height))),
            pipeline_layout.clone(),
            &[],
            Rasterizer::new(PolygonMode::Fill, FrontFace::CounterClockwise, CullMode::None, None),
            (vertex_shader, None),
            (fragment_shader, None),
            Some("ui_composite.pipeline"),
        )?;
        let sampler = Sampler::new(
            device.clone(),
            Filtering::Linear,
            Filtering::Linear,
            MipmapMode::ModeNearest,
            0.0,
        )?;
        Ok(Self {
            pipeline_layout,
            graphics_pipeline,
            descriptor_set_layout,
            sampler,
        })
    }

    pub fn record_rendering_commands(
        &self,
        queue_family: Arc<QueueFamily>,
        draw_area: Image2DDimensions,
        hdr_image: Arc<ImageView>,
        ui_image: Arc<ImageView>,
        output_image: Arc<ImageView>,
        recorder: &mut CommandBufferRecorder,
    ) {
        let descriptor_pool = DescriptorPool::new(
            hdr_image.image().get_parent_device(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(0, 2, 0, 0, 0, 0, 0, 0, 0, None),
                1,
            ),
            Some("ui_composite.descriptor_pool"),
        )
        .unwrap();
        let descriptor_set =
            DescriptorSet::new(descriptor_pool, self.descriptor_set_layout.clone()).unwrap();
        descriptor_set
            .bind_resources(|binder| {
                binder
                    .bind_combined_images_samplers(
                        0,
                        &[(ImageLayout::ShaderReadOnlyOptimal, hdr_image.clone(), self.sampler.clone())],
                    )
                    .unwrap();
                binder
                    .bind_combined_images_samplers(
                        1,
                        &[(ImageLayout::ShaderReadOnlyOptimal, ui_image.clone(), self.sampler.clone())],
                    )
                    .unwrap();
            })
            .unwrap();

        recorder.pipeline_barriers([ImageMemoryBarrier::new(
            [PipelineStage::FragmentShader].as_slice().into(),
            [MemoryAccessAs::ShaderRead].as_slice().into(),
            [PipelineStage::ColorAttachmentOutput].as_slice().into(),
            [MemoryAccessAs::ColorAttachmentWrite].as_slice().into(),
            output_image.image().into(),
            ImageLayout::Undefined,
            ImageLayout::ColorAttachmentOptimal,
            queue_family.clone(),
            queue_family.clone(),
        )
        .into()]);

        recorder.graphics_rendering(
            draw_area,
            [DynamicRenderingColorAttachment::new(
                output_image.clone(),
                RenderingAttachmentSetup::clear(ColorClearValues::Vec4(0.0, 0.0, 0.0, 1.0)),
                AttachmentStoreOp::Store,
            )]
            .as_slice(),
            None,
            None,
            |recorder| {
                recorder.bind_graphics_pipeline(self.graphics_pipeline.clone(), None, None);
                recorder.bind_descriptor_sets_for_graphics_pipeline(
                    self.pipeline_layout.clone(),
                    0,
                    &[descriptor_set],
                );
                recorder.draw(0, 6, 0, 1);
            },
        );

        recorder.pipeline_barriers([ImageMemoryBarrier::new(
            [PipelineStage::ColorAttachmentOutput].as_slice().into(),
            [MemoryAccessAs::ColorAttachmentWrite].as_slice().into(),
            [PipelineStage::FragmentShader].as_slice().into(),
            [MemoryAccessAs::ShaderRead].as_slice().into(),
            output_image.image().into(),
            ImageLayout::ColorAttachmentOptimal,
            ImageLayout::ShaderReadOnlyOptimal,
            queue_family.clone(),
            queue_family.clone(),
        )
        .into()]);
    }
}
