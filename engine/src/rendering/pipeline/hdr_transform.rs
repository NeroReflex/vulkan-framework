use std::sync::{Arc, Mutex};

use inline_spirv::*;

use vulkan_framework::{
    clear_values::ColorClearValues,
    command_buffer::CommandBufferRecorder,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor,
        DescriptorPoolSizesAcceletarionStructureKHR, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    dynamic_rendering::{
        AttachmentStoreOp, DynamicRendering, DynamicRenderingColorAttachment,
        RenderingAttachmentSetup,
    },
    graphics_pipeline::{
        CullMode, DepthCompareOp, DepthConfiguration, FrontFace, GraphicsPipeline, PolygonMode,
        Rasterizer, Scissor, Viewport,
    },
    image::{
        CommonImageFormat, ConcreteImageDescriptor, Image, Image2DDimensions, ImageFlags,
        ImageFormat, ImageLayout, ImageMultisampling, ImageTiling, ImageUseAs,
    },
    image_view::{ImageView, ImageViewType},
    memory_barriers::{ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    memory_heap::MemoryType,
    memory_management::{MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait},
    memory_pool::MemoryPoolFeatures,
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

/*
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

const mat3 aces_input_matrix =
mat3(
    vec3(0.59719f, 0.35458f, 0.04823f),
    vec3(0.07600f, 0.90834f, 0.01566f),
    vec3(0.02840f, 0.13383f, 0.83777f)
);

const mat3 aces_output_matrix =
mat3(
    vec3(1.60475f, -0.53108f, -0.07367f),
    vec3(-0.10208f, 1.10813f, -0.00605f),
    vec3(-0.00327f, -0.07276f,  1.07602f)
);

vec3 rtt_and_odt_fit(vec3 v)
{
    vec3 a = v * (v + 0.0245786f) - 0.000090537f;
    vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

vec3 aces_fitted(vec3 v)
{
    v = aces_input_matrix * v;
    v = rtt_and_odt_fit(v);
    return aces_output_matrix * v;
}

void main() {
    const vec3 input_color = texture(src, in_vTextureUV).xyz;
    outColor = vec4(aces_fitted(input_color), 1.0) /* * (hdr.gamma + hdr.exposure) */
;
}
"#,
    glsl,
    frag,
    vulkan1_0,
    entry = "main"
);
*/

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

vec3 uncharted2_tonemap_partial(vec3 x)
{
    float A = 0.15f;
    float B = 0.50f;
    float C = 0.10f;
    float D = 0.20f;
    float E = 0.02f;
    float F = 0.30f;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

vec3 uncharted2_filmic(vec3 v)
{
    float exposure_bias = 2.0f;
    vec3 curr = uncharted2_tonemap_partial(v * exposure_bias);

    vec3 W = vec3(11.2f);
    vec3 white_scale = vec3(1.0f) / uncharted2_tonemap_partial(W);
    return curr * white_scale;
}

void main() {
    const vec3 input_color = texture(src, in_vTextureUV).xyz;
    outColor = vec4(uncharted2_tonemap_partial(input_color), 1.0) /* * (hdr.gamma + hdr.exposure) */;
}
"#,
    glsl,
    frag,
    vulkan1_0,
    entry = "main"
);

pub struct HDRTransform {
    descriptor_sets: smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    push_constant_size: u32,
    push_constant_access: ShaderStagesAccess,
    pipeline_layout: Arc<PipelineLayout>,
    graphics_pipeline: Arc<GraphicsPipeline>,

    descriptor_set_layout: Arc<DescriptorSetLayout>,

    image_dimensions: Image2DDimensions,
    image_views: smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    sampler: Arc<Sampler>,
}

/// Represents the stage of the pipeline that preceeds the final rendering:
/// the one that transform the raw result into something that can be sampled
/// to generate the image to be presented: applies tone mapping and/or hdr.
impl HDRTransform {
    #[inline]
    fn image_output_format() -> ImageFormat {
        CommonImageFormat::r32g32b32a32_sfloat.into()
    }

    /// Returns the descriptor set layout for the hdr image.
    #[inline(always)]
    pub fn descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.descriptor_set_layout.clone()
    }

    pub fn new(
        memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,
        frames_in_flight: u32,
        render_area: &RenderingDimensions,
    ) -> RenderingResult<Self> {
        let mut mem_manager = memory_manager.lock().unwrap();

        let device = mem_manager.get_parent_device();

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

        let mut image_handles = vec![];
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

            image_handles.push(image.into());
        }

        let mut image_views: smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]> =
            smallvec::smallvec![];
        for (index, image) in mem_manager
            .allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::new(false),
                image_handles,
                MemoryManagementTags::default()
                    .with_name("hdr_transform".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?
            .into_iter()
            .enumerate()
        {
            let image_view_name = format!("hdr_transform.image_view[{index}]");
            let image_view = ImageView::new(
                image.image(),
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
            descriptor_sets,

            push_constant_size,
            push_constant_access,
            pipeline_layout,
            graphics_pipeline,

            descriptor_set_layout,

            image_dimensions,
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
    ) -> Arc<ImageView> {
        self.descriptor_sets[current_frame]
            .bind_resources(|binder| {
                binder
                    .bind_combined_images_samplers(
                        0,
                        &[(
                            ImageLayout::ShaderReadOnlyOptimal,
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
        recorder.pipeline_barriers([ImageMemoryBarrier::new(
            PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
            MemoryAccess::from([].as_slice()),
            PipelineStages::from([PipelineStage::AllGraphics].as_slice()),
            MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
            image_view.image().into(),
            ImageLayout::Undefined,
            ImageLayout::ColorAttachmentOptimal,
            queue_family.clone(),
            queue_family.clone(),
        )
        .into()]);

        let rendering_color_attachments = [DynamicRenderingColorAttachment::new(
            self.image_views[current_frame].clone(),
            RenderingAttachmentSetup::clear(ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0)),
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
                    { std::mem::size_of_val(&push_constant) },
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

        // Insert a barrier to transition image layout from the final rendering output to renderquad input
        // while also ensuring the rendering operation of final rendering pipeline has completed before initiating
        // the final renderquad step.
        recorder.pipeline_barriers([ImageMemoryBarrier::new(
            PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
            MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
            PipelineStages::from([PipelineStage::FragmentShader].as_slice()),
            MemoryAccess::from([MemoryAccessAs::ShaderRead].as_slice()),
            image_view.image().into(),
            ImageLayout::ColorAttachmentOptimal,
            ImageLayout::ShaderReadOnlyOptimal,
            queue_family.clone(),
            queue_family.clone(),
        )
        .into()]);

        image_view
    }
}
