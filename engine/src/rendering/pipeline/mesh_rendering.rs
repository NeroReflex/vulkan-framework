use std::sync::{Arc, Mutex};

use inline_spirv::*;

use crate::rendering::{
    MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingResult, rendering_dimensions::RenderingDimensions,
    resources::object::Manager,
};

use vulkan_framework::{
    clear_values::{ColorClearValues, DepthClearValues},
    command_buffer::CommandBufferRecorder,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    dynamic_rendering::{
        AttachmentStoreOp, DynamicRendering, DynamicRenderingColorAttachment,
        DynamicRenderingDepthAttachment, RenderingAttachmentSetup,
    },
    graphics_pipeline::{
        AttributeType, CullMode, DepthCompareOp, DepthConfiguration, FrontFace, GraphicsPipeline,
        PolygonMode, Rasterizer, Scissor, VertexInputAttribute, VertexInputBinding,
        VertexInputRate, Viewport,
    },
    image::{
        AllocatedImage, CommonImageFormat, ConcreteImageDescriptor, Image, Image1DTrait,
        Image2DDimensions, Image2DTrait, ImageDimensions, ImageFlags, ImageFormat, ImageLayout,
        ImageMultisampling, ImageTiling, ImageUsage, ImageUseAs,
    },
    image_view::{ImageView, ImageViewType},
    memory_barriers::{ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    memory_heap::MemoryType,
    memory_management::{MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait},
    memory_pool::MemoryPoolFeatures,
    pipeline_layout::{PipelineLayout, PipelineLayoutDependant},
    pipeline_stage::{PipelineStage, PipelineStages},
    push_constant_range::PushConstanRange,
    queue_family::QueueFamily,
    sampler::{Filtering, MipmapMode, Sampler},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::{ShaderStageAccessIn, ShaderStagesAccess},
    shaders::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
};

const MESH_RENDERING_VERTEX_SPV: &[u32] = inline_spirv!(
    r#"
#version 450 core

layout (location = 0) in vec3 vertex_position_modelspace;
layout (location = 1) in vec3 vertex_normal_modelspace;
layout (location = 2) in vec2 vertex_texture;

layout (location = 3) in vec4 ModelMatrix_first_row;
layout (location = 4) in vec4 ModelMatrix_second_row;
layout (location = 5) in vec4 ModelMatrix_third_row;

//layout (location = 3) in uint vMaterialIndex;

layout (location = 0) out vec4 out_vPosition_worldspace;
layout (location = 1) out vec4 out_vNormal_worldspace;
layout (location = 2) out vec2 out_vTextureUV;
layout (location = 3) out vec4 out_vPosition_worldspace_minus_eye_position;
layout (location = 4) out flat vec4 eyePosition_worldspace;

layout(std140, set = 2, binding = 0) uniform camera_uniform {
	mat4 viewMatrix;
	mat4 projectionMatrix;
} camera;

layout(push_constant) uniform MeshData {
    mat3x4 load_matrix;
    uint mesh_id;
} mesh_data;

void main() {
    const mat4 LoadMatrix = mat4(mesh_data.load_matrix[0], mesh_data.load_matrix[1], mesh_data.load_matrix[2], vec4(0.0, 0.0, 0.0, 1.0));
    const mat4 InstanceMatrix = mat4(ModelMatrix_first_row, ModelMatrix_second_row, ModelMatrix_third_row, vec4(0.0, 0.0, 0.0, 1.0));

    const mat4 ModelMatrix = InstanceMatrix * LoadMatrix;

    const mat4 MVP = camera.projectionMatrix * camera.viewMatrix * ModelMatrix; 

    // Get the eye position
	const vec4 eye_position = vec4(camera.viewMatrix[3][0], camera.viewMatrix[3][1], camera.viewMatrix[3][2], 1.0);
	eyePosition_worldspace = eye_position;

	vec4 vPosition_worldspace = ModelMatrix * vec4(vertex_position_modelspace, 1.0);
	vPosition_worldspace /= vPosition_worldspace.w;

    out_vPosition_worldspace = vPosition_worldspace;
	out_vTextureUV = vec2(vertex_texture.x, 1-vertex_texture.y);
	out_vPosition_worldspace_minus_eye_position = vec4((vPosition_worldspace - eyePosition_worldspace).xyz, 1.0);
	out_vNormal_worldspace = vec4((ModelMatrix * vec4(vertex_normal_modelspace, 0.0)).xyz, 0.0);

    gl_Position = MVP * vec4(vertex_position_modelspace, 1.0);
}
"#,
    vert
);

const MESH_RENDERING_FRAGMENT_SPV: &[u32] = inline_spirv!(
    r#"
#version 450 core

// ============================================ FRAGMENT OUTPUT ==================================================
layout (location = 0) out vec4 out_vPosition;           // Search for GBUFFER_FB0
layout (location = 1) out vec4 out_vNormal;             // Search for GBUFFER_FB1
layout (location = 2) out vec4 out_vDiffuse;            // Search for GBUFFER_FB2
// ===============================================================================================================

layout (location = 0) in vec4 in_vPosition_worldspace;
layout (location = 1) in vec4 in_vNormal_worldspace;
layout (location = 2) in vec2 in_vTextureUV;
layout (location = 3) in vec4 in_vPosition_worldspace_minus_eye_position;
layout (location = 4) in flat vec4 in_eyePosition_worldspace;

layout(push_constant) uniform MeshData {
    mat3x4 load_matrix;
    uint mesh_id;
} mesh_data;

// MUST match with MAX_TEXTURES on rust side
layout(set = 0, binding = 0) uniform sampler2D textures[256];

struct material_t {
    uint diffuse_texture_index;
    uint normal_texture_index;
    uint reflection_texture_index;
    uint displacement_texture_index;
};

struct mesh_to_material_t {
    uint material_index;
};

layout(std430, set = 1, binding = 0) readonly buffer material
{
    material_t info[];
};

layout(std430, set = 1, binding = 1) readonly buffer meshes
{
    mesh_to_material_t material_for_mesh[];
};

void main() {
    // Calculate position of the current fragment
    //vec4 vPosition_worldspace = vec4((in_vPosition_worldspace_minus_eye_position + in_eyePosition_worldspace).xyz, 1.0);

    vec3 dFdxPos = dFdx( in_vPosition_worldspace_minus_eye_position.xyz );
	vec3 dFdyPos = dFdy( in_vPosition_worldspace_minus_eye_position.xyz );
	const vec3 facenormal = cross(dFdxPos, dFdyPos);

    // The normal can either be calculated or provided from the mesh. Just pick the provided one if it is valid.
    const vec3 bestNormal = normalize(length(in_vNormal_worldspace.xyz) < 0.000001f ? facenormal : in_vNormal_worldspace.xyz);

    const uint material_id = material_for_mesh[mesh_data.mesh_id].material_index;
    const uint diffuse_texture_index = info[material_id].diffuse_texture_index;

    // in OpenGL depth is in range [-1;+1], while in vulkan it is [0.0;1.0]
    // see https://docs.vulkan.org/guide/latest/depth.html "Porting from OpenGL"
    //vPosition_worldspace.z = (vPosition_worldspace.z + vPosition_worldspace.w) * 0.5;

    out_vPosition = in_vPosition_worldspace;
    out_vNormal = vec4(bestNormal.xyz, 0.0);
    out_vDiffuse = texture(textures[diffuse_texture_index], in_vTextureUV);
}
"#,
    frag
);

/// Represents the stage of the pipeline responsible for drawing static meshes.
pub struct MeshRendering {
    image_dimensions: Image2DDimensions,

    gbuffer_descriptor_set_layout: Arc<DescriptorSetLayout>,
    gbuffer_descriptor_sets:
        smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    push_constants_stages: ShaderStagesAccess,
    graphics_pipeline: Arc<GraphicsPipeline>,

    gbuffer_depth_stencil_image_views:
        smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    gbuffer_position_image_views:
        smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    gbuffer_normal_image_views:
        smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    gbuffer_texture_image_views:
        smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
}

impl MeshRendering {
    #[inline(always)]
    fn output_image_color_layout() -> ImageLayout {
        ImageLayout::ColorAttachmentOptimal
    }

    #[inline(always)]
    fn output_image_depth_stencil_layout() -> ImageLayout {
        ImageLayout::DepthStencilAttachmentOptimal
    }

    #[inline(always)]
    fn output_image_color_format() -> ImageFormat {
        ImageFormat::from(CommonImageFormat::r32g32b32a32_sfloat)
    }

    #[inline(always)]
    fn output_image_depth_stencil_format() -> ImageFormat {
        ImageFormat::from(CommonImageFormat::d32_sfloat_s8_uint)
    }

    /// Returns the descriptor set layout for the gbuffer
    #[inline(always)]
    pub fn descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.gbuffer_descriptor_set_layout.clone()
    }

    pub fn new(
        memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,
        textures_descriptor_set_layout: Arc<DescriptorSetLayout>,
        materials_descriptor_set_layout: Arc<DescriptorSetLayout>,
        view_projection_descriptor_set_layout: Arc<DescriptorSetLayout>,
        render_area: &RenderingDimensions,
        frames_in_flight: u32,
    ) -> RenderingResult<Self> {
        let mut mem_manager = memory_manager.lock().unwrap();

        let device = mem_manager.get_parent_device();

        let image_dimensions = render_area.into();

        let mut gbuffer_depth_stencil_image_handles = vec![];
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

            gbuffer_depth_stencil_image_handles.push(image.into());
        }

        let mut gbuffer_position_image_handles = vec![];
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

            gbuffer_position_image_handles.push(image.into());
        }

        let mut gbuffer_normal_image_handles = vec![];
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

            gbuffer_normal_image_handles.push(image.into());
        }

        let mut gbuffer_texture_image_handles = vec![];
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

            gbuffer_texture_image_handles.push(image.into());
        }

        let mut gbuffer_depth_stencil_image_views: smallvec::SmallVec<
            [Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        for (index, image) in mem_manager
            .allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::new(false),
                gbuffer_depth_stencil_image_handles,
                MemoryManagementTags::default()
                    .with_name("mesh_rendering".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?
            .into_iter()
            .enumerate()
        {
            let image_view_name =
                format!("mesh_rendering.gbuffer_depth_stencil_image_views[{index}]");
            gbuffer_depth_stencil_image_views.push(ImageView::new(
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
            )?);
        }

        let mut gbuffer_position_image_views: smallvec::SmallVec<
            [Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        let mut gbuffer_position_images: smallvec::SmallVec<
            [Arc<AllocatedImage>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        for (index, image) in mem_manager
            .allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::new(false),
                gbuffer_position_image_handles,
                MemoryManagementTags::default()
                    .with_name("mesh_rendering".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?
            .into_iter()
            .enumerate()
        {
            let allocated_image = image.image();
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
        for (index, image) in mem_manager
            .allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::new(false),
                gbuffer_normal_image_handles,
                MemoryManagementTags::default()
                    .with_name("mesh_rendering".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?
            .into_iter()
            .enumerate()
        {
            let allocated_image = image.image();
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
        for (index, image) in mem_manager
            .allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::new(false),
                gbuffer_texture_image_handles,
                MemoryManagementTags::default()
                    .with_name("mesh_rendering".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?
            .into_iter()
            .enumerate()
        {
            let allocated_image = image.image();
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

        let gbuffer_descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    3u32 * frames_in_flight,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    None,
                ),
                frames_in_flight,
            ),
            Some("mesh_rendering.gbuffer_descriptor_pool"),
        )?;

        let gbuffer_sampler = Sampler::new(
            device.clone(),
            Filtering::Linear,
            Filtering::Linear,
            MipmapMode::ModeLinear,
            1.0,
        )?;

        let mut gbuffer_descriptor_sets =
            smallvec::SmallVec::<_>::with_capacity(frames_in_flight as usize);
        for frame_index in 0..(frames_in_flight as usize) {
            let descriptor_set = DescriptorSet::new(
                gbuffer_descriptor_pool.clone(),
                DescriptorSetLayout::new(
                    device.clone(),
                    [BindingDescriptor::new(
                        [ShaderStageAccessIn::RayTracing(vulkan_framework::shader_stage_access::ShaderStageAccessInRayTracingKHR::RayGen), ShaderStageAccessIn::Fragment].as_slice().into(),
                        BindingType::Native(NativeBindingType::CombinedImageSampler),
                        0,
                        3u32,
                    )]
                    .as_slice(),
                )?,
            )?;

            descriptor_set.bind_resources(|binder| {
                binder
                    .bind_combined_images_samplers(
                        0,
                        [
                            (
                                ImageLayout::ShaderReadOnlyOptimal,
                                gbuffer_position_image_views[frame_index].clone(),
                                gbuffer_sampler.clone(),
                            ),
                            (
                                ImageLayout::ShaderReadOnlyOptimal,
                                gbuffer_normal_image_views[frame_index].clone(),
                                gbuffer_sampler.clone(),
                            ),
                            (
                                ImageLayout::ShaderReadOnlyOptimal,
                                gbuffer_texture_image_views[frame_index].clone(),
                                gbuffer_sampler.clone(),
                            ),
                        ]
                        .as_slice(),
                    )
                    .unwrap();
            })?;

            gbuffer_descriptor_sets.push(descriptor_set);
        }

        let gbuffer_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [BindingDescriptor::new(
                [ShaderStageAccessIn::RayTracing(vulkan_framework::shader_stage_access::ShaderStageAccessInRayTracingKHR::RayGen), ShaderStageAccessIn::Fragment].as_slice().into(),
                BindingType::Native(NativeBindingType::CombinedImageSampler),
                0,
                3,
            )]
            .as_slice(),
        )?;

        let vertex_shader =
            VertexShader::new(device.clone(), &[], &[], MESH_RENDERING_VERTEX_SPV).unwrap();

        let fragment_shader =
            FragmentShader::new(device.clone(), &[], &[], MESH_RENDERING_FRAGMENT_SPV).unwrap();

        let push_constants_stages = [ShaderStageAccessIn::Vertex, ShaderStageAccessIn::Fragment]
            .as_slice()
            .into();
        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            [
                textures_descriptor_set_layout,
                materials_descriptor_set_layout,
                view_projection_descriptor_set_layout,
            ]
            .as_slice(),
            [PushConstanRange::new(0, 52u32, push_constants_stages)].as_slice(),
            Some("mesh_rendering.pipeline_layout"),
        )?;

        let graphics_pipeline =
            GraphicsPipeline::new(
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
                    DepthCompareOp::Less,
                    Some((0.0, 1.0)),
                )),
                Some(Viewport::new(
                    0.0f32,
                    0.0f32,
                    image_dimensions.width() as f32,
                    image_dimensions.height() as f32,
                    0.0f32,
                    1.0f32,
                )),
                Some(Scissor::new(0, 0, image_dimensions)),
                pipeline_layout,
                [
                    VertexInputBinding::new(
                        VertexInputRate::PerVertex,
                        (4u32) * (3u32 + 3u32 + 2u32),
                        [
                            // vertex position data
                            VertexInputAttribute::new(0, 0, AttributeType::Vec3),
                            // vertex normal data
                            VertexInputAttribute::new(1, 4 * 3, AttributeType::Vec3),
                            // vertex text coords
                            VertexInputAttribute::new(2, 4 * (3 + 3), AttributeType::Vec2),
                        ]
                        .as_slice(),
                    ),
                    VertexInputBinding::new(
                        VertexInputRate::PerInstance,
                        // mat 4x3
                        core::mem::size_of::<
                            vulkan_framework::ash::vk::AccelerationStructureInstanceKHR,
                        >() as u32,
                        [
                            // model matrix first row
                            VertexInputAttribute::new(3, 0, AttributeType::Vec4),
                            // model matrix second row
                            VertexInputAttribute::new(4, 4 * 4, AttributeType::Vec4),
                            // model matrix third row
                            VertexInputAttribute::new(5, 4 * 8, AttributeType::Vec4),
                        ]
                        .as_slice(),
                    ),
                ]
                .as_slice(),
                Rasterizer::new(
                    PolygonMode::Fill,
                    FrontFace::Clockwise,
                    CullMode::Back,
                    None,
                ),
                (vertex_shader, None),
                (fragment_shader, None),
                Some("mesh_rendering.graphics_pipeline"),
            )?;

        Ok(Self {
            image_dimensions,

            gbuffer_descriptor_set_layout,
            gbuffer_descriptor_sets,

            push_constants_stages,
            graphics_pipeline,

            gbuffer_depth_stencil_image_views,
            gbuffer_position_image_views,
            gbuffer_normal_image_views,
            gbuffer_texture_image_views,
        })
    }

    /// Record commands used to rendering to the gbuffer, that will be returned
    ///
    /// gbuffer_stages and gbuffer_access are used to build an image barrier for
    /// images on the gbuffer: the caller must tell this function how these images
    /// will be used
    pub fn record_rendering_commands<ManagerT>(
        &self,
        view_projection_descriptor_set: Arc<DescriptorSet>,
        queue_family: Arc<QueueFamily>,
        gbuffer_stages: PipelineStages,
        gbuffer_access: MemoryAccess,
        current_frame: usize,
        meshes: ManagerT,
        recorder: &mut CommandBufferRecorder,
    ) -> Arc<DescriptorSet>
    where
        ManagerT: std::ops::Deref<Target = Manager>,
    {
        let position_imageview = self.gbuffer_position_image_views[current_frame].clone();
        let normal_imageview = self.gbuffer_normal_image_views[current_frame].clone();
        let texture_imageview = self.gbuffer_texture_image_views[current_frame].clone();
        let depth_stencil_imageview = self.gbuffer_depth_stencil_image_views[current_frame].clone();

        // update materials descriptor sets (to make them relevants to this frame)
        meshes.update_buffers(recorder, current_frame, queue_family.clone());

        // Transition the framebuffer images into depth/color attachment optimal layout,
        // so that the graphics pipeline has it in the best format
        recorder.image_barriers(
            [
                ImageMemoryBarrier::new(
                    PipelineStages::from([].as_slice()),
                    MemoryAccess::from([].as_slice()),
                    PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                    MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                    position_imageview.image().into(),
                    ImageLayout::Undefined,
                    Self::output_image_color_layout(),
                    queue_family.clone(),
                    queue_family.clone(),
                ),
                ImageMemoryBarrier::new(
                    PipelineStages::from([].as_slice()),
                    MemoryAccess::from([].as_slice()),
                    PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                    MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                    normal_imageview.image().into(),
                    ImageLayout::Undefined,
                    Self::output_image_color_layout(),
                    queue_family.clone(),
                    queue_family.clone(),
                ),
                ImageMemoryBarrier::new(
                    PipelineStages::from([].as_slice()),
                    MemoryAccess::from([].as_slice()),
                    PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                    MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                    texture_imageview.image().into(),
                    ImageLayout::Undefined,
                    Self::output_image_color_layout(),
                    queue_family.clone(),
                    queue_family.clone(),
                ),
                ImageMemoryBarrier::new(
                    PipelineStages::from([].as_slice()),
                    MemoryAccess::from([].as_slice()),
                    PipelineStages::from(
                        [
                            PipelineStage::EarlyFragmentTests,
                            PipelineStage::LateFragmentTests,
                        ]
                        .as_slice(),
                    ),
                    MemoryAccess::from(
                        [
                            MemoryAccessAs::DepthStencilAttachmentWrite,
                            MemoryAccessAs::DepthStencilAttachmentRead,
                        ]
                        .as_slice(),
                    ),
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
            DynamicRenderingColorAttachment::new(
                position_imageview.clone(),
                RenderingAttachmentSetup::clear(ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0)),
                AttachmentStoreOp::Store,
            ),
            DynamicRenderingColorAttachment::new(
                normal_imageview.clone(),
                RenderingAttachmentSetup::clear(ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0)),
                AttachmentStoreOp::Store,
            ),
            DynamicRenderingColorAttachment::new(
                texture_imageview.clone(),
                RenderingAttachmentSetup::clear(ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0)),
                AttachmentStoreOp::Store,
            ),
        ];
        let rendering_depth_attachment = DynamicRenderingDepthAttachment::new(
            depth_stencil_imageview.clone(),
            RenderingAttachmentSetup::clear(DepthClearValues::new(1.0)),
            AttachmentStoreOp::Store,
        );
        recorder.graphics_rendering(
            self.image_dimensions,
            rendering_color_attachments.as_slice(),
            Some(&rendering_depth_attachment),
            None,
            |recorder| {
                recorder.bind_graphics_pipeline(self.graphics_pipeline.clone(), None, None);

                // bind the view-projection matrix
                recorder.bind_descriptor_sets_for_graphics_pipeline(
                    self.graphics_pipeline.get_parent_pipeline_layout(),
                    2,
                    [view_projection_descriptor_set].as_slice(),
                );

                // performs the actual rendering
                meshes.deref().guided_rendering(
                    recorder,
                    current_frame,
                    self.graphics_pipeline.get_parent_pipeline_layout(),
                    0,
                    1,
                    0,
                    self.push_constants_stages,
                );
            },
        );

        // gbuffers are bound to a descriptor set and will be used along the pipeline that way:
        // place image barriers to transition them to the right format
        recorder.image_barriers(
            [
                ImageMemoryBarrier::new(
                    PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                    MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                    gbuffer_stages,
                    gbuffer_access,
                    position_imageview.image().into(),
                    Self::output_image_color_layout(),
                    ImageLayout::ShaderReadOnlyOptimal,
                    queue_family.clone(),
                    queue_family.clone(),
                ),
                ImageMemoryBarrier::new(
                    PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                    MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                    gbuffer_stages,
                    gbuffer_access,
                    normal_imageview.image().into(),
                    Self::output_image_color_layout(),
                    ImageLayout::ShaderReadOnlyOptimal,
                    queue_family.clone(),
                    queue_family.clone(),
                ),
                ImageMemoryBarrier::new(
                    PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                    MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                    gbuffer_stages,
                    gbuffer_access,
                    texture_imageview.image().into(),
                    Self::output_image_color_layout(),
                    ImageLayout::ShaderReadOnlyOptimal,
                    queue_family.clone(),
                    queue_family.clone(),
                ),
            ]
            .as_slice(),
        );

        self.gbuffer_descriptor_sets[current_frame].clone()
    }
}
