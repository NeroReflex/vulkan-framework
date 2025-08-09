use std::sync::Arc;

use inline_spirv::*;

use crate::rendering::{
    MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingResult, rendering_dimensions::RenderingDimensions,
    resources::object::Manager,
};

use vulkan_framework::{
    command_buffer::{ClearValues, ColorClearValues, CommandBufferRecorder},
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
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
        ImageMultisampling, ImageTiling, ImageUsage, ImageUseAs,
    },
    image_view::{ImageView, ImageViewType},
    memory_allocator::{DefaultAllocator, MemoryAllocator},
    memory_barriers::{ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    memory_heap::{ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryRequirements, MemoryType},
    memory_pool::{MemoryPool, MemoryPoolFeatures},
    memory_requiring::AllocationRequiring,
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

layout (location = 0) out vec4 out_vPosition_worldspace_minus_eye_position;
layout (location = 1) out vec4 out_vNormal_worldspace;
layout (location = 2) out vec2 out_vTextureUV;

layout(location = 4) out flat vec4 eyePosition_worldspace;

layout(std140, set = 2, binding = 0) uniform camera_uniform {
	mat4 viewMatrix;
	mat4 projectionMatrix;
} camera;

void main() {
    const mat4 ModelMatrix = mat4(ModelMatrix_first_row, ModelMatrix_second_row, ModelMatrix_third_row, vec4(0.0, 0.0, 0.0, 1.0));

    const mat4 MVP = camera.projectionMatrix * camera.viewMatrix * ModelMatrix; 

    // Get the eye position
	const vec4 eye_position = vec4(camera.viewMatrix[3][0], camera.viewMatrix[3][1], camera.viewMatrix[3][2], 1.0);
	eyePosition_worldspace = eye_position;

	vec4 vPosition_worldspace = ModelMatrix * vec4(vertex_position_modelspace, 1.0);
	vPosition_worldspace /= vPosition_worldspace.w;

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

layout (location = 0) in vec4 in_vPosition_worldspace_minus_eye_position;
layout (location = 1) in vec4 in_vNormal_worldspace;
layout (location = 2) in vec2 in_vTextureUV;

layout(location = 4) in flat vec4 in_eyePosition_worldspace;

layout(push_constant) uniform MaterialIDs {
    uint material_id;
} material;

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

layout(std430, set = 1, binding = 0) readonly buffer meshes
{
    mesh_to_material_t map[];
};

void main() {
    // Calculate position of the current fragment
    vec4 vPosition_worldspace = vec4((in_vPosition_worldspace_minus_eye_position + in_eyePosition_worldspace).xyz, 1.0);

    vec3 dFdxPos = dFdx( in_vPosition_worldspace_minus_eye_position.xyz );
	vec3 dFdyPos = dFdy( in_vPosition_worldspace_minus_eye_position.xyz );
	const vec3 facenormal = cross(dFdxPos, dFdyPos);

    // The normal can either be calculated or provided from the mesh. Just pick the provided one if it is valid.
    const vec3 bestNormal = normalize(length(in_vNormal_worldspace.xyz) < 0.000001f ? facenormal : in_vNormal_worldspace.xyz);

    const uint diffuse_texture_index = info[material.material_id].diffuse_texture_index;

    // in OpenGL depth is in range [-1;+1], while in vulkan it is [0.0;1.0]
    // see https://docs.vulkan.org/guide/latest/depth.html "Porting from OpenGL"
    vPosition_worldspace.z = (vPosition_worldspace.z + vPosition_worldspace.w) * 0.5;

    out_vPosition = vPosition_worldspace;
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

    push_constants_access: ShaderStagesAccess,
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
    fn output_image_color_layout() -> ImageLayout {
        ImageLayout::ColorAttachmentOptimal
    }

    fn output_image_depth_stencil_layout() -> ImageLayout {
        ImageLayout::DepthStencilAttachmentOptimal
    }

    fn output_image_color_format() -> ImageFormat {
        ImageFormat::from(CommonImageFormat::r32g32b32a32_sfloat)
    }

    fn output_image_depth_stencil_format() -> ImageFormat {
        ImageFormat::from(CommonImageFormat::d32_sfloat_s8_uint)
    }

    /// Returns the descriptor set layout for the gbuffer
    pub fn descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.gbuffer_descriptor_set_layout.clone()
    }

    pub fn new(
        device: Arc<Device>,
        textures_descriptor_set_layout: Arc<DescriptorSetLayout>,
        materials_descriptor_set_layout: Arc<DescriptorSetLayout>,
        view_projection_descriptor_set_layout: Arc<DescriptorSetLayout>,
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
            .map(|obj| {
                obj.allocation_requirements().size() + obj.allocation_requirements().alignment()
            })
            .chain(gbuffer_position_image_handles.iter().map(|obj| {
                obj.allocation_requirements().size() + obj.allocation_requirements().alignment()
            }))
            .chain(gbuffer_normal_image_handles.iter().map(|obj| {
                obj.allocation_requirements().size() + obj.allocation_requirements().alignment()
            }))
            .chain(gbuffer_texture_image_handles.iter().map(|obj| {
                obj.allocation_requirements().size() + obj.allocation_requirements().alignment()
            }))
            .sum();

        let framebuffer_memory_pool = {
            let image_handles: Vec<&dyn AllocationRequiring> = gbuffer_depth_stencil_image_handles
                .iter()
                .map(|obj| obj as &dyn AllocationRequiring)
                .chain(
                    gbuffer_position_image_handles
                        .iter()
                        .map(|obj| obj as &dyn AllocationRequiring),
                )
                .chain(
                    gbuffer_normal_image_handles
                        .iter()
                        .map(|obj| obj as &dyn AllocationRequiring),
                )
                .chain(
                    gbuffer_texture_image_handles
                        .iter()
                        .map(|obj| obj as &dyn AllocationRequiring),
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
            let allocated_image = AllocatedImage::new(framebuffer_memory_pool.clone(), image)?;
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
            let allocated_image = AllocatedImage::new(framebuffer_memory_pool.clone(), image)?;
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
            let allocated_image = AllocatedImage::new(framebuffer_memory_pool.clone(), image)?;
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
            let allocated_image = AllocatedImage::new(framebuffer_memory_pool.clone(), image)?;
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
                        ShaderStagesAccess::graphics(),
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
                ShaderStagesAccess::graphics(),
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

        let push_constants_access = [ShaderStageAccessIn::Fragment].as_slice().into();

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            [
                textures_descriptor_set_layout,
                materials_descriptor_set_layout,
                view_projection_descriptor_set_layout,
            ]
            .as_slice(),
            [PushConstanRange::new(0, 4u32, push_constants_access)].as_slice(),
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
                DepthCompareOp::Less,
                Some((0.0, 1.0)),
            )),
            None,
            None,
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
                    4u32 * 12u32,
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

            push_constants_access,
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
                    PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
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
                    PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
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
                    PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
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
                    PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                    MemoryAccess::from([].as_slice()),
                    PipelineStages::from([PipelineStage::EarlyFragmentTests].as_slice()),
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
            ClearValues::new(Some(ColorClearValues::Vec4(1.0, 1.0, 1.0, 1.0))),
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
                    4u32,
                    self.push_constants_access,
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
