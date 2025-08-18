use std::{
    ops::DerefMut,
    sync::{Arc, Mutex},
};

use inline_spirv::inline_spirv;
use vulkan_framework::{
    binding_tables::RaytracingBindingTables,
    clear_values::ColorClearValues,
    command_buffer::CommandBufferRecorder,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::DeviceOwned,
    image::{
        CommonImageFormat, ConcreteImageDescriptor, Image, Image3DDimensions, ImageFlags,
        ImageFormat, ImageLayout, ImageMultisampling, ImageSubresourceRange, ImageTiling,
        ImageUseAs,
    },
    image_view::ImageView,
    memory_barriers::{ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    memory_heap::MemoryType,
    memory_management::{MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait},
    memory_pool::MemoryPoolFeatures,
    pipeline_layout::{PipelineLayout, PipelineLayoutDependant},
    pipeline_stage::{PipelineStage, PipelineStageRayTracingPipelineKHR, PipelineStages},
    queue_family::QueueFamily,
    raytracing_pipeline::RaytracingPipeline,
    sampler::{Filtering, MipmapMode, Sampler},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::{ShaderStageAccessIn, ShaderStageAccessInRayTracingKHR},
    shaders::{
        closest_hit_shader::ClosestHitShader, miss_shader::MissShader, raygen_shader::RaygenShader,
    },
};

use crate::rendering::{
    MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingResult, rendering_dimensions::RenderingDimensions,
};

const RAYGEN_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_flags_primitive_culling : require
#extension GL_EXT_nonuniform_qualifier : enable

layout (set = 0, binding = 0, std430) readonly buffer tlas_instances
{
    uint data[];
};

layout (set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

// gbuffer: 0 for position, 1 for normal, 2 for diffuse texture
layout (set = 1, binding = 0) uniform sampler2D gbuffer[3];

layout(push_constant) uniform DirectionalLightingData {
    uint lights_count;
} directional_lighting_data;

layout(std140, set = 2, binding = 0) uniform camera_uniform {
	mat4 viewMatrix;
	mat4 projectionMatrix;
} camera;

uniform layout (set = 3, binding = 0, rgba32f) image2D outputImage;

struct hit_payload_t {
    bool hit;
};

layout(location = 0) rayPayloadEXT hit_payload_t payload;

void main() {
    const ivec2 resolution = imageSize(outputImage);

    const vec2 position_xy = vec2(float(gl_LaunchIDEXT.x) / float(resolution.x), float(gl_LaunchIDEXT.y) / float(resolution.y));

    const vec3 origin = texture(gbuffer[0], position_xy).xyz;
    const vec3 normal = texture(gbuffer[1], position_xy).xyz;

    /*
        // other flags: gl_RayFlagsCullNoOpaqueEXT gl_RayFlagsNoneEXT
        traceRayEXT(topLevelAS, gl_RayFlagsSkipAABBEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xff, 0, 0, 0, origin.xyz, 0.1, ray_dir.xyz, 10000.0, 0);
    */

    imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), vec4(0.7, 0.7, 0.7, 1.0));
}
"#,
    glsl,
    rgen,
    vulkan1_2,
    entry = "main"
);

const MISS_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require

struct hit_payload_t {
    bool hit;
};

layout(location = 0) rayPayloadEXT hit_payload_t payload;

void main() {
    payload.hit = false;
}
"#,
    glsl,
    rmiss,
    vulkan1_2,
    entry = "main"
);

const CHIT_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#define Buffer(Alignment) \
  layout(buffer_reference, scalar) buffer

struct vertex_buffer_element_t {
    vec3 position;
    vec3 normal;
    vec2 texture_uv;
};

Buffer(64) VertexBuffer {
  vertex_buffer_element_t vertex_data[ /* address by IndexBuffer */ ];
};

Buffer(64) IndexBuffer {
  uvec3 vertex_index[ /* address by gl_PrimitiveID */ ];
};

Buffer(64) TransformBuffer {
  mat3x4 transform[ /* address by  */ ];
};

struct instance_buffer_t {
    mat3x4 model_matrix;
};

Buffer(64) InstanceBuffer {
  instance_buffer_t transform[ /* address by gl_InstanceID */ ];
};

struct tlas_instance_data_t {
    IndexBuffer ib;
    VertexBuffer vb;
    TransformBuffer tb;
    InstanceBuffer instance;
};

layout(std430, set = 0, binding = 0) readonly buffer tlas_instances
{
    tlas_instance_data_t data[ /* address by gl_InstanceID */ ];
};

struct hit_payload_t {
    bool hit;
};

layout(location = 0) rayPayloadEXT hit_payload_t payload;

hitAttributeEXT vec2 attribs;

/*
 * gl_CustomInstanceID => ci ho messo il numero della mesh per ricondurmi al materiale giusto
 * gl_InstanceID => il numero della istanza: ogni istanza ha un numero progressivo che parte da 0
 * gl_PrimitiveID => l'indice del triangolo colpito
 */

void main() {
    payload.hit = true;
    //const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    //hitValue = barycentricCoords;
}
"#,
    glsl,
    rchit,
    vulkan1_2,
    entry = "main"
);

type GIBuffersType = smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

type RaytracingSBTType =
    smallvec::SmallVec<[Arc<RaytracingBindingTables>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

pub struct GILighting {
    queue_family: Arc<QueueFamily>,

    raytracing_pipeline: Arc<RaytracingPipeline>,

    raytracing_gibuffer: GIBuffersType,

    raytracing_sbts: RaytracingSBTType,

    output_descriptor_sets:
        smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    gibuffer_descriptor_set_layout: Arc<DescriptorSetLayout>,
    gibuffer_descriptor_sets:
        smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    renderarea_width: u32,
    renderarea_height: u32,
}

impl GILighting {
    /// Returns the descriptor set layout for the gbuffer
    #[inline(always)]
    pub fn descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.gibuffer_descriptor_set_layout.clone()
    }

    pub fn new(
        queue_family: Arc<QueueFamily>,
        render_area: &RenderingDimensions,
        memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,
        rt_descriptor_set_layout: Arc<DescriptorSetLayout>,
        gbuffer_descriptor_set_layout: Arc<DescriptorSetLayout>,
        dlbuffer_descriptor_set_layout: Arc<DescriptorSetLayout>,
        frames_in_flight: u32,
    ) -> RenderingResult<Self> {
        let device = queue_family.get_parent_device();

        let output_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [
                // outputImage
                BindingDescriptor::new(
                    [ShaderStageAccessIn::RayTracing(
                        ShaderStageAccessInRayTracingKHR::RayGen,
                    )]
                    .as_slice()
                    .into(),
                    BindingType::Native(NativeBindingType::StorageImage),
                    0,
                    1,
                ),
            ]
            .as_slice(),
        )?;

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            [
                rt_descriptor_set_layout,
                gbuffer_descriptor_set_layout,
                dlbuffer_descriptor_set_layout,
                output_descriptor_set_layout.clone(),
            ]
            .as_slice(),
            [].as_slice(),
            Some("gi_lighting_pipeline_layout"),
        )?;

        let raygen_shader = RaygenShader::new(device.clone(), RAYGEN_SPV)?;
        let miss_shader = MissShader::new(device.clone(), MISS_SPV).unwrap();
        //let anyhit_shader = AnyHitShader::new(dev.clone(), AHIT_SPV).unwrap();
        let closesthit_shader = ClosestHitShader::new(device.clone(), CHIT_SPV).unwrap();
        //let callable_shader = CallableShader::new(dev.clone(), CALLABLE_SPV).unwrap();

        let raytracing_pipeline = RaytracingPipeline::new(
            pipeline_layout.clone(),
            1,
            raygen_shader,
            None,
            miss_shader,
            None,
            closesthit_shader,
            None,
            Some("gi_lighting_raytracing_pipeline!"),
        )?;

        let mut raytracing_ldbuffer_unallocated = vec![];
        for frame_index in 0..frames_in_flight {
            raytracing_ldbuffer_unallocated.push(
                Image::new(
                    device.clone(),
                    ConcreteImageDescriptor::new(
                        render_area.into(),
                        [
                            ImageUseAs::TransferDst,
                            ImageUseAs::Storage,
                            ImageUseAs::Sampled,
                        ]
                        .as_slice()
                        .into(),
                        ImageMultisampling::SamplesPerPixel1,
                        1,
                        1,
                        ImageFormat::from(CommonImageFormat::r32g32b32a32_sfloat),
                        ImageFlags::empty(),
                        ImageTiling::Optimal,
                    ),
                    None,
                    Some(format!("raytracing_global_illumination_image[{frame_index}]").as_str()),
                )?
                .into(),
            );
        }

        let (raytracing_gibuffer, raytracing_sbts) = {
            let mut mem_manager = memory_manager.lock().unwrap();

            let raytracing_gibuffer_allocated = mem_manager.allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::new(false),
                raytracing_ldbuffer_unallocated,
                MemoryManagementTags::default()
                    .with_name("gi_lighting".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?;

            let raytracing_sbts = (0..frames_in_flight)
                .map(|_frame_index| {
                    RaytracingBindingTables::new(
                        raytracing_pipeline.clone(),
                        mem_manager.deref_mut(),
                        MemoryManagementTags::default()
                            .with_name("global_illumination_lighting".to_string())
                            .with_size(MemoryManagementTagSize::MediumSmall),
                    )
                    .unwrap()
                })
                .collect::<RaytracingSBTType>();

            let mut raytracing_gibuffer = GIBuffersType::with_capacity(frames_in_flight as usize);
            for frame_index in 0..(frames_in_flight as usize) {
                raytracing_gibuffer.push(ImageView::new(
                    raytracing_gibuffer_allocated[frame_index].image(),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(
                        format!("raytracing_global_illumination_image_imageview[{frame_index}]")
                            .as_str(),
                    ),
                )?);
            }

            (raytracing_gibuffer, raytracing_sbts)
        };

        let output_descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    0,
                    0,
                    frames_in_flight,
                    0,
                    0,
                    0,
                    0,
                    0,
                    None,
                ),
                frames_in_flight,
            ),
            Some("gi_lighting_descriptor_pool"),
        )?;

        let mut output_descriptor_sets = smallvec::SmallVec::<
            [Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        >::with_capacity(frames_in_flight as usize);
        for index in 0..(frames_in_flight as usize) {
            let descriptor_set = DescriptorSet::new(
                output_descriptor_pool.clone(),
                output_descriptor_set_layout.clone(),
            )?;

            descriptor_set.bind_resources(|binder| {
                // bind the output image(s)
                binder
                    .bind_storage_images_with_same_layout(
                        0,
                        ImageLayout::General,
                        [raytracing_gibuffer[index].clone()].as_slice(),
                    )
                    .unwrap();
            })?;

            output_descriptor_sets.push(descriptor_set);
        }

        // this is the descriptor set that is reserved for OTHER pipelines to use the data calculated in this one
        let gibuffer_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [BindingDescriptor::new(
                [ShaderStageAccessIn::Fragment].as_slice().into(),
                BindingType::Native(NativeBindingType::CombinedImageSampler),
                0,
                1,
            )]
            .as_slice(),
        )?;

        let gibuffer_descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    frames_in_flight,
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
            Some("global_illumination_buffer_descriptor_pool"),
        )?;

        let dlbuffer_sampler = Sampler::new(
            device.clone(),
            Filtering::Nearest,
            Filtering::Nearest,
            MipmapMode::ModeNearest,
            0.0,
        )?;

        let mut gibuffer_descriptor_sets =
            smallvec::SmallVec::<_>::with_capacity(frames_in_flight as usize);
        for index in 0_usize..(frames_in_flight as usize) {
            let descriptor_set = DescriptorSet::new(
                gibuffer_descriptor_pool.clone(),
                gibuffer_descriptor_set_layout.clone(),
            )?;

            descriptor_set.bind_resources(|binder| {
                binder
                    .bind_combined_images_samplers_with_same_layout_and_sampler(
                        0,
                        ImageLayout::ShaderReadOnlyOptimal,
                        dlbuffer_sampler.clone(),
                        [raytracing_gibuffer[index].clone()].as_slice(),
                    )
                    .unwrap();
            })?;

            gibuffer_descriptor_sets.push(descriptor_set);
        }

        let renderarea_width = render_area.width();
        let renderarea_height = render_area.height();

        Ok(Self {
            queue_family,

            raytracing_pipeline,

            raytracing_gibuffer,
            raytracing_sbts,

            output_descriptor_sets,

            gibuffer_descriptor_set_layout,
            gibuffer_descriptor_sets,

            renderarea_width,
            renderarea_height,
        })
    }

    pub fn record_rendering_commands(
        &self,
        raytracing_descriptor_set: Arc<DescriptorSet>,
        gbuffer_descriptor_set: Arc<DescriptorSet>,
        dlbuffer_descriptor_set: Arc<DescriptorSet>,
        current_frame: usize,
        gibuffer_stages: PipelineStages,
        gibuffer_access: MemoryAccess,
        recorder: &mut CommandBufferRecorder,
    ) -> Arc<DescriptorSet> {
        // TODO: of all word positions from GBUFFER, (order them, maybe) and for each directional light
        // use those to decide the portion that has to be rendered as depth in the shadow map

        // Here clear the image and transition its layout for using it in the raytracing pipeline
        {
            let image_srr: ImageSubresourceRange =
                self.raytracing_gibuffer[current_frame].image().into();

            recorder.image_barriers(
                [ImageMemoryBarrier::new(
                    [].as_slice().into(),
                    [].as_slice().into(),
                    [PipelineStage::RayTracingPipelineKHR(
                        PipelineStageRayTracingPipelineKHR::RayTracingShader,
                    )]
                    .as_slice()
                    .into(),
                    [MemoryAccessAs::ShaderRead, MemoryAccessAs::ShaderWrite]
                        .as_slice()
                        .into(),
                    image_srr.clone(),
                    ImageLayout::Undefined,
                    ImageLayout::TransferDstOptimal,
                    self.queue_family.clone(),
                    self.queue_family.clone(),
                )]
                .as_slice(),
            );

            recorder.clear_color_image(
                ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0),
                image_srr.clone(),
            );

            recorder.image_barriers(
                [ImageMemoryBarrier::new(
                    [PipelineStage::Transfer].as_slice().into(),
                    [MemoryAccessAs::TransferWrite].as_slice().into(),
                    [PipelineStage::RayTracingPipelineKHR(
                        PipelineStageRayTracingPipelineKHR::RayTracingShader,
                    )]
                    .as_slice()
                    .into(),
                    [MemoryAccessAs::ShaderRead, MemoryAccessAs::ShaderWrite]
                        .as_slice()
                        .into(),
                    image_srr,
                    ImageLayout::TransferDstOptimal,
                    ImageLayout::General,
                    self.queue_family.clone(),
                    self.queue_family.clone(),
                )]
                .as_slice(),
            );
        }

        recorder.bind_ray_tracing_pipeline(self.raytracing_pipeline.clone());
        recorder.bind_descriptor_sets_for_ray_tracing_pipeline(
            self.raytracing_pipeline.get_parent_pipeline_layout(),
            0,
            [
                raytracing_descriptor_set,
                gbuffer_descriptor_set,
                dlbuffer_descriptor_set,
                self.output_descriptor_sets[current_frame].clone(),
            ]
            .as_slice(),
        );

        recorder.trace_rays(
            self.raytracing_sbts[current_frame].clone(),
            Image3DDimensions::new(self.renderarea_width, self.renderarea_height, 1),
        );

        let image_srr: ImageSubresourceRange =
            self.raytracing_gibuffer[current_frame].image().into();

        recorder.image_barriers(
            [ImageMemoryBarrier::new(
                [PipelineStage::RayTracingPipelineKHR(
                    PipelineStageRayTracingPipelineKHR::RayTracingShader,
                )]
                .as_slice()
                .into(),
                [
                    MemoryAccessAs::MemoryWrite,
                    MemoryAccessAs::MemoryRead,
                    MemoryAccessAs::ShaderWrite,
                    MemoryAccessAs::ShaderRead,
                ]
                .as_slice()
                .into(),
                gibuffer_stages,
                gibuffer_access,
                image_srr,
                ImageLayout::General,
                ImageLayout::ShaderReadOnlyOptimal,
                self.queue_family.clone(),
                self.queue_family.clone(),
            )]
            .as_slice(),
        );

        self.gibuffer_descriptor_sets[current_frame].clone()
    }
}
