use std::{
    ops::DerefMut,
    sync::{Arc, Mutex},
};

use inline_spirv::inline_spirv;
use vulkan_framework::{
    binding_tables::RaytracingBindingTables,
    buffer::{
        AllocatedBuffer, Buffer, BufferSubresourceRange, BufferTrait, BufferUseAs,
        ConcreteBufferDescriptor,
    },
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
    memory_barriers::{BufferMemoryBarrier, ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    memory_heap::MemoryType,
    memory_management::{MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait},
    memory_pool::MemoryPoolFeatures,
    pipeline_layout::{PipelineLayout, PipelineLayoutDependant},
    pipeline_stage::{PipelineStage, PipelineStageRayTracingPipelineKHR, PipelineStages},
    push_constant_range::PushConstanRange,
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
    MAX_DIRECTIONAL_LIGHTS, MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingResult,
    rendering_dimensions::RenderingDimensions, resources::directional_lights::DirectionalLights,
};

const RAYGEN_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_flags_primitive_culling : require
#extension GL_EXT_nonuniform_qualifier : enable

#define MAX_DIRECTIONAL_LIGHTS 8

layout (set = 0, binding = 0, std430) readonly buffer tlas_instances
{
    uint data[];
};

layout (set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

// gbuffer: 0 for position, 1 for normal, 2 for diffuse texture
layout (set = 1, binding = 0) uniform sampler2D gbuffer[3];

struct light_t {
    float direction_x;
    float direction_y;
    float direction_z;

    float intensity_x;
    float intensity_y;
    float intensity_z;
};

layout (set = 2, binding = 0, std430) readonly buffer directional_lights
{
    light_t light[];
};

uniform layout (set = 2, binding = 1, rg32f) image2D outputImage[MAX_DIRECTIONAL_LIGHTS];

layout(push_constant) uniform DirectionalLightingData {
    uint lights_count;
} directional_lighting_data;

layout(location = 0) rayPayloadEXT bool hitValue;

void main() {
    const ivec2 resolution = imageSize(outputImage[0]);

    const vec2 position_xy = vec2(float(gl_LaunchIDEXT.x) / float(resolution.x), float(gl_LaunchIDEXT.y) / float(resolution.y));

    const vec3 origin = texture(gbuffer[0], position_xy).xyz;
    const vec3 normal = texture(gbuffer[1], position_xy).xyz;

    for (uint light_index = 0; light_index < directional_lighting_data.lights_count; light_index++) {
        const vec3 light_dir = vec3(light[light_index].direction_x, light[light_index].direction_y, light[light_index].direction_z);
        const vec3 ray_dir = -1.0 * light_dir;

        hitValue = true;

        if (!(origin.x == 0 && origin.y == 0 && origin.z == 0)) {
            // other flags: gl_RayFlagsCullNoOpaqueEXT gl_RayFlagsNoneEXT
            traceRayEXT(topLevelAS, gl_RayFlagsSkipAABBEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xff, 0, 0, 0, origin.xyz, 0.1, ray_dir.xyz, 10000.0, 0);
        }

        float diffuse_contribution = 0.0;
        if (!hitValue) {
            diffuse_contribution = max(dot(normal, ray_dir), 0.0);
        }

        imageStore(outputImage[light_index], ivec2(gl_LaunchIDEXT.xy), vec4(diffuse_contribution, 0.0, 0.0, 0.0));
    }
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

layout(location = 0) rayPayloadInEXT bool hitValue;

void main() {
    hitValue = false;
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
#extension GL_EXT_buffer_reference : require

#define Buffer(Alignment) \
  layout(buffer_reference, std430, buffer_reference_align = Alignment) buffer

struct vertex_buffer_element_t {
    vec3 position;
    vec3 normal;
    vec2 texture_uv;
};

Buffer(64) VertexBuffer {
  vertex_buffer_element_t vertex_data[];
};

Buffer(64) IndexBuffer {
  uint vertex_index[];
};

Buffer(64) TransformBuffer {
  mat3x4 transform[];
};

struct instance_buffer_t {
    mat3x4 model_matrix;
};

Buffer(64) InstanceBuffer {
  instance_buffer_t transform[];
};

struct tlas_instance_data_t {
    IndexBuffer ib;
    VertexBuffer vb;
    TransformBuffer tb;
    InstanceBuffer instance;
    uint instance_num;
    uint padding;
};

layout(std430, set = 0, binding = 0) readonly buffer tlas_instances
{
    tlas_instance_data_t data[];
};

layout(location = 0) rayPayloadInEXT bool hitValue;

//hitAttributeEXT vec2 attribs;

void main() {
    //const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    //hitValue = barycentricCoords;
}
"#,
    glsl,
    rchit,
    vulkan1_2,
    entry = "main"
);

type DirectionsBuffersType =
    smallvec::SmallVec<[Arc<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

type LDBuffersTypePerDLightsGroup =
    smallvec::SmallVec<[Arc<ImageView>; MAX_DIRECTIONAL_LIGHTS as usize]>;

type LDBuffersType =
    smallvec::SmallVec<[LDBuffersTypePerDLightsGroup; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

type RaytracingSBTPerDLightType =
    smallvec::SmallVec<[Arc<RaytracingBindingTables>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

pub struct DirectionalLighting {
    queue_family: Arc<QueueFamily>,

    raytracing_pipeline: Arc<RaytracingPipeline>,

    raytracing_directions: DirectionsBuffersType,
    raytracing_ldbuffer: LDBuffersType,

    raytracing_sbts: RaytracingSBTPerDLightType,

    output_descriptor_sets:
        smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    dlbuffer_descriptor_set_layout: Arc<DescriptorSetLayout>,
    dlbuffer_descriptor_sets:
        smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    renderarea_width: u32,
    renderarea_height: u32,
}

impl DirectionalLighting {
    /// Returns the descriptor set layout for the gbuffer
    #[inline(always)]
    pub fn descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.dlbuffer_descriptor_set_layout.clone()
    }

    pub fn new(
        queue_family: Arc<QueueFamily>,
        render_area: &RenderingDimensions,
        memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,
        rt_descriptor_set_layout: Arc<DescriptorSetLayout>,
        gbuffer_descriptor_set_layout: Arc<DescriptorSetLayout>,
        frames_in_flight: u32,
    ) -> RenderingResult<Self> {
        let device = queue_family.get_parent_device();

        let output_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [
                // Directional lights buffer
                BindingDescriptor::new(
                    [ShaderStageAccessIn::RayTracing(
                        ShaderStageAccessInRayTracingKHR::RayGen,
                    )]
                    .as_slice()
                    .into(),
                    BindingType::Native(NativeBindingType::StorageBuffer),
                    0,
                    1,
                ),
                // outputImage(s)
                BindingDescriptor::new(
                    [ShaderStageAccessIn::RayTracing(
                        ShaderStageAccessInRayTracingKHR::RayGen,
                    )]
                    .as_slice()
                    .into(),
                    BindingType::Native(NativeBindingType::StorageImage),
                    1,
                    MAX_DIRECTIONAL_LIGHTS,
                ),
            ]
            .as_slice(),
        )?;

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            [
                rt_descriptor_set_layout.clone(),
                gbuffer_descriptor_set_layout,
                output_descriptor_set_layout.clone(),
            ]
            .as_slice(),
            &[PushConstanRange::new(
                0,
                std::mem::size_of::<u32>() as u32,
                [ShaderStageAccessIn::RayTracing(
                    ShaderStageAccessInRayTracingKHR::RayGen,
                )]
                .as_slice()
                .into(),
            )],
            Some("directional_lighting_pipeline_layout"),
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
            Some("directional_lighting_raytracing_pipeline!"),
        )?;

        let mut raytracing_ldbuffer_unallocated = vec![];
        for frame_index in 0..frames_in_flight {
            for light_index in 0..MAX_DIRECTIONAL_LIGHTS {
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
                            ImageFormat::from(CommonImageFormat::r32g32_sfloat),
                            ImageFlags::empty(),
                            ImageTiling::Optimal,
                        ),
                        None,
                        Some(format!("raytracing_ldimage[{frame_index}][{light_index}]").as_str()),
                    )?
                    .into(),
                );
            }
        }

        let raytracing_directions_unallocated = (0..frames_in_flight)
            .map(|index| {
                Buffer::new(
                    device.clone(),
                    ConcreteBufferDescriptor::new(
                        [BufferUseAs::TransferDst, BufferUseAs::StorageBuffer]
                            .as_slice()
                            .into(),
                        4u64 * 6u64 * (MAX_DIRECTIONAL_LIGHTS as u64),
                    ),
                    None,
                    Some(format!("raytracing_directions[{index}]").as_str()),
                )
                .unwrap()
                .into()
            })
            .collect::<Vec<_>>();

        let (raytracing_directions, raytracing_ldbuffer, raytracing_sbts) = {
            let mut mem_manager = memory_manager.lock().unwrap();

            let raytracing_directions_allocated = mem_manager.allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::new(false),
                raytracing_directions_unallocated,
                MemoryManagementTags::default()
                    .with_name("directional_lighting".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?;

            let raytracing_ldbuffer_allocated = mem_manager.allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::new(false),
                raytracing_ldbuffer_unallocated,
                MemoryManagementTags::default()
                    .with_name("directional_lighting".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?;

            let raytracing_directions = raytracing_directions_allocated
                .into_iter()
                .map(|allocated| allocated.buffer())
                .collect::<DirectionsBuffersType>();

            let raytracing_sbts = (0..frames_in_flight)
                .map(|_frame_index| {
                    RaytracingBindingTables::new(
                        raytracing_pipeline.clone(),
                        mem_manager.deref_mut(),
                        MemoryManagementTags::default()
                            .with_name("directional_lighting".to_string())
                            .with_size(MemoryManagementTagSize::MediumSmall),
                    )
                    .unwrap()
                })
                .collect::<RaytracingSBTPerDLightType>();

            let mut raytracing_ldbuffer = LDBuffersType::with_capacity(frames_in_flight as usize);
            for frame_index in 0..frames_in_flight {
                let mut raytracing_light_ldbuffer =
                    LDBuffersTypePerDLightsGroup::with_capacity(MAX_DIRECTIONAL_LIGHTS as usize);
                for light_index in 0..MAX_DIRECTIONAL_LIGHTS {
                    raytracing_light_ldbuffer.push(ImageView::new(
                        raytracing_ldbuffer_allocated
                            [((frame_index * MAX_DIRECTIONAL_LIGHTS) + light_index) as usize]
                            .image(),
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        Some(
                            format!("raytracing_ldimage_imageview[{frame_index}][{light_index}]")
                                .as_str(),
                        ),
                    )?);
                }

                raytracing_ldbuffer.push(raytracing_light_ldbuffer);
            }

            (raytracing_directions, raytracing_ldbuffer, raytracing_sbts)
        };

        let output_descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    0,
                    0,
                    MAX_DIRECTIONAL_LIGHTS * frames_in_flight,
                    0,
                    0,
                    frames_in_flight,
                    0,
                    0,
                    None,
                ),
                frames_in_flight,
            ),
            Some("directional_lighting_descriptor_pool"),
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
                // bind the buffer with lights definitions
                binder
                    .bind_storage_buffers(
                        0,
                        [(
                            raytracing_directions[index].clone() as Arc<dyn BufferTrait>,
                            None,
                            None,
                        )]
                        .as_slice(),
                    )
                    .unwrap();

                // bind the output image(s)
                binder
                    .bind_storage_images_with_same_layout(
                        1,
                        ImageLayout::General,
                        raytracing_ldbuffer[index].as_slice(),
                    )
                    .unwrap();
            })?;

            output_descriptor_sets.push(descriptor_set);
        }

        // this is the descriptor set that is reserved for OTHER pipelines to use the data calculated in this one
        let dlbuffer_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [
                BindingDescriptor::new(
                    [ShaderStageAccessIn::Fragment].as_slice().into(),
                    BindingType::Native(NativeBindingType::StorageBuffer),
                    0,
                    1,
                ),
                BindingDescriptor::new(
                    [ShaderStageAccessIn::Fragment].as_slice().into(),
                    BindingType::Native(NativeBindingType::CombinedImageSampler),
                    1,
                    MAX_DIRECTIONAL_LIGHTS,
                ),
            ]
            .as_slice(),
        )?;

        let dlbuffer_descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    frames_in_flight * MAX_DIRECTIONAL_LIGHTS,
                    0,
                    0,
                    0,
                    0,
                    frames_in_flight,
                    0,
                    0,
                    None,
                ),
                frames_in_flight,
            ),
            Some("dlbuffer_descriptor_pool"),
        )?;

        let dlbuffer_sampler = Sampler::new(
            device.clone(),
            Filtering::Nearest,
            Filtering::Nearest,
            MipmapMode::ModeNearest,
            0.0,
        )?;

        let mut dlbuffer_descriptor_sets =
            smallvec::SmallVec::<_>::with_capacity(frames_in_flight as usize);
        for index in 0_usize..(frames_in_flight as usize) {
            let descriptor_set = DescriptorSet::new(
                dlbuffer_descriptor_pool.clone(),
                dlbuffer_descriptor_set_layout.clone(),
            )?;

            descriptor_set.bind_resources(|binder| {
                binder
                    .bind_storage_buffers(
                        0,
                        [(
                            raytracing_directions[index].clone() as Arc<dyn BufferTrait>,
                            None,
                            None,
                        )]
                        .as_slice(),
                    )
                    .unwrap();
                binder
                    .bind_combined_images_samplers_with_same_layout_and_sampler(
                        1,
                        ImageLayout::ShaderReadOnlyOptimal,
                        dlbuffer_sampler.clone(),
                        raytracing_ldbuffer[index].as_slice(),
                    )
                    .unwrap();
            })?;

            dlbuffer_descriptor_sets.push(descriptor_set);
        }

        let renderarea_width = render_area.width();
        let renderarea_height = render_area.height();

        Ok(Self {
            queue_family,

            raytracing_pipeline,

            raytracing_directions,
            raytracing_ldbuffer,
            raytracing_sbts,

            output_descriptor_sets,

            dlbuffer_descriptor_set_layout,
            dlbuffer_descriptor_sets,

            renderarea_width,
            renderarea_height,
        })
    }

    pub fn record_rendering_commands(
        &self,
        raytracing_descriptor_set: Arc<DescriptorSet>,
        gbuffer_descriptor_set: Arc<DescriptorSet>,
        directional_lights: &DirectionalLights,
        current_frame: usize,
        dlbuffer_stages: PipelineStages,
        dlbuffer_access: MemoryAccess,
        recorder: &mut CommandBufferRecorder,
    ) -> Arc<DescriptorSet> {
        // TODO: of all word positions from GBUFFER, (order them, maybe) and for each directional light
        // use those to decide the portion that has to be rendered as depth in the shadow map

        // Here clear the image and transition its layout for using it in the raytracing pipeline
        {
            for light_index in 0..MAX_DIRECTIONAL_LIGHTS {
                let image_srr: ImageSubresourceRange = self.raytracing_ldbuffer[current_frame]
                    [light_index as usize]
                    .image()
                    .into();

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
        }

        // get the number of directional lights to compute and transfer into the buffer theirs directions
        let lights_count = directional_lights.count();
        let size_of_light = 4u64 * 6u64;

        if lights_count > 0 {
            recorder.buffer_barriers(
                [BufferMemoryBarrier::new(
                    [].as_slice().into(),
                    [].as_slice().into(),
                    [PipelineStage::Transfer].as_slice().into(),
                    [MemoryAccessAs::TransferWrite].as_slice().into(),
                    BufferSubresourceRange::new(
                        self.raytracing_directions[current_frame].clone(),
                        0,
                        (lights_count as u64) * size_of_light,
                    ),
                    self.queue_family.clone(),
                    self.queue_family.clone(),
                )]
                .as_slice(),
            );

            let mut light_index = 0u64;
            directional_lights.foreach(|dir_light| {
                recorder.copy_buffer(
                    dir_light.clone(),
                    self.raytracing_directions[current_frame].clone(),
                    [(0u64, (light_index as u64) * size_of_light, size_of_light)].as_slice(),
                );

                light_index += 1u64;
            });

            recorder.buffer_barriers(
                [BufferMemoryBarrier::new(
                    [PipelineStage::Transfer].as_slice().into(),
                    [MemoryAccessAs::TransferWrite].as_slice().into(),
                    [
                        PipelineStage::RayTracingPipelineKHR(
                            PipelineStageRayTracingPipelineKHR::RayTracingShader,
                        ),
                        PipelineStage::AllGraphics,
                    ]
                    .as_slice()
                    .into(),
                    [MemoryAccessAs::MemoryRead, MemoryAccessAs::ShaderRead]
                        .as_slice()
                        .into(),
                    BufferSubresourceRange::new(
                        self.raytracing_directions[current_frame].clone(),
                        0,
                        size_of_light * (lights_count as u64),
                    ),
                    self.queue_family.clone(),
                    self.queue_family.clone(),
                )]
                .as_slice(),
            );
        };

        recorder.bind_ray_tracing_pipeline(self.raytracing_pipeline.clone());
        recorder.bind_descriptor_sets_for_ray_tracing_pipeline(
            self.raytracing_pipeline.get_parent_pipeline_layout(),
            0,
            [
                raytracing_descriptor_set,
                gbuffer_descriptor_set,
                self.output_descriptor_sets[current_frame].clone(),
            ]
            .as_slice(),
        );

        recorder.push_constant(
            self.raytracing_pipeline.get_parent_pipeline_layout(),
            [ShaderStageAccessIn::RayTracing(
                ShaderStageAccessInRayTracingKHR::RayGen,
            )]
            .as_slice()
            .into(),
            0,
            unsafe { std::mem::transmute::<&u32, &[u8; 4]>(&lights_count) },
        );

        recorder.trace_rays(
            self.raytracing_sbts[current_frame].clone(),
            Image3DDimensions::new(self.renderarea_width, self.renderarea_height, 1),
        );

        let mut image_barriers = vec![];
        for light_index in 0..MAX_DIRECTIONAL_LIGHTS {
            let image_srr: ImageSubresourceRange = self.raytracing_ldbuffer[current_frame]
                [light_index as usize]
                .image()
                .into();

            image_barriers.push(ImageMemoryBarrier::new(
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
                dlbuffer_stages,
                dlbuffer_access,
                image_srr,
                ImageLayout::General,
                ImageLayout::ShaderReadOnlyOptimal,
                self.queue_family.clone(),
                self.queue_family.clone(),
            ));
        }

        recorder.image_barriers(&image_barriers.as_slice());

        self.dlbuffer_descriptor_sets[current_frame].clone()
    }
}
