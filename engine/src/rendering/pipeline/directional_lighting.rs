use std::{
    ops::DerefMut,
    sync::{Arc, Mutex},
};

use inline_spirv::inline_spirv;
use vulkan_framework::{
    acceleration_structure::top_level::TopLevelAccelerationStructure,
    binding_tables::RaytracingBindingTables,
    buffer::{
        AllocatedBuffer, Buffer, BufferSubresourceRange, BufferTrait, BufferUseAs,
        ConcreteBufferDescriptor,
    },
    command_buffer::CommandBufferRecorder,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor,
        DescriptorPoolSizesAcceletarionStructureKHR, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::DeviceOwned,
    image::{
        CommonImageFormat, ConcreteImageDescriptor, Image, Image1DTrait, Image2DTrait,
        Image3DDimensions, ImageFlags, ImageFormat, ImageLayout, ImageMultisampling,
        ImageSubresourceRange, ImageTiling, ImageUseAs,
    },
    image_view::ImageView,
    memory_barriers::{BufferMemoryBarrier, ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    memory_heap::{MemoryHostVisibility, MemoryType},
    memory_management::{MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait},
    memory_pool::MemoryPoolFeatures,
    pipeline_layout::{PipelineLayout, PipelineLayoutDependant},
    pipeline_stage::{PipelineStage, PipelineStageRayTracingPipelineKHR, PipelineStages},
    push_constant_range::PushConstanRange,
    queue_family::QueueFamily,
    raytracing_pipeline::RaytracingPipeline,
    sampler::{Filtering, MipmapMode, Sampler},
    shader_layout_binding::{
        AccelerationStructureBindingType, BindingDescriptor, BindingType, NativeBindingType,
    },
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

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;

uniform layout(set = 0, binding = 1, r32ui) uimage2D outputImage;

// gbuffer: 0 for position, 1 for normal, 2 for diffuse texture
layout(set = 1, binding = 0) uniform sampler2D gbuffer[3];

layout(std430, set = 0, binding = 2) readonly buffer directional_lights
{
    vec3 direction[];
};

layout(push_constant) uniform DirectionalLightingData {
    uint light_index;
} directional_lighting_data;

layout(location = 0) rayPayloadEXT bool hitValue;

void main() {
    const vec2 resolution = vec2(imageSize(outputImage));

    const ivec2 pixelCoords = ivec2(gl_LaunchIDEXT.xy);

    const vec2 position_xy = vec2(float(gl_LaunchIDEXT.x) / float(resolution.x), float(gl_LaunchIDEXT.y) / float(resolution.y));

    const vec3 origin = texture(gbuffer[0], position_xy).xyz;
    const vec3 direction = -1.0 * direction[directional_lighting_data.light_index];

    hitValue = true;

    traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, origin.xyz, 0.001, direction.xyz, 100000.0, 0);
    //                      gl_RayFlagsNoneEXT

    const uint light_hit_bool = (!hitValue ? 1 : 0) << (32 - directional_lighting_data.light_index);

    // Store the hit boolean to the image
    imageAtomicOr(outputImage, pixelCoords, light_hit_bool);
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

//uniform layout(binding=0, set = 0, r32) writeonly uimage2D someImage;

//layout(binding = 0, set = 1) uniform accelerationStructureEXT topLevelAS;

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
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInEXT bool hitValue;

void main() {
    hitValue = false;
}
"#,
    glsl,
    rchit,
    vulkan1_2,
    entry = "main"
);

type DirectionsBuffersType =
    smallvec::SmallVec<[Arc<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

type LDBuffersType = smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

type RaytracingSBTPerDLightType =
    smallvec::SmallVec<[Arc<RaytracingBindingTables>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

type RaytracingSBTType =
    smallvec::SmallVec<[RaytracingSBTPerDLightType; MAX_DIRECTIONAL_LIGHTS as usize]>;

pub struct DirectionalLighting {
    queue_family: Arc<QueueFamily>,

    raytracing_pipeline: Arc<RaytracingPipeline>,

    raytracing_directions: DirectionsBuffersType,
    raytracing_ldbuffer: LDBuffersType,

    raytracing_sbts: RaytracingSBTType,

    raytracing_descriptor_sets:
        smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    dlbuffer_descriptor_set_layout: Arc<DescriptorSetLayout>,
    dlbuffer_descriptor_sets:
        smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
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
        gbuffer_descriptor_set_layout: Arc<DescriptorSetLayout>,
        frames_in_flight: u32,
    ) -> RenderingResult<Self> {
        let device = queue_family.get_parent_device();

        let rt_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [
                BindingDescriptor::new(
                    [ShaderStageAccessIn::RayTracing(
                        ShaderStageAccessInRayTracingKHR::RayGen,
                    )]
                    .as_slice()
                    .into(),
                    BindingType::AccelerationStructure(
                        AccelerationStructureBindingType::AccelerationStructure,
                    ),
                    0,
                    1,
                ),
                BindingDescriptor::new(
                    [ShaderStageAccessIn::RayTracing(
                        ShaderStageAccessInRayTracingKHR::RayGen,
                    )]
                    .as_slice()
                    .into(),
                    BindingType::Native(NativeBindingType::StorageImage),
                    1,
                    1,
                ),
                BindingDescriptor::new(
                    [ShaderStageAccessIn::RayTracing(
                        ShaderStageAccessInRayTracingKHR::RayGen,
                    )]
                    .as_slice()
                    .into(),
                    BindingType::Native(NativeBindingType::StorageBuffer),
                    2,
                    1,
                ),
            ]
            .as_slice(),
        )?;

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            &[
                rt_descriptor_set_layout.clone(),
                gbuffer_descriptor_set_layout,
            ],
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
        for index in 0..frames_in_flight {
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
                        ImageFormat::from(CommonImageFormat::r32_uint),
                        ImageFlags::empty(),
                        ImageTiling::Optimal,
                    ),
                    None,
                    Some(format!("raytracing_ldimage[{index}]").as_str()),
                )?
                .into(),
            );
        }

        let raytracing_directions_unallocated = (0..frames_in_flight)
            .map(|index| {
                Buffer::new(
                    device.clone(),
                    ConcreteBufferDescriptor::new(
                        [BufferUseAs::TransferDst, BufferUseAs::StorageBuffer]
                            .as_slice()
                            .into(),
                        4u64 * 3u64 * (MAX_DIRECTIONAL_LIGHTS as u64),
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
                &MemoryType::DeviceLocal(Some(MemoryHostVisibility::hidden())),
                &MemoryPoolFeatures::new(false),
                raytracing_directions_unallocated,
                MemoryManagementTags::default()
                    .with_name("directional_lighting".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?;

            let raytracing_ldbuffer_allocated = mem_manager.allocate_resources(
                &MemoryType::DeviceLocal(Some(MemoryHostVisibility::hidden())),
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

            let raytracing_sbts = (0..MAX_DIRECTIONAL_LIGHTS)
                .map(|_sbt_index| {
                    (0..frames_in_flight)
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
                        .collect::<RaytracingSBTPerDLightType>()
                })
                .collect::<RaytracingSBTType>();

            let raytracing_ldbuffer = raytracing_ldbuffer_allocated
                .into_iter()
                .enumerate()
                .map(|(index, allocated)| {
                    ImageView::new(
                        allocated.image(),
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        Some(format!("raytracing_ldimage_imageview[{index}]").as_str()),
                    )
                    .unwrap()
                })
                .collect::<LDBuffersType>();

            (raytracing_directions, raytracing_ldbuffer, raytracing_sbts)
        };

        let raytracing_descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    0,
                    0,
                    frames_in_flight,
                    0,
                    0,
                    frames_in_flight,
                    0,
                    0,
                    Some(DescriptorPoolSizesAcceletarionStructureKHR::new(1)),
                ),
                frames_in_flight,
            ),
            Some("directional_lighting_descriptor_pool"),
        )?;

        let mut raytracing_descriptor_sets = smallvec::SmallVec::<
            [Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        >::with_capacity(frames_in_flight as usize);
        for index in 0..(frames_in_flight as usize) {
            let descriptor_set = DescriptorSet::new(
                raytracing_descriptor_pool.clone(),
                rt_descriptor_set_layout.clone(),
            )?;

            descriptor_set.bind_resources(|binder| {
                binder
                    .bind_storage_images(
                        1,
                        [(ImageLayout::General, raytracing_ldbuffer[index].clone())].as_slice(),
                    )
                    .unwrap();

                binder
                    .bind_storage_buffers(
                        2,
                        [(
                            raytracing_directions[index].clone() as Arc<dyn BufferTrait>,
                            None,
                            None,
                        )]
                        .as_slice(),
                    )
                    .unwrap();
            })?;

            raytracing_descriptor_sets.push(descriptor_set);
        }

        let dlbuffer_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [BindingDescriptor::new(
                [ShaderStageAccessIn::Fragment].as_slice().into(),
                BindingType::Native(NativeBindingType::CombinedImageSampler),
                0,
                1,
            )]
            .as_slice(),
        )?;

        let dlbuffer_descriptor_pool = DescriptorPool::new(
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
            Some(""),
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
                    .bind_combined_images_samplers(
                        0,
                        [(
                            ImageLayout::ShaderReadOnlyOptimal,
                            raytracing_ldbuffer[index].clone(),
                            dlbuffer_sampler.clone(),
                        )]
                        .as_slice(),
                    )
                    .unwrap()
            })?;

            dlbuffer_descriptor_sets.push(descriptor_set);
        }

        Ok(Self {
            queue_family,

            raytracing_pipeline,

            raytracing_directions,
            raytracing_ldbuffer,
            raytracing_sbts,

            raytracing_descriptor_sets,

            dlbuffer_descriptor_set_layout,
            dlbuffer_descriptor_sets,
        })
    }

    pub fn record_rendering_commands(
        &self,
        tlas: Arc<TopLevelAccelerationStructure>,
        gbuffer_descriptor_set: Arc<DescriptorSet>,
        directional_lights: &DirectionalLights,
        current_frame: usize,
        dlbuffer_stages: PipelineStages,
        dlbuffer_access: MemoryAccess,
        recorder: &mut CommandBufferRecorder,
    ) -> Arc<DescriptorSet> {
        let image_view = self.raytracing_ldbuffer[current_frame].clone();

        // TODO: of all word positions from GBUFFER, (order them, maybe) and for each directional light
        // use those to decide the portion that has to be rendered as depth in the shadow map

        // Here clear the image and transition its layout for using it in the raytracing pipeline
        {
            let image_srr: ImageSubresourceRange =
                self.raytracing_ldbuffer[current_frame].image().into();

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

            recorder.clear_color_image(image_srr.clone());

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

        // get the number of directional lights to compute and transfer into the buffer theirs directions
        let lights_count = directional_lights.count();
        let size_of_direction = 4u64 * 3u64;

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
                        (lights_count as u64) * size_of_direction,
                    ),
                    self.queue_family.clone(),
                    self.queue_family.clone(),
                )]
                .as_slice(),
            );

            directional_lights.foreach(|dir_light| {
                recorder.copy_buffer(
                    dir_light.clone(),
                    self.raytracing_directions[current_frame].clone(),
                    [(
                        0u64,
                        (lights_count as u64) * size_of_direction,
                        size_of_direction,
                    )]
                    .as_slice(),
                );
            });

            recorder.buffer_barriers(
                [BufferMemoryBarrier::new(
                    [PipelineStage::Transfer].as_slice().into(),
                    [MemoryAccessAs::TransferWrite].as_slice().into(),
                    [PipelineStage::RayTracingPipelineKHR(
                        PipelineStageRayTracingPipelineKHR::RayTracingShader,
                    )]
                    .as_slice()
                    .into(),
                    [MemoryAccessAs::ShaderRead].as_slice().into(),
                    BufferSubresourceRange::new(
                        self.raytracing_directions[current_frame].clone(),
                        0,
                        size_of_direction * (lights_count as u64),
                    ),
                    self.queue_family.clone(),
                    self.queue_family.clone(),
                )]
                .as_slice(),
            );
        };

        self.raytracing_descriptor_sets[current_frame]
            .bind_resources(|binder| {
                binder.bind_tlas(0, [tlas].as_slice()).unwrap();
            })
            .unwrap();

        recorder.bind_ray_tracing_pipeline(self.raytracing_pipeline.clone());
        recorder.bind_descriptor_sets_for_ray_tracing_pipeline(
            self.raytracing_pipeline.get_parent_pipeline_layout(),
            0,
            [
                self.raytracing_descriptor_sets[current_frame].clone(),
                gbuffer_descriptor_set,
            ]
            .as_slice(),
        );

        for light_index in 0..lights_count {
            recorder.push_constant(
                self.raytracing_pipeline.get_parent_pipeline_layout(),
                [ShaderStageAccessIn::RayTracing(
                    ShaderStageAccessInRayTracingKHR::RayGen,
                )]
                .as_slice()
                .into(),
                0,
                unsafe { std::mem::transmute::<&u32, &[u8; 4]>(&light_index) },
            );

            recorder.trace_rays(
                self.raytracing_sbts[light_index as usize][current_frame].clone(),
                match image_view.image().dimensions() {
                    vulkan_framework::image::ImageDimensions::Image1D { extent } => {
                        Image3DDimensions::new(extent.width(), 1, 1)
                    }
                    vulkan_framework::image::ImageDimensions::Image2D { extent } => {
                        Image3DDimensions::new(extent.width(), extent.height(), 1)
                    }
                    vulkan_framework::image::ImageDimensions::Image3D { extent } => extent,
                },
            );
        }

        recorder.image_barriers(
            [ImageMemoryBarrier::new(
                [PipelineStage::RayTracingPipelineKHR(
                    PipelineStageRayTracingPipelineKHR::RayTracingShader,
                )]
                .as_slice()
                .into(),
                [].as_slice().into(),
                dlbuffer_stages,
                dlbuffer_access,
                self.raytracing_ldbuffer[current_frame].image().into(),
                ImageLayout::General,
                ImageLayout::ShaderReadOnlyOptimal,
                self.queue_family.clone(),
                self.queue_family.clone(),
            )]
            .as_slice(),
        );

        self.dlbuffer_descriptor_sets[current_frame].clone()
    }
}
