use std::sync::Arc;

use inline_spirv::inline_spirv;
use vulkan_framework::{
    acceleration_structure::top_level::TopLevelAccelerationStructure,
    command_buffer::CommandBufferRecorder,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor,
        DescriptorPoolSizesAcceletarionStructureKHR, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::Device,
    pipeline_layout::{PipelineLayout, PipelineLayoutDependant},
    raytracing_pipeline::RaytracingPipeline,
    shader_layout_binding::{AccelerationStructureBindingType, BindingDescriptor, BindingType},
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

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;

uniform layout(set = 0, binding = 1, rgba32f) writeonly image2D outputImage;

layout(location = 0) rayPayloadEXT vec3 hitValue;

void main() {
    const vec2 resolution = vec2(imageSize(outputImage));

    const ivec2 pixelCoords = ivec2(gl_LaunchIDEXT.xy);

    const vec2 position_xy = vec2((float(pixelCoords.x) + 0.5) / resolution.x, (float(pixelCoords.y) + 0.5) / resolution.y);
    const vec3 origin = vec3(position_xy, -0.5);
    const vec3 direction = vec3(0.0, 0.0, 1.0);

    vec4 output_color = vec4(1.0, 0.0, 0.0, 0.0);

    hitValue = vec3(0.0, 0.0, 0.1);

    traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, origin.xyz, 0.001, direction.xyz, 10.0, 0);
    //                      gl_RayFlagsNoneEXT

    // Store the output color to the image
    imageStore(outputImage, pixelCoords, vec4(hitValue.xyz, 1.0));
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

//uniform layout(binding=0, set = 0, rgba32f) writeonly image2D someImage;

//layout(binding = 0, set = 1) uniform accelerationStructureEXT topLevelAS;

layout(location = 0) rayPayloadInEXT vec3 hitValue;

void main() {
    hitValue = vec3(0.0, 0.0, 0.2);
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

layout(location = 0) rayPayloadInEXT vec3 hitValue;
hitAttributeEXT vec2 attribs;

//layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
//layout(binding = 1, set = 0) uniform image2D outputImage;

void main() {
    const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    hitValue = barycentricCoords;
}
"#,
    glsl,
    rchit,
    vulkan1_2,
    entry = "main"
);

pub struct DirectionalLighting {
    raytracing_pipeline: Arc<RaytracingPipeline>,

    raytracing_descriptor_sets:
        smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
}

impl DirectionalLighting {
    pub fn new(
        device: Arc<Device>,
        render_area: &RenderingDimensions,
        frames_in_flight: u32,
    ) -> RenderingResult<Self> {
        let rt_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [BindingDescriptor::new(
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
            )]
            .as_slice(),
        )?;

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            &[rt_descriptor_set_layout.clone()],
            &[],
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

        let raytracing_descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
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
        for _ in 0..frames_in_flight {
            raytracing_descriptor_sets.push(DescriptorSet::new(
                raytracing_descriptor_pool.clone(),
                rt_descriptor_set_layout.clone(),
            )?);
        }

        Ok(Self {
            raytracing_pipeline,

            raytracing_descriptor_sets,
        })
    }

    pub fn record_rendering_commands(
        &self,
        tlas: Arc<TopLevelAccelerationStructure>,
        current_frame: usize,
        recorder: &mut CommandBufferRecorder,
    ) {
        self.raytracing_descriptor_sets[current_frame]
            .bind_resources(|binder| {
                binder.bind_tlas(0, [tlas].as_slice()).unwrap();
            })
            .unwrap();

        recorder.bind_ray_tracing_pipeline(self.raytracing_pipeline.clone());
        recorder.bind_descriptor_sets_for_ray_tracing_pipeline(
            self.raytracing_pipeline.get_parent_pipeline_layout(),
            0,
            [self.raytracing_descriptor_sets[current_frame].clone()].as_slice(),
        );
    }
}
