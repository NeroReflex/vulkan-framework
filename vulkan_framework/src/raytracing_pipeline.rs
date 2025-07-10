use std::ffi::CStr;
use std::sync::Arc;

use crate::shaders::{
    any_hit_shader::AnyHitShader, callable_shader::CallableShader,
    closest_hit_shader::ClosestHitShader, intersection_shader::IntersectionShader,
    miss_shader::MissShader, raygen_shader::RaygenShader,
};

use crate::device::{Device, DeviceOwned};
use crate::instance::InstanceOwned;

use crate::pipeline_layout::{PipelineLayout, PipelineLayoutDependant};
use crate::prelude::{VulkanError, VulkanResult};

use crate::shader_trait::PrivateShaderTrait;

pub struct RaytracingPipeline {
    device: Arc<Device>,
    pipeline_layout: Arc<PipelineLayout>,
    pipeline: ash::vk::Pipeline,
    max_pipeline_ray_recursion_depth: u32,
    shader_group_size: u32,
    callable_shader_present: bool,
}

impl PipelineLayoutDependant for RaytracingPipeline {
    fn get_parent_pipeline_layout(&self) -> Arc<PipelineLayout> {
        self.pipeline_layout.clone()
    }
}

impl DeviceOwned for RaytracingPipeline {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Drop for RaytracingPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_pipeline(
                self.pipeline,
                self.device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

impl RaytracingPipeline {
    pub fn callable_shader_present(&self) -> bool {
        self.callable_shader_present
    }

    pub fn shader_group_size(&self) -> u32 {
        self.shader_group_size
    }

    #[inline]
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.pipeline)
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::Pipeline {
        self.pipeline
    }

    pub fn max_pipeline_ray_recursion_depth(&self) -> u32 {
        self.max_pipeline_ray_recursion_depth
    }

    pub fn new(
        pipeline_layout: Arc<PipelineLayout>,
        max_pipeline_ray_recursion_depth: u32,
        raygen_shader: Arc<RaygenShader>,
        maybe_intersection_shader: Option<Arc<IntersectionShader>>,
        miss_shader: Arc<MissShader>,
        maybe_anyhit_shader: Option<Arc<AnyHitShader>>,
        closesthit_shader: Arc<ClosestHitShader>,
        maybe_callable_shader: Option<Arc<CallableShader>>,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = pipeline_layout.get_parent_device();

        let main_name =
            unsafe { CStr::from_bytes_with_nul_unchecked(&[109u8, 97u8, 105u8, 110u8, 0u8]) };

        match device.ray_tracing_info() {
            Some(info) => {
                assert!(max_pipeline_ray_recursion_depth <= info.max_ray_recursion_depth())
            }
            None => return Err(VulkanError::Framework(crate::prelude::FrameworkError::Unknown(Some("Raytracing supported features are not available, probably due to a framework bug".to_string())))),
        }

        match device.ash_ext_raytracing_pipeline_khr() {
            Some(raytracing_ext) => {
                //ash::vk::PipelineShaderStageCreateInfo::builder()

                let mut stages_create_info: smallvec::SmallVec<
                    [ash::vk::PipelineShaderStageCreateInfo; 8],
                > = smallvec::smallvec![];
                let mut shader_group_create_info: smallvec::SmallVec<
                    [ash::vk::RayTracingShaderGroupCreateInfoKHR; 8],
                > = smallvec::smallvec![];

                // VK_SHADER_UNUSED_KHR
                let shader_unused_khr = !0u32;

                // RayGen
                {
                    stages_create_info.push(
                        ash::vk::PipelineShaderStageCreateInfo::builder()
                            .stage(ash::vk::ShaderStageFlags::RAYGEN_KHR)
                            .module(raygen_shader.ash_handle())
                            .name(main_name)
                            .build(),
                    );

                    shader_group_create_info.push(
                        ash::vk::RayTracingShaderGroupCreateInfoKHR::builder()
                            .ty(ash::vk::RayTracingShaderGroupTypeKHR::GENERAL)
                            .general_shader((stages_create_info.len() - 1) as u32)
                            .closest_hit_shader(shader_unused_khr)
                            .closest_hit_shader(shader_unused_khr)
                            .any_hit_shader(shader_unused_khr)
                            .intersection_shader(shader_unused_khr)
                            .build(),
                    );
                }

                // Miss
                {
                    stages_create_info.push(
                        ash::vk::PipelineShaderStageCreateInfo::builder()
                            .stage(ash::vk::ShaderStageFlags::MISS_KHR)
                            .module(miss_shader.ash_handle())
                            .name(main_name)
                            .build(),
                    );

                    shader_group_create_info.push(
                        ash::vk::RayTracingShaderGroupCreateInfoKHR::builder()
                            .ty(ash::vk::RayTracingShaderGroupTypeKHR::GENERAL)
                            .general_shader((stages_create_info.len() - 1) as u32)
                            .closest_hit_shader(shader_unused_khr)
                            .closest_hit_shader(shader_unused_khr)
                            .any_hit_shader(shader_unused_khr)
                            .intersection_shader(shader_unused_khr)
                            .build(),
                    );
                }

                // ClosestHit
                {
                    stages_create_info.push(
                        ash::vk::PipelineShaderStageCreateInfo::builder()
                            .stage(ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                            .module(closesthit_shader.ash_handle())
                            .name(main_name)
                            .build(),
                    );

                    shader_group_create_info.push(
                        ash::vk::RayTracingShaderGroupCreateInfoKHR::builder()
                            .ty(ash::vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                            .general_shader(shader_unused_khr)
                            .closest_hit_shader((stages_create_info.len() - 1) as u32)
                            .any_hit_shader(shader_unused_khr)
                            .intersection_shader(shader_unused_khr)
                            .build(),
                    );
                }

                // Intersection
                if let Option::Some(intersection_shader) = &maybe_intersection_shader {
                    stages_create_info.push(
                        ash::vk::PipelineShaderStageCreateInfo::builder()
                            .stage(ash::vk::ShaderStageFlags::INTERSECTION_KHR)
                            .module(intersection_shader.ash_handle())
                            .name(main_name)
                            .build(),
                    );

                    shader_group_create_info.push(
                        ash::vk::RayTracingShaderGroupCreateInfoKHR::builder()
                            .ty(ash::vk::RayTracingShaderGroupTypeKHR::GENERAL)
                            .general_shader(shader_unused_khr)
                            .closest_hit_shader(shader_unused_khr)
                            .closest_hit_shader(shader_unused_khr)
                            .any_hit_shader(shader_unused_khr)
                            .intersection_shader((stages_create_info.len() - 1) as u32)
                            .build(),
                    );
                }

                // AnyHit
                if let Option::Some(anyhit_shader) = &maybe_anyhit_shader {
                    stages_create_info.push(
                        ash::vk::PipelineShaderStageCreateInfo::builder()
                            .stage(ash::vk::ShaderStageFlags::INTERSECTION_KHR)
                            .module(anyhit_shader.ash_handle())
                            .name(main_name)
                            .build(),
                    );

                    shader_group_create_info.push(
                        ash::vk::RayTracingShaderGroupCreateInfoKHR::builder()
                            .ty(ash::vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                            .general_shader(shader_unused_khr)
                            .closest_hit_shader(shader_unused_khr)
                            .closest_hit_shader(shader_unused_khr)
                            .any_hit_shader((stages_create_info.len() - 1) as u32)
                            .intersection_shader(shader_unused_khr)
                            .build(),
                    );
                }

                // Callable
                let mut callable_shader_present = false;
                if let Option::Some(callable_shader) = &maybe_callable_shader {
                    callable_shader_present = true;

                    stages_create_info.push(
                        ash::vk::PipelineShaderStageCreateInfo::builder()
                            .stage(ash::vk::ShaderStageFlags::CALLABLE_KHR)
                            .module(callable_shader.ash_handle())
                            .name(main_name)
                            .build(),
                    );
                }

                let create_info = ash::vk::RayTracingPipelineCreateInfoKHR::builder()
                    .layout(pipeline_layout.ash_handle())
                    .stages(stages_create_info.as_slice())
                    .max_pipeline_ray_recursion_depth(max_pipeline_ray_recursion_depth)
                    .groups(shader_group_create_info.as_slice())
                    .build();

                match unsafe {
                    raytracing_ext.create_ray_tracing_pipelines(
                        ash::vk::DeferredOperationKHR::null(),
                        ash::vk::PipelineCache::null(),
                        &[create_info],
                        device.get_parent_instance().get_alloc_callbacks(),
                    )
                } {
                    Ok(pipelines) => {
                        assert_eq!(pipelines.len(), 1);

                        let pipeline = pipelines[0];

                        let mut obj_name_bytes = vec![];
                        if let Some(ext) = device.get_parent_instance().get_debug_ext_extension() {
                            if let Some(name) = debug_name {
                                for name_ch in name.as_bytes().iter() {
                                    obj_name_bytes.push(*name_ch);
                                }
                                obj_name_bytes.push(0x00);

                                unsafe {
                                    let object_name = std::ffi::CStr::from_bytes_with_nul_unchecked(
                                        obj_name_bytes.as_slice(),
                                    );
                                    // set device name for debugging
                                    let dbg_info = ash::vk::DebugUtilsObjectNameInfoEXT::builder()
                                        .object_type(ash::vk::ObjectType::PIPELINE)
                                        .object_handle(ash::vk::Handle::as_raw(pipeline))
                                        .object_name(object_name)
                                        .build();

                                    if let Err(err) = ext.set_debug_utils_object_name(
                                        device.ash_handle().handle(),
                                        &dbg_info,
                                    ) {
                                        #[cfg(debug_assertions)]
                                        {
                                            println!("Error setting the Debug name for the newly created Pipeline, will use handle. Error: {}", err)
                                        }
                                    }
                                }
                            }
                        }

                        Ok(Arc::new(Self {
                            device,
                            pipeline_layout,
                            pipeline,
                            max_pipeline_ray_recursion_depth,
                            shader_group_size: shader_group_create_info.len() as u32,
                            callable_shader_present,
                        }))
                    }
                    Err(err) => Err(VulkanError::Vulkan(
                        err.as_raw(),
                        Some(format!("Error creating the raytracing pipeline: {}", err)),
                    )),
                }
            }
            None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_ray_tracing_pipeline",
            ))),
        }
    }
}
