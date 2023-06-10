use std::ffi::CStr;
use std::sync::Arc;

use crate::any_hit_shader::AnyHitShader;
use crate::callable_shader::CallableShader;
use crate::closest_hit_shader::ClosestHitShader;
use crate::compute_shader::ComputeShader;
use crate::device::{Device, DeviceOwned};
use crate::instance::InstanceOwned;

use crate::intersection_shader::IntersectionShader;
use crate::miss_shader::MissShader;
use crate::pipeline_layout::{PipelineLayout, PipelineLayoutDependant};
use crate::prelude::{VulkanError, VulkanResult};
use crate::raygen_shader::RaygenShader;
use crate::shader_trait::PrivateShaderTrait;

pub struct RaytracingPipeline {
    device: Arc<Device>,
    pipeline_layout: Arc<PipelineLayout>,
    pipeline: ash::vk::Pipeline,
    max_pipeline_ray_recursion_depth: u32,
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
        raygen: (Arc<RaygenShader>, Option<String>),
        intersection: (Arc<IntersectionShader>, Option<String>),
        miss: (Arc<MissShader>, Option<String>),
        anyhit: (Arc<AnyHitShader>, Option<String>),
        closesthit: (Arc<ClosestHitShader>, Option<String>),
        callable: (Arc<CallableShader>, Option<String>),
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = pipeline_layout.get_parent_device();

        match device.ray_tracing_info() {
            Some(info) => {
                assert!(info.max_ray_recursion_depth() <= max_pipeline_ray_recursion_depth)
            },
            None => {
                return Err(VulkanError::Unspecified)
            }
        }

        match device.ash_ext_raytracing_pipeline_khr() {
            Some(raytracing_ext) => {
                let (raygen_shader, raygen_name) = raygen;
                let raygen_name: &CStr = match raygen_name {
                    Option::Some(_n) => {
                        todo!()
                    }
                    Option::None => {
                        unsafe { CStr::from_bytes_with_nul_unchecked(&[109u8, 97u8, 105u8, 110u8, 0u8]) }
                        // main
                    }
                };

                let (miss_shader, miss_name) = miss;
                let miss_name: &CStr = match miss_name {
                    Option::Some(_n) => {
                        todo!()
                    }
                    Option::None => {
                        unsafe { CStr::from_bytes_with_nul_unchecked(&[109u8, 97u8, 105u8, 110u8, 0u8]) }
                        // main
                    }
                };

                let (intersection_shader, intersection_name) = intersection;
                let intersection_name: &CStr = match intersection_name {
                    Option::Some(_n) => {
                        todo!()
                    }
                    Option::None => {
                        unsafe { CStr::from_bytes_with_nul_unchecked(&[109u8, 97u8, 105u8, 110u8, 0u8]) }
                        // main
                    }
                };

                let (anyhit_shader, anyhit_name) = anyhit;
                let anyhit_name: &CStr = match anyhit_name {
                    Option::Some(_n) => {
                        todo!()
                    }
                    Option::None => {
                        unsafe { CStr::from_bytes_with_nul_unchecked(&[109u8, 97u8, 105u8, 110u8, 0u8]) }
                        // main
                    }
                };

                let (closesthit_shader, closesthit_name) = closesthit;
                let closesthit_name: &CStr = match closesthit_name {
                    Option::Some(_n) => {
                        todo!()
                    }
                    Option::None => {
                        unsafe { CStr::from_bytes_with_nul_unchecked(&[109u8, 97u8, 105u8, 110u8, 0u8]) }
                        // main
                    }
                };

                let (callable_shader, callable_name) = callable;
                let callable_name: &CStr = match callable_name {
                    Option::Some(_n) => {
                        todo!()
                    }
                    Option::None => {
                        unsafe { CStr::from_bytes_with_nul_unchecked(&[109u8, 97u8, 105u8, 110u8, 0u8]) }
                        // main
                    }
                };

                //ash::vk::PipelineShaderStageCreateInfo::builder()
                
                let stages_create_info: smallvec::SmallVec<[ash::vk::PipelineShaderStageCreateInfo; 8]> = smallvec::smallvec![
                    ash::vk::PipelineShaderStageCreateInfo::builder()
                        .stage(ash::vk::ShaderStageFlags::RAYGEN_KHR)
                        .module(raygen_shader.ash_handle())
                        .name(raygen_name)
                        .build(),
                    ash::vk::PipelineShaderStageCreateInfo::builder()
                        .stage(ash::vk::ShaderStageFlags::INTERSECTION_KHR)
                        .module(intersection_shader.ash_handle())
                        .name(intersection_name)
                        .build(),
                    ash::vk::PipelineShaderStageCreateInfo::builder()
                        .stage(ash::vk::ShaderStageFlags::MISS_KHR)
                        .module(miss_shader.ash_handle())
                        .name(miss_name)
                        .build(),
                    ash::vk::PipelineShaderStageCreateInfo::builder()
                        .stage(ash::vk::ShaderStageFlags::ANY_HIT_KHR)
                        .module(anyhit_shader.ash_handle())
                        .name(anyhit_name)
                        .build(),
                    ash::vk::PipelineShaderStageCreateInfo::builder()
                        .stage(ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                        .module(closesthit_shader.ash_handle())
                        .name(closesthit_name)
                        .build(),
                    ash::vk::PipelineShaderStageCreateInfo::builder()
                        .stage(ash::vk::ShaderStageFlags::CALLABLE_KHR)
                        .module(callable_shader.ash_handle())
                        .name(callable_name)
                        .build(),
                ];

                let create_info = ash::vk::RayTracingPipelineCreateInfoKHR::builder()
                    .layout(pipeline_layout.ash_handle())
                    .stages(stages_create_info.as_slice())
                    .max_pipeline_ray_recursion_depth(max_pipeline_ray_recursion_depth)
                    .build();

                match unsafe {
                    raytracing_ext.create_ray_tracing_pipelines(
                        ash::vk::DeferredOperationKHR::null(),
                        ash::vk::PipelineCache::null(),
                        &[create_info],
                        device.get_parent_instance().get_alloc_callbacks()
                    )
                } {
                    Ok(pipelines) => {
                        let pipeline = pipelines[0];

                        if pipelines.len() != 1 {
                            #[cfg(debug_assertions)]
                            {
                                panic!("Error creating the compute pipeline: expected 1 pipeline to be created, instead {} were created.", pipelines.len())
                            }

                            return Err(VulkanError::Unspecified);
                        }

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

                                    match ext.set_debug_utils_object_name(
                                        device.ash_handle().handle(),
                                        &dbg_info,
                                    ) {
                                        Ok(_) => {
                                            #[cfg(debug_assertions)]
                                            {
                                                println!("Pipeline Debug object name changed");
                                            }
                                        }
                                        Err(err) => {
                                            #[cfg(debug_assertions)]
                                            {
                                                panic!("Error setting the Debug name for the newly created Pipeline, will use handle. Error: {}", err)
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        Ok(Arc::new(Self {
                            device,
                            pipeline_layout,
                            pipeline,
                            max_pipeline_ray_recursion_depth
                        }))
                    },
                    Err(_err) => {
                        Err(VulkanError::Unspecified)
                    }
                }
            },
            None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_ray_tracing_pipeline",
            )))
        }
    }
}
