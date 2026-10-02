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
use crate::prelude::{FrameworkError, VulkanError, VulkanResult};

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

    pub(crate) fn shader_group_create_infos(
        intersection_stage: Option<u32>,
        any_hit_stage: Option<u32>,
        callable_stage: Option<u32>,
    ) -> smallvec::SmallVec<[ash::vk::RayTracingShaderGroupCreateInfoKHR<'static>; 6]> {
        let unused = ash::vk::RayTracingShaderGroupCreateInfoKHR::default()
            .general_shader(ash::vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(ash::vk::SHADER_UNUSED_KHR)
            .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
            .intersection_shader(ash::vk::SHADER_UNUSED_KHR);
        let mut groups = smallvec::smallvec![
            unused
                .ty(ash::vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0),
            unused
                .ty(ash::vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(1),
            unused
                .ty(ash::vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .closest_hit_shader(2),
        ];
        if let Some(stage) = intersection_stage {
            groups.push(
                unused
                    .ty(ash::vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                    .intersection_shader(stage),
            );
        }
        if let Some(stage) = any_hit_stage {
            groups.push(
                unused
                    .ty(ash::vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                    .any_hit_shader(stage),
            );
        }
        if let Some(stage) = callable_stage {
            groups.push(
                unused
                    .ty(ash::vk::RayTracingShaderGroupTypeKHR::GENERAL)
                    .general_shader(stage),
            );
        }
        groups
    }

    /// Group order is raygen, miss, closest-hit, optional intersection, optional
    /// any-hit, and optional callable. Optional hit shaders form separate groups;
    /// their SBT record indices follow the closest-hit record at index zero.
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

        for shader_device in [
            Some(raygen_shader.get_parent_device()),
            Some(miss_shader.get_parent_device()),
            Some(closesthit_shader.get_parent_device()),
            maybe_intersection_shader
                .as_ref()
                .map(|shader| shader.get_parent_device()),
            maybe_anyhit_shader
                .as_ref()
                .map(|shader| shader.get_parent_device()),
            maybe_callable_shader
                .as_ref()
                .map(|shader| shader.get_parent_device()),
        ]
        .into_iter()
        .flatten()
        {
            if shader_device != device {
                return Err(FrameworkError::ResourceFromIncompatibleDevice.into());
            }
        }
        let info = device
            .ray_tracing_info()
            .as_ref()
            .ok_or_else(|| VulkanError::MissingExtension("VK_KHR_ray_tracing_pipeline".into()))?;
        if max_pipeline_ray_recursion_depth > info.max_ray_recursion_depth() {
            return Err(ash::vk::Result::ERROR_INITIALIZATION_FAILED.into());
        }

        match device.ash_ext_raytracing_pipeline_khr() {
            Some(raytracing_ext) => {
                let mut stages_create_info: smallvec::SmallVec<
                    [ash::vk::PipelineShaderStageCreateInfo; 8],
                > = smallvec::smallvec![];

                stages_create_info.push(
                    ash::vk::PipelineShaderStageCreateInfo::default()
                        .stage(ash::vk::ShaderStageFlags::RAYGEN_KHR)
                        .module(raygen_shader.ash_handle())
                        .name(main_name),
                );

                stages_create_info.push(
                    ash::vk::PipelineShaderStageCreateInfo::default()
                        .stage(ash::vk::ShaderStageFlags::MISS_KHR)
                        .module(miss_shader.ash_handle())
                        .name(main_name),
                );

                stages_create_info.push(
                    ash::vk::PipelineShaderStageCreateInfo::default()
                        .stage(ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                        .module(closesthit_shader.ash_handle())
                        .name(main_name),
                );

                let intersection_stage = maybe_intersection_shader.as_ref().map(|shader| {
                    let index = stages_create_info.len() as u32;
                    stages_create_info.push(
                        ash::vk::PipelineShaderStageCreateInfo::default()
                            .stage(ash::vk::ShaderStageFlags::INTERSECTION_KHR)
                            .module(shader.ash_handle())
                            .name(main_name),
                    );
                    index
                });

                let any_hit_stage = maybe_anyhit_shader.as_ref().map(|shader| {
                    let index = stages_create_info.len() as u32;
                    stages_create_info.push(
                        ash::vk::PipelineShaderStageCreateInfo::default()
                            .stage(ash::vk::ShaderStageFlags::ANY_HIT_KHR)
                            .module(shader.ash_handle())
                            .name(main_name),
                    );
                    index
                });

                let callable_stage = maybe_callable_shader.as_ref().map(|shader| {
                    let index = stages_create_info.len() as u32;
                    stages_create_info.push(
                        ash::vk::PipelineShaderStageCreateInfo::default()
                            .stage(ash::vk::ShaderStageFlags::CALLABLE_KHR)
                            .module(shader.ash_handle())
                            .name(main_name),
                    );
                    index
                });
                let callable_shader_present = callable_stage.is_some();
                let shader_group_create_info = Self::shader_group_create_infos(
                    intersection_stage,
                    any_hit_stage,
                    callable_stage,
                );

                let create_info = ash::vk::RayTracingPipelineCreateInfoKHR::default()
                    .layout(pipeline_layout.ash_handle())
                    .stages(stages_create_info.as_slice())
                    .max_pipeline_ray_recursion_depth(max_pipeline_ray_recursion_depth)
                    .groups(shader_group_create_info.as_slice());

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
                        if let Some(ext) = device.ash_ext_debug_utils_ext() {
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
                                    let dbg_info = ash::vk::DebugUtilsObjectNameInfoEXT::default()
                                        .object_handle(pipeline)
                                        .object_name(object_name);

                                    if let Err(err) = ext.set_debug_utils_object_name(&dbg_info) {
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
                    Err((pipelines, err)) => {
                        for pipeline in pipelines {
                            if pipeline != ash::vk::Pipeline::null() {
                                unsafe {
                                    device.ash_handle().destroy_pipeline(
                                        pipeline,
                                        device.get_parent_instance().get_alloc_callbacks(),
                                    )
                                };
                            }
                        }
                        Err(err.into())
                    }
                }
            }
            None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_ray_tracing_pipeline",
            ))),
        }
    }
}
