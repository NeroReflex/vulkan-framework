use std::ffi::CStr;
use std::sync::Arc;

use crate::device::{Device, DeviceOwned};
use crate::instance::InstanceOwned;

use crate::pipeline_layout::{PipelineLayout, PipelineLayoutDependant};
use crate::prelude::{VulkanError, VulkanResult};
use crate::shader_trait::PrivateShaderTrait;
use crate::shaders::compute_shader::ComputeShader;

pub struct ComputePipeline {
    device: Arc<Device>,
    pipeline_layout: Arc<PipelineLayout>,
    pipeline: ash::vk::Pipeline,
}

impl PipelineLayoutDependant for ComputePipeline {
    fn get_parent_pipeline_layout(&self) -> Arc<PipelineLayout> {
        self.pipeline_layout.clone()
    }
}

impl DeviceOwned for ComputePipeline {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_pipeline(
                self.pipeline,
                self.device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

/*impl DescriptorSetLayoutsDependant for ComputePipeline {
    fn get_descriptor_set_layouts(&self) -> Arc<DescriptorSetLayout> {
        self.descriptor_set_layouts.clone()
    }
}*/

impl ComputePipeline {
    #[inline]
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.pipeline)
    }

    #[inline]
    pub(crate) fn ash_handle(&self) -> ash::vk::Pipeline {
        self.pipeline
    }

    pub fn new(
        base_pipeline: Option<Arc<ComputePipeline>>,
        pipeline_layout: Arc<PipelineLayout>,
        compute_shader: (Arc<ComputeShader>, Option<String>),
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = pipeline_layout.get_parent_device();

        let (shader, shader_entry_name) = compute_shader;

        let name: &CStr = match shader_entry_name {
            Option::Some(_n) => {
                todo!()
            }
            Option::None => {
                unsafe { CStr::from_bytes_with_nul_unchecked(&[109u8, 97u8, 105u8, 110u8, 0u8]) }
                // main
            }
        };

        let shader_stage_info = ash::vk::PipelineShaderStageCreateInfo::default()
            .stage(ash::vk::ShaderStageFlags::COMPUTE)
            .module(shader.ash_handle())
            .name(name);

        let create_info = [ash::vk::ComputePipelineCreateInfo::default()
            .layout(pipeline_layout.ash_handle())
            .stage(shader_stage_info)
            .base_pipeline_handle(match &base_pipeline {
                Some(old_pipeline) => old_pipeline.ash_handle(),
                None => ash::vk::Pipeline::null(),
            })];

        match unsafe {
            pipeline_layout
                .get_parent_device()
                .ash_handle()
                .create_compute_pipelines(
                    ash::vk::Handle::from_raw(0),
                    create_info.as_slice(),
                    device.get_parent_instance().get_alloc_callbacks(),
                )
        } {
            Ok(pipelines) => {
                drop(base_pipeline);

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

                            if let Err(err) = ext.set_debug_utils_object_name(&dbg_info,
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
                }))
            }
            Err((_, err)) => Err(VulkanError::Vulkan(
                err.as_raw(),
                Some(format!("Error creating the compute pipeline: {}", err)),
            )),
        }
    }
}
