use std::ffi::CStr;
use std::sync::Arc;

use crate::compute_shader::ComputeShader;
use crate::device::{Device, DeviceOwned};
use crate::instance::InstanceOwned;

use crate::pipeline_layout::{PipelineLayout, PipelineLayoutDependant};
use crate::prelude::{VulkanError, VulkanResult};
use crate::shader_trait::PrivateShaderTrait;

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
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.pipeline)
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::Pipeline {
        self.pipeline
    }

    pub fn new(
        pipeline_layout: Arc<PipelineLayout>,
        shader: Arc<ComputeShader>,
        shader_entry_name: Option<String>,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = pipeline_layout.get_parent_device();

        let name: &CStr = match shader_entry_name {
            Option::Some(_n) => {
                todo!()
            }
            Option::None => {
                unsafe { CStr::from_bytes_with_nul_unchecked(&[109u8, 97u8, 105u8, 110u8, 0u8]) }
                // main
            }
        };

        let shader_stage_info = ash::vk::PipelineShaderStageCreateInfo::builder()
            .stage(ash::vk::ShaderStageFlags::COMPUTE)
            .module(shader.ash_handle())
            .name(name)
            .build();

        let create_info = [ash::vk::ComputePipelineCreateInfo::builder()
            .layout(pipeline_layout.ash_handle())
            .stage(shader_stage_info)
            .build()];

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
                }))
            }
            Err((_, err)) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error creating the compute pipeline: {}", err)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }
}
