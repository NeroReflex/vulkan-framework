use std::sync::Arc;

use crate::device::{Device, DeviceOwned};
use crate::instance::InstanceOwned;

use crate::pipeline_layout::{PipelineLayout, PipelineLayoutDependant};
use crate::prelude::{VulkanError, VulkanResult};

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
    pub fn new(pipeline_layout: Arc<PipelineLayout>) -> VulkanResult<Arc<Self>> {
        let device = pipeline_layout.get_parent_device();

        let create_info = [ash::vk::ComputePipelineCreateInfo::builder()
            .layout(pipeline_layout.ash_handle())
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
                if pipelines.len() != 1 {
                    #[cfg(debug_assertions)]
                    {
                        println!("Error creating the compute pipeline: expected 1 pipeline to be created, instead {} were created.", pipelines.len());
                        assert_eq!(true, false)
                    }

                    return Err(VulkanError::Unspecified);
                }

                Ok(Arc::new(Self {
                    device,
                    pipeline_layout,
                    pipeline: pipelines[0],
                }))
            }
            Err((_, err)) => {
                #[cfg(debug_assertions)]
                {
                    println!("Error creating the compute pipeline: {}", err);
                    assert_eq!(true, false)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }
}
