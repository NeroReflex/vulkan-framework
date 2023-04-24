use std::sync::Arc;
use std::vec::Vec;

use crate::{device::{Device, DeviceOwned}, instance::InstanceOwned, prelude::{VulkanResult, VulkanError}, push_constant_range::PushConstanRange, descriptor_set_layout::DescriptorSetLayout};

pub struct PipelineLayout {
    device: Arc<Device>,
    layout_bindings: Vec<Arc<DescriptorSetLayout>>,
    push_constant_ranges: Vec<Arc<PushConstanRange>>,
    pipeline_layout: ash::vk::PipelineLayout,
}

pub trait PipelineLayoutDependant {
    fn get_parent_pipeline_layout(&self) -> Arc<PipelineLayout>;
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe { self.device.ash_handle().destroy_pipeline_layout(self.pipeline_layout, self.device.get_parent_instance().get_alloc_callbacks()) }
    }
}

impl DeviceOwned for PipelineLayout {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl PipelineLayout {
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.pipeline_layout)
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn new(
        device: Arc<Device>,
        binding_descriptors: &[Arc<DescriptorSetLayout>],
        constant_ranges: &[Arc<PushConstanRange>],
    ) -> VulkanResult<Arc<Self>> {
        let set_layouts = binding_descriptors.iter().map(
            |layout_binding| {
                // TODO: make sure all of these are from the same device
                //assert_eq!(layout_binding.get_parent_device(), device);

                layout_binding.ash_handle()
            }
        ).collect::<Vec<ash::vk::DescriptorSetLayout>>();

        let ranges = constant_ranges.iter().map(
            |r| {
                r.ash_handle()
            }
        ).collect::<Vec<ash::vk::PushConstantRange>>();

        let create_info = ash::vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(set_layouts.as_slice())
            .push_constant_ranges(ranges.as_slice())
            .build();

        match unsafe { device.ash_handle().create_pipeline_layout(&create_info, device.get_parent_instance().get_alloc_callbacks()) } {
            Ok(pipeline_layout) => {
                Ok(
                    Arc::new(
                        Self {
                            device,
                            pipeline_layout,
                            layout_bindings: binding_descriptors.iter().map(|layout_binding| layout_binding.clone()).collect::<Vec<Arc<DescriptorSetLayout>>>(),
                            push_constant_ranges: constant_ranges.iter().map(|r| r.clone()).collect::<Vec<Arc<PushConstanRange>>>()
                        }
                    )
                )
            },
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    println!("Error creating the pipeline layout: {}", err);
                    assert_eq!(true, false)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }
}