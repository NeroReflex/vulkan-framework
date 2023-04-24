use std::sync::Arc;

use crate::descriptor_set_layout::DescriptorSetLayout;

use crate::device::{DeviceOwned, Device};
use crate::instance::InstanceOwned;
use crate::push_constant_range::PushConstanRange;
use crate::{prelude::VulkanResult};

pub struct ComputePipeline {
    device: Arc<Device>,
    descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>>,
    pipeline: ash::vk::Pipeline,
}

impl DeviceOwned for ComputePipeline {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe { self.device.ash_handle().destroy_pipeline(self.pipeline, self.device.get_parent_instance().get_alloc_callbacks()) }
    }
}

/*impl DescriptorSetLayoutsDependant for ComputePipeline {
    fn get_descriptor_set_layouts(&self) -> Arc<DescriptorSetLayout> {
        self.descriptor_set_layouts.clone()
    }
}*/

impl ComputePipeline {
    pub fn new(
        device: Arc<Device>,
        descriptor_set_layouts: &[Arc<DescriptorSetLayout>]
    ) -> VulkanResult<Arc<Self>> {


        todo!()
    }
}