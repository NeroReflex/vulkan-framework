use std::sync::Arc;

use crate::{prelude::{VulkanResult, VulkanError}, raytracing_pipeline::RaytracingPipeline, device::DeviceOwned};

pub struct RaytracingBindingTables {

}

impl RaytracingBindingTables {
    pub fn new(
        raytracing_pipeline: RaytracingPipeline
    ) -> VulkanResult<Arc<Self>> {
        let device = raytracing_pipeline.get_parent_device();

        match device.ray_tracing_info() {
            Some(rt_info) => {
                
                todo!()
            },
            None => {
                Err(VulkanError::Unspecified)
            }
        }
    }
}
