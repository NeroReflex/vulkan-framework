use std::sync::Arc;

use crate::{prelude::VulkanResult, raytracing_pipeline::RaytracingPipeline};

pub struct RaytracingBindingTables {

}

impl RaytracingBindingTables {
    pub fn new(
        raytracing_pipeline: RaytracingPipeline
    ) -> VulkanResult<Arc<Self>> {
        todo!()
    }
}
