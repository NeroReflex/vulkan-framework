use std::sync::Arc;

use vulkan_framework::{
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor,
        DescriptorPoolSizesAcceletarionStructureKHR, DescriptorPoolSizesConcreteDescriptor,
    },
    device::Device,
};

use crate::rendering::RenderingResult;

pub struct MeshManager {
    device: Arc<Device>,

    _descriptor_pool: Arc<DescriptorPool>,
    //descriptor_sets: smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
}

impl MeshManager {
    pub fn new(device: Arc<Device>, frames_in_flight: u32) -> RenderingResult<Self> {
        let descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    Some(DescriptorPoolSizesAcceletarionStructureKHR::new(
                        frames_in_flight,
                    )),
                ),
                frames_in_flight,
            ),
            Some("mesh_manager_descriptor_pool"),
        )?;

        Ok(Self {
            device,
            _descriptor_pool: descriptor_pool,
        })
    }
}
