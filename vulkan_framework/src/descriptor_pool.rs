use std::sync::Arc;

use crate::device::Device;

pub struct DescriptorPool {
    device: Arc<Device>,
    pool: ash::vk::DescriptorPool
}