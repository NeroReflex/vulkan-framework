use std::sync::Arc;

use crate::device::Device;
use crate::instance::Instance;
use crate::prelude::*;
use crate::queue_family::{ConcreteQueueFamilyDescriptor, QueueFamilySupportedOperationType};

/// Test helper: create a minimal Instance + Device suitable for headless tests.
pub fn setup_test_device() -> VulkanResult<(Arc<Instance>, Arc<Device>)> {
    setup_test_device_with_queue_count(1)
}

pub fn setup_test_device_with_queue_count(
    queue_count: usize,
) -> VulkanResult<(Arc<Instance>, Arc<Device>)> {
    // Create instance without requesting validation layers or surface extensions
    let instance = Instance::new(
        &[],
        &[],
        &"vulkan_framework_tests".to_string(),
        &"test".to_string(),
    )?;

    // Device creation clamps this request to the family's available queue count.
    let priorities = vec![1.0; queue_count];
    let queue_descriptor = ConcreteQueueFamilyDescriptor::new(
        &[QueueFamilySupportedOperationType::Compute],
        &priorities,
    );

    let device = Device::new(
        instance.clone(),
        &[queue_descriptor],
        &[],
        Some("test_device"),
    )?;

    Ok((instance, device))
}
