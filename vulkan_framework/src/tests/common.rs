use std::sync::Arc;

use crate::instance::Instance;
use crate::device::Device;
use crate::prelude::*;
use crate::queue_family::{ConcreteQueueFamilyDescriptor, QueueFamilySupportedOperationType};

/// Test helper: create a minimal Instance + Device suitable for headless tests.
pub fn setup_test_device() -> VulkanResult<(Arc<Instance>, Arc<Device>)> {
    // Create instance without requesting validation layers or surface extensions
    let instance = Instance::new(&[], &[], &"vulkan_framework_tests".to_string(), &"test".to_string())?;

    // Request a single compute-capable queue (no present/surface required)
    let queue_descriptor = ConcreteQueueFamilyDescriptor::new(
        &[QueueFamilySupportedOperationType::Compute],
        &[1.0],
    );

    let device = Device::new(instance.clone(), &[queue_descriptor], &[], Some("test_device"))?;

    Ok((instance, device))
}
