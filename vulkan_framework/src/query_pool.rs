use std::sync::Arc;

use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    prelude::VulkanResult,
};

/// A pool of GPU timestamp queries.
///
/// The pool must outlive every command buffer that writes into it: recorders
/// do not keep it alive.
pub struct TimestampQueryPool {
    device: Arc<Device>,
    pool: ash::vk::QueryPool,
    count: u32,
    period_ns: f64,
    valid_bits: u32,
}

impl Drop for TimestampQueryPool {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_query_pool(
                self.pool,
                self.device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

impl DeviceOwned for TimestampQueryPool {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl TimestampQueryPool {
    pub fn new(device: Arc<Device>, count: u32) -> VulkanResult<Arc<Self>> {
        let instance = device.get_parent_instance();
        let properties = unsafe {
            instance
                .ash_handle()
                .get_physical_device_properties(*device.ash_physical_device_handle())
        };
        let create_info = ash::vk::QueryPoolCreateInfo::default()
            .query_type(ash::vk::QueryType::TIMESTAMP)
            .query_count(count);
        let pool = unsafe {
            device
                .ash_handle()
                .create_query_pool(&create_info, instance.get_alloc_callbacks())
        }?;
        // Queries start in an undefined state: reset them on the host so that the
        // first read reports "not ready" instead of garbage.
        unsafe { device.ash_handle().reset_query_pool(pool, 0, count) };
        Ok(Arc::new(Self {
            device,
            pool,
            count,
            period_ns: properties.limits.timestamp_period as f64,
            valid_bits: 64,
        }))
    }

    #[inline]
    pub(crate) fn ash_handle(&self) -> ash::vk::QueryPool {
        self.pool
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    /// Nanoseconds per timestamp tick.
    pub fn period_ns(&self) -> f64 {
        self.period_ns
    }

    pub fn valid_bits(&self) -> u32 {
        self.valid_bits
    }

    /// Reads `count` timestamps starting at `first`. Returns `None` when any of
    /// them is not available yet; never blocks.
    pub fn read(&self, first: u32, count: u32) -> VulkanResult<Option<Vec<u64>>> {
        let mut data = vec![[0u64; 2]; count as usize];
        let result = unsafe {
            self.device.ash_handle().get_query_pool_results(
                self.pool,
                first,
                data.as_mut_slice(),
                ash::vk::QueryResultFlags::TYPE_64 | ash::vk::QueryResultFlags::WITH_AVAILABILITY,
            )
        };
        match result {
            Ok(()) | Err(ash::vk::Result::NOT_READY) => {}
            Err(err) => return Err(err.into()),
        }
        if data.iter().any(|pair| pair[1] == 0) {
            return Ok(None);
        }
        Ok(Some(data.into_iter().map(|pair| pair[0]).collect()))
    }
}
