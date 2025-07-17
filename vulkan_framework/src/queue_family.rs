use std::ops::Deref;

use std::sync::Arc;

#[cfg(feature = "better_mutex")]
use parking_lot::{const_mutex, Mutex};

#[cfg(not(feature = "better_mutex"))]
use std::sync::Mutex;

use crate::prelude::FrameworkError;
use crate::{
    device::{Device, DeviceOwned},
    prelude::{VulkanError, VulkanResult},
};

#[derive(/*Copy,*/ Clone)]
pub enum QueueFamilySupportedOperationType {
    Compute,
    Graphics,
    Transfer,
    Present(Arc<crate::surface::Surface>),
}

#[derive(Clone)]
pub struct ConcreteQueueFamilyDescriptor {
    supported_operations: Vec<QueueFamilySupportedOperationType>,
    queue_priorities: Vec<f32>,
}

impl ConcreteQueueFamilyDescriptor {
    pub fn new(
        supported_operations: &[QueueFamilySupportedOperationType],
        queue_priorities: &[f32],
    ) -> Self {
        Self {
            supported_operations: supported_operations.to_vec(),
            queue_priorities: queue_priorities.to_vec(),
        }
    }

    pub fn max_queues(&self) -> usize {
        self.queue_priorities.len()
    }

    pub fn get_queue_priorities(&self) -> &[f32] {
        self.queue_priorities.as_slice()
    }

    pub fn get_supported_operations(&self) -> &[QueueFamilySupportedOperationType] {
        self.supported_operations.as_slice()
    }
}

pub struct QueueFamily {
    device: Arc<Device>,
    descriptor: ConcreteQueueFamilyDescriptor,
    created_queues: Mutex<u64>,
    family_index: u32,
}

pub trait QueueFamilyOwned {
    fn get_parent_queue_family(&self) -> Arc<QueueFamily>;
}

impl DeviceOwned for QueueFamily {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Drop for QueueFamily {
    fn drop(&mut self) {
        // Nothing to be done here
    }
}

impl QueueFamily {
    pub(crate) fn get_family_index(&self) -> u32 {
        self.family_index
    }

    pub fn new(device: Arc<Device>, index_of_required_queue: usize) -> VulkanResult<Arc<Self>> {
        #[cfg(feature = "better_mutex")]
        let created_queues = const_mutex(0);

        #[cfg(not(feature = "better_mutex"))]
        let created_queues = Mutex::new(0);

        match device.move_out_queue_family(index_of_required_queue) {
            Ok((queue_family, descriptor)) => Ok(Arc::new(Self {
                device: device.clone(),
                descriptor,
                created_queues,
                family_index: queue_family,
            })),
            Err(err) => Err(err),
        }
    }

    pub(crate) fn move_out_queue(&self) -> VulkanResult<(u32, f32)> {
        #[cfg(feature = "better_mutex")]
        let created_queues = self.created_queues.lock();

        #[cfg(not(feature = "better_mutex"))]
        let created_queues = match self.created_queues.lock() {
            Ok(lock) => lock,
            Err(err) => {
                return Err(VulkanError::Framework(FrameworkError::MutexError(format!(
                    "{err}"
                ))))
            }
        };

        let created_queues_num = *(created_queues.deref());
        let total_number_of_queues = self.descriptor.queue_priorities.len();
        match created_queues_num < total_number_of_queues as u64 {
            true => Ok((
                created_queues_num as u32,
                self.descriptor.queue_priorities[created_queues_num as usize],
            )),
            false => Err(VulkanError::Framework(FrameworkError::TooManyQueues(
                created_queues_num as usize,
                total_number_of_queues,
            ))),
        }
    }
}
