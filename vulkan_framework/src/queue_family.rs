use ash::vk::Queue;

use crate::{
    device::{Device, DeviceOwned},
    result::VkError,
};

#[derive(/*Copy,*/ Clone)]
pub enum QueueFamilySupportedOperationType<'a> {
    Compute,
    Graphics,
    Transfer,
    Present(&'a crate::surface::Surface<'a>),
}

#[derive(Clone)]
pub struct ConcreteQueueFamilyDescriptor<'a> {
    supported_operations: Vec<QueueFamilySupportedOperationType<'a>>,
    queue_priorities: Vec<f32>,
}

impl<'a> ConcreteQueueFamilyDescriptor<'a> {
    pub fn new(
        supported_operations: Vec<QueueFamilySupportedOperationType<'a>>,
        queue_priorities: &[f32],
    ) -> Self {
        Self {
            supported_operations: supported_operations.iter().map(|el| el.clone()).collect(),
            queue_priorities: queue_priorities.iter().map(|a| a.clone()).collect(),
        }
    }

    pub fn max_queues(&self) -> usize {
        self.queue_priorities.len()
    }

    pub fn get_queue_priorities(&self) -> &[f32] {
        self.queue_priorities.as_slice()
    }

    pub fn get_supported_operations(&self) -> &[QueueFamilySupportedOperationType] {
        self.supported_operations.as_slice().as_ref()
    }
}

pub struct QueueFamily<'qf> {
    device: &'qf Device<'qf>,
    supported_queues: u64,
    created_queues: u64,
    family_index: u32,
}

impl<'qf> DeviceOwned<'qf> for QueueFamily<'qf> {
    fn get_parent_device(&self) -> &'qf Device {
        self.device.clone()
    }
}

impl<'qf> Drop for QueueFamily<'qf> {
    fn drop(&mut self) {
        // Nothing to be done here
    }
}

impl<'qf> QueueFamily<'qf> {
    pub fn new(device: &'qf Device<'qf>, index_of_required_queue: usize) -> Result<Self, VkError> {
        match device.move_out_queue_family(index_of_required_queue) {
            Some((queue_family, description)) => Ok(Self {
                device: device,
                supported_queues: description.max_queues() as u64,
                created_queues: 0,
                family_index: queue_family.clone(),
            }),
            None => {
                #[cfg(debug_assertions)]
                {
                    println!("Something bad happened while moving out the queue family from the provided Device");
                    assert_eq!(true, false)
                }

                Err(VkError {})
            }
        }
    }
}
