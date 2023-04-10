use std::sync::Mutex;

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

    pub fn get_supported_operations(&self) -> &[QueueFamilySupportedOperationType<'a>] {
        self.supported_operations.as_slice()
    }
}

pub struct QueueFamily<'qf> {
    device: &'qf Device<'qf>,
    descriptor: ConcreteQueueFamilyDescriptor<'qf>,
    created_queues: Mutex<u64>,
    family_index: u32,
}

pub(crate) trait QueueFamilyOwned<'qf> {
    fn get_parent_queue_family(&self) -> &'qf QueueFamily<'qf>;
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
    pub fn new<'device>(device: &'device Device<'qf>, index_of_required_queue: usize) -> Result<QueueFamily<'device>, VkError>
    where
        'device: 'qf
    {
        match device.move_out_queue_family(index_of_required_queue) {
            Some((queue_family, description)) => Ok(Self {
                device: device.clone(),
                descriptor: description,
                created_queues: Mutex::new(0),
                family_index: queue_family,
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
