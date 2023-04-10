use std::{sync::Mutex, ops::Deref};

use ash::vk::Queue;

use crate::{
    device::{Device, DeviceOwned},
    result::VkError,
};

#[derive(/*Copy,*/ Clone)]
pub enum QueueFamilySupportedOperationType<'ctx, 'instance, 'surface> {
    Compute,
    Graphics,
    Transfer,
    Present(&'surface crate::surface::Surface<'ctx, 'instance>),
}

#[derive(Clone)]
pub struct ConcreteQueueFamilyDescriptor<'ctx, 'instance, 'surface> {
    supported_operations: Vec<QueueFamilySupportedOperationType<'ctx, 'instance, 'surface>>,
    queue_priorities: Vec<f32>,
}

impl<'ctx, 'instance, 'surface> ConcreteQueueFamilyDescriptor<'ctx, 'instance, 'surface> {
    pub fn new(
        supported_operations: &[QueueFamilySupportedOperationType<'ctx, 'instance, 'surface>],
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

    pub fn get_supported_operations(&self) -> &[QueueFamilySupportedOperationType<'ctx, 'instance, 'surface>] {
        self.supported_operations.as_slice()
    }
}

pub struct QueueFamily<'ctx, 'instance, 'device> {
    device: &'device Device<'ctx, 'instance>,
    descriptor: Vec<f32>,
    created_queues: Mutex<u64>,
    family_index: u32,
}

pub(crate) trait QueueFamilyOwned<'ctx, 'instance, 'device> {
    fn get_parent_queue_family(&self) -> &QueueFamily<'ctx, 'instance, 'device>;
}

impl<'ctx, 'instance, 'device> DeviceOwned<'instance> for QueueFamily<'ctx, 'instance, 'device> {
    fn get_parent_device(&self) -> &'device Device<'ctx, 'instance> {
        self.device.clone()
    }
}

impl<'ctx, 'instance, 'device> Drop for QueueFamily<'ctx, 'instance, 'device> {
    fn drop(&mut self) {
        // Nothing to be done here
    }
}

impl<'ctx, 'instance, 'device> QueueFamily<'ctx, 'instance, 'device> {
    pub(crate) fn get_family_index(&self) -> u32 {
        self.family_index
    }

    pub fn new(device: &'device Device<'ctx, 'instance>, index_of_required_queue: usize) -> Result<Self, VkError>
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

    pub(crate) fn move_out_queue(&self) -> Option<(u32, f32)> {
        match self.created_queues.lock() {
            Ok(created_queues) => {
                let created_queues_num = *(created_queues.deref());
                match created_queues_num < self.descriptor.len() as u64 {
                    true => {
                        let priority: f32 = self.descriptor[created_queues_num as usize];

                        Some((created_queues_num as u32, priority))
                    },
                    false => {
                        #[cfg(debug_assertions)]
                        {
                            println!("From this QueueFamily the number of created Queue(s) is {} out of a maximum supported number of {} has already been created.", created_queues_num, self.descriptor.len());
                            assert_eq!(true, false)
                        }

                        Option::None
                    }
                }
            },
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    println!("Error acquiring internal mutex: {}", err);
                    assert_eq!(true, false)
                }

                Option::None
            }
        }
    }
}
