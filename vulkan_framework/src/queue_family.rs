use std::sync::{Arc, Weak, Mutex};

use ash::vk::Queue;

use crate::{device::{Device, DeviceOwned}, result::VkError};

#[derive(/*Copy,*/ Clone)]
pub enum QueueFamilySupportedOperationType {
    Compute,
    Graphics,
    Transfer,
    Present(Weak<Mutex<crate::surface::Surface>>),
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
            supported_operations: supported_operations.iter().map(|el| el.clone()).collect(),
            queue_priorities: queue_priorities.iter().map(|a| a.clone()).collect(),
        }
    }

    pub fn max_queues(&self) -> u32 {
        self.queue_priorities.len() as u32
    }

    pub fn get_queue_priorities(&self) -> &[f32] {
        self.queue_priorities.as_slice()
    }

    pub fn get_supported_operations(&self) -> &[QueueFamilySupportedOperationType] {
        self.supported_operations.as_slice().as_ref()
    }
}

pub struct QueueFamily {
    device: Weak<Mutex<Device>>,
    supported_queues: u64,
    created_queues: u64,
    family_index: u32,
}

impl crate::device::DeviceOwned for QueueFamily {
    fn get_parent_device(&self) -> Weak<Mutex<crate::device::Device>> {
        self.device.clone()
    }
}

impl Drop for QueueFamily {
    fn drop(&mut self) {
        match self.get_parent_device().upgrade() {
            Some(device_mutex) => {
                match device_mutex.lock() {
                    Ok(_device) => {
                        // Nothing to be done here
                    },
                    Err(err) => {
                        #[cfg(debug_assertions)]
                        {
                            println!("Cannot acquire Device mutex: {}.", err);
                            assert_eq!(true, false)
                        }
                    }
                }
                
            }
            None => {
                #[cfg(debug_assertions)]
                {
                    println!("Parent Device has already been deleted.");
                    assert_eq!(true, false)
                }
            }
        }
    }
}

impl QueueFamily {
    pub fn new(device_weak_ptr: Weak<Mutex<Device>>, index_of_required_queue: usize) -> Result<Arc<Mutex<Self>>, VkError> {
        todo!()
    }
}