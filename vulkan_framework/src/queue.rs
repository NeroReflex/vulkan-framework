use crate::{
    queue_family::*,
    result::VkError, device::DeviceOwned,
};

pub struct Queue<'ctx, 'instance, 'device, 'queue_family> {
    queue_family: &'queue_family QueueFamily<'ctx, 'instance, 'device>,
    priority: f32,
    queue: ash::vk::Queue,
}

impl<'ctx, 'instance, 'device, 'queue_family> QueueFamilyOwned<'ctx, 'instance, 'device> for Queue<'ctx, 'instance, 'device, 'queue_family> {
    fn get_parent_queue_family(&self) -> &'queue_family QueueFamily<'ctx, 'instance, 'device> {
        self.queue_family
    }
}

impl<'ctx, 'instance, 'device, 'queue_family> Drop for Queue<'ctx, 'instance, 'device, 'queue_family> {
    fn drop(&mut self) {
        // Nothing to be done here, seems like queues are not to be deleted... A real shame!
    }
}

impl<'ctx, 'instance, 'device, 'queue_family> Queue<'ctx, 'instance, 'device, 'queue_family> {
    pub fn get_priority(&self) -> f32 {
        self.priority
    }
    
    pub fn new(queue_family: &'queue_family QueueFamily<'ctx, 'instance, 'device>) -> Result<Self, VkError> {
        match queue_family.move_out_queue() {
            Some((queue_index, priority)) => {
                let queue = unsafe {
                    queue_family.get_parent_device().ash_handle().get_device_queue(queue_family.get_family_index(), queue_index)
                };

                Ok(Self{
                    queue_family: queue_family,
                    priority: priority,
                    queue: queue
                })
            },
            None => {
                #[cfg(debug_assertions)]
                {
                    println!("Something bad happened while moving out a queue from the provided QueueFamily");
                    assert_eq!(true, false)
                }

                Err(VkError {})
            }
        }
    }


}