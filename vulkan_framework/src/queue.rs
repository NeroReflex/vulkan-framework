use crate::{
    device::DeviceOwned,
    instance::InstanceOwned,
    prelude::{VulkanError, VulkanResult},
    queue_family::*,
};

use std::sync::Arc;

pub struct Queue {
    _name_bytes: Vec<u8>,
    queue_family: Arc<QueueFamily>,
    priority: f32,
    queue: ash::vk::Queue,
}

impl QueueFamilyOwned for Queue {
    fn get_parent_queue_family(&self) -> Arc<QueueFamily> {
        self.queue_family.clone()
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        // Nothing to be done here, seems like queues are not to be deleted... A real shame!
    }
}

impl Queue {
    pub fn get_priority(&self) -> f32 {
        self.priority
    }

    pub fn new(
        queue_family: Arc<QueueFamily>,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        match queue_family.move_out_queue() {
            Some((queue_index, priority)) => {
                let queue = unsafe {
                    queue_family
                        .get_parent_device()
                        .ash_handle()
                        .get_device_queue(queue_family.get_family_index(), queue_index)
                };

                let mut obj_name_bytes = vec![];

                let device = queue_family.get_parent_device();
                let instance = device.get_parent_instance();

                match instance.get_debug_ext_extension() {
                    Some(ext) => {
                        match debug_name {
                            Some(name) => {
                                for name_ch in name.as_bytes().iter() {
                                    obj_name_bytes.push(*name_ch);
                                }
                                obj_name_bytes.push(0x00);

                                unsafe {
                                    let object_name = std::ffi::CStr::from_bytes_with_nul_unchecked(
                                        obj_name_bytes.as_slice(),
                                    );
                                    // set device name for debugging
                                    let dbg_info = ash::vk::DebugUtilsObjectNameInfoEXT::builder()
                                        .object_type(ash::vk::ObjectType::QUEUE)
                                        .object_handle(ash::vk::Handle::as_raw(queue.clone()))
                                        .object_name(object_name)
                                        .build();

                                    match ext.set_debug_utils_object_name(
                                        device.ash_handle().handle(),
                                        &dbg_info,
                                    ) {
                                        Ok(_) => {
                                            #[cfg(debug_assertions)]
                                            {
                                                println!("Queue Debug object name changed");
                                            }
                                        }
                                        Err(err) => {
                                            #[cfg(debug_assertions)]
                                            {
                                                println!("Error setting the Debug name for the newly created Queue, will use handle. Error: {}", err);
                                                assert_eq!(true, false);
                                            }
                                        }
                                    }
                                }
                            }
                            None => {}
                        };
                    }
                    None => {}
                }

                Ok(Arc::new(Self {
                    _name_bytes: obj_name_bytes,
                    queue_family: queue_family,
                    priority: priority,
                    queue: queue,
                }))
            }
            None => {
                #[cfg(debug_assertions)]
                {
                    println!("Something bad happened while moving out a queue from the provided QueueFamily");
                    assert_eq!(true, false)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }
}
