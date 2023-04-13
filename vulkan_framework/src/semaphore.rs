use std::{ops::Deref, sync::Mutex};

use ash::vk::Queue;

use std::sync::Arc;

use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    prelude::{VulkanError, VulkanResult},
};

pub struct Semaphore {
    device: Arc<Device>,
    semaphore: ash::vk::Semaphore,
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_semaphore(
                self.semaphore.clone(),
                self.device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

impl DeviceOwned for Semaphore {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Semaphore {
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.semaphore.clone())
    }

    pub fn new(
        device: Arc<Device>,
        starts_in_signaled_state: bool,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let create_info = ash::vk::SemaphoreCreateInfo::builder()
            .flags(ash::vk::SemaphoreCreateFlags::empty()) // At  the time of writing reserved for future use
            .build();

        match unsafe {
            device.ash_handle().create_semaphore(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        } {
            Ok(semaphore) => {
                let mut obj_name_bytes = vec![];
                match device.get_parent_instance().get_debug_ext_extension() {
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
                                        .object_type(ash::vk::ObjectType::SEMAPHORE)
                                        .object_handle(ash::vk::Handle::as_raw(semaphore.clone()))
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

                Ok(Arc::new(Self { device, semaphore }))
            }
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    println!("Error creating the fence: {}", err);
                }

                Err(VulkanError::Unspecified)
            }
        }
    }
}
