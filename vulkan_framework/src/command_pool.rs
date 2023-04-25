use std::sync::Arc;

use crate::{queue_family::{QueueFamilyOwned, QueueFamily}, prelude::{VulkanResult, VulkanError}, device::DeviceOwned, instance::InstanceOwned};

pub struct CommandPool {
    queue_family: Arc<QueueFamily>,
    command_pool: ash::vk::CommandPool,
}

pub trait CommandPoolOwned {
    fn get_parent_command_pool(&self) -> Arc<CommandPool>;
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        let device = self.queue_family.get_parent_device();
        unsafe {
            device.ash_handle().destroy_command_pool(self.command_pool, device.get_parent_instance().get_alloc_callbacks())
        }
    }
}

impl QueueFamilyOwned for CommandPool {
    fn get_parent_queue_family(&self) -> std::sync::Arc<crate::queue_family::QueueFamily> {
        self.queue_family.clone()
    }
}

impl CommandPool {
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.command_pool)
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::CommandPool {
        self.command_pool
    }

    pub fn new(queue_family: Arc<QueueFamily>, debug_name: Option<&str>) -> VulkanResult<Arc<Self>> {
        let device = queue_family.get_parent_device();

        let create_info = ash::vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family.get_family_index())
            .flags(ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER | ash::vk::CommandPoolCreateFlags::TRANSIENT)
            .build();

        match unsafe { device.ash_handle().create_command_pool(&create_info, device.get_parent_instance().get_alloc_callbacks()) } {
            Ok(command_pool) => {
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
                                        .object_type(ash::vk::ObjectType::COMMAND_POOL)
                                        .object_handle(ash::vk::Handle::as_raw(command_pool))
                                        .object_name(object_name)
                                        .build();

                                    match ext.set_debug_utils_object_name(
                                        device.ash_handle().handle(),
                                        &dbg_info,
                                    ) {
                                        Ok(_) => {
                                            #[cfg(debug_assertions)]
                                            {
                                                println!("Command Pool Debug object name changed");
                                            }
                                        }
                                        Err(err) => {
                                            #[cfg(debug_assertions)]
                                            {
                                                println!("Error setting the Debug name for the newly created Command Pool, will use handle. Error: {}", err);
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

                Ok(
                    Arc::new(
                        Self {
                            queue_family,
                            command_pool
                        }
                    )
                )
            },
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    println!("Error creating the compute pipeline: {}", err);
                    assert_eq!(true, false)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }

}